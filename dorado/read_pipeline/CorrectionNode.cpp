#include "CorrectionNode.h"

#include "correct/conversions.h"
#include "correct/decode.h"
#include "correct/features.h"
#include "correct/infer.h"
#include "correct/windows.h"
#include "utils/bam_utils.h"
#include "utils/gpu_profiling.h"
#include "utils/sequence_utils.h"
#include "utils/string_utils.h"
#include "utils/types.h"
#if DORADO_CUDA_BUILD
#include "utils/cuda_utils.h"
#endif
#include "hts_io/FastxRandomReader.h"

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <ATen/Tensor.h>
#include <htslib/faidx.h>
#include <htslib/sam.h>
#include <minimap.h>
#include <spdlog/spdlog.h>
#include <torch/script.h>

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace dorado::correction;

namespace {

dorado::BamPtr create_bam_record(const std::string& read_id, const std::string& seq) {
    bam1_t* rec = bam_init1();
    bam_set1(rec, read_id.length(), read_id.c_str(), 4 /*flag*/, -1 /*tid*/, -1 /*pos*/, 0 /*mapq*/,
             0 /*n_cigar*/, nullptr /*cigar*/, -1 /*mtid*/, -1 /*mpos*/, 0 /*isize*/, seq.size(),
             seq.data(), nullptr, 0);
    return dorado::BamPtr(rec);
}

std::vector<dorado::CigarOp> parse_cigar(const uint32_t* cigar, uint32_t n_cigar) {
    std::vector<dorado::CigarOp> cigar_ops;
    cigar_ops.resize(n_cigar);
    for (uint32_t i = 0; i < n_cigar; i++) {
        uint32_t op = cigar[i] & 0xf;
        uint32_t len = cigar[i] >> 4;
        if (op == MM_CIGAR_MATCH) {
            cigar_ops[i] = {dorado::CigarOpType::MATCH, len};
        } else if (op == MM_CIGAR_INS) {
            cigar_ops[i] = {dorado::CigarOpType::INS, len};
        } else if (op == MM_CIGAR_DEL) {
            cigar_ops[i] = {dorado::CigarOpType::DEL, len};
        } else {
            throw std::runtime_error("Unknown cigar op: " + std::to_string(op));
        }
    }
    return cigar_ops;
}

bool populate_alignments(dorado::CorrectionAlignments& alignments,
                         dorado::hts_io::FastxRandomReader* reader) {
    const auto& tname = alignments.read_name;
    alignments.read_seq = reader->fetch_seq(tname);
    alignments.read_qual = reader->fetch_qual(tname);
    int tlen = (int)alignments.read_seq.length();
    auto num_qnames = alignments.qnames.size();
    alignments.seqs.resize(num_qnames);
    alignments.quals.resize(num_qnames);
    alignments.cigars.resize(num_qnames);

    // In some cases the target read length reported by mm2 has differed from the
    // read length when loaded from the fastq. So we check that here and skip
    // any alignments where information is inconsisteny.
    // TODO: This was mainly observed before a bug fix for proper loading
    // of split mm2 indices was added. However the check is being kept around
    // for now, and can be removed later.
    std::vector<size_t> pos_to_remove;
    for (size_t i = 0; i < num_qnames; i++) {
        const std::string& qname = alignments.qnames[i];
        alignments.seqs[i] = reader->fetch_seq(qname);
        if ((int)alignments.seqs[i].length() != alignments.overlaps[i].qlen) {
            spdlog::error("qlen from before {} and qlen from after {} don't match for {}",
                          alignments.overlaps[i].qlen, alignments.seqs[i].length(), qname);
            return false;
        }
        alignments.quals[i] = reader->fetch_qual(qname);
        if (alignments.overlaps[i].tlen != tlen) {
            spdlog::error("tlen from before {} and tlen from after {} don't match for {}",
                          alignments.overlaps[i].tlen, tlen, tname);
            return false;
        }
        alignments.cigars[i] = parse_cigar(alignments.mm2_cigars[i].data(),
                                           (uint32_t)alignments.mm2_cigars[i].size());
        alignments.mm2_cigars[i] = {};
    }

    return alignments.check_consistent_overlaps();
}

std::vector<std::string> concatenate_corrected_windows(const std::vector<std::string>& cons) {
    std::vector<std::string> corrected_seqs;

    std::string corrected_seq = "";

    for (const auto& s : cons) {
        if (s.empty()) {
            if (!corrected_seq.empty()) {
                corrected_seqs.push_back(std::move(corrected_seq));
                corrected_seq = "";
            }
        } else {
            corrected_seq += s;
        }
    }
    if (!corrected_seq.empty()) {
        corrected_seqs.push_back(std::move(corrected_seq));
    }
    return corrected_seqs;
}

}  // namespace

namespace dorado {

void CorrectionNode::concat_features_and_send(const std::vector<std::string>& to_decode,
                                              const std::string& read_name) {
    LOG_TRACE("decoding window for {}", read_name);
    auto corrected_seqs = concatenate_corrected_windows(to_decode);
    if (corrected_seqs.size() == 1) {
        BamMessage rec{create_bam_record(read_name, corrected_seqs[0]), nullptr};
        send_message_to_sink(std::move(rec));
    } else {
        for (size_t s = 0; s < corrected_seqs.size(); s++) {
            const std::string new_name = read_name + ":" + std::to_string(s);
            BamMessage rec{create_bam_record(new_name, corrected_seqs[s]), nullptr};
            send_message_to_sink(std::move(rec));
        }
    }
}

void CorrectionNode::decode_fn() {
    spdlog::debug("Starting decode thread!");

    WindowFeatures item;
    while (m_inferred_features_queue.try_pop(item) != utils::AsyncQueueStatus::Terminate) {
        utils::ScopedProfileRange spr("decode_loop", 1);
        auto read_name = item.read_name;
        std::vector<std::string> to_decode;
        auto pos = item.window_idx;
        auto corrected_seq = decode_window(item);
        {
            std::lock_guard<std::mutex> lock(m_features_mutex);
            auto find_iter = m_features_by_id.find(read_name);
            if (find_iter == m_features_by_id.end()) {
                spdlog::error("Decoded feature list not found for {}.", read_name);
                continue;
            }
            auto& output_features = find_iter->second;
            output_features[pos] = std::move(corrected_seq);
            auto& pending = m_pending_features_by_id.find(read_name)->second;
            pending--;
            if (pending == 0) {
                // Got all features!
                to_decode = std::move(output_features);
                m_features_by_id.erase(read_name);
                m_pending_features_by_id.erase(read_name);
            }
        }

        if (!to_decode.empty()) {
            concat_features_and_send(to_decode, read_name);
        }
    }
}

void CorrectionNode::infer_fn(const std::string& device_str, int mtx_idx, int batch_size) {
    spdlog::debug("Starting process thread for {}!", device_str);
    m_num_active_infer_threads++;

    torch::Device device = torch::Device(device_str);

#if DORADO_CUDA_BUILD
    c10::optional<c10::Stream> stream;
    if (device.is_cuda()) {
        stream = c10::cuda::getStreamFromPool(false, device.index());
    }
    c10::cuda::OptionalCUDAStreamGuard guard(stream);
#endif

    at::InferenceMode infer_guard;

    auto model_path = (m_model_config.model_dir / m_model_config.weights_file).string();
    torch::jit::script::Module module;
    try {
        spdlog::debug("Loading model on {}...", device_str);
        module = torch::jit::load(model_path, device);
        spdlog::debug("Loaded model on {}!", device_str);
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model from " + model_path +
                                 " with error: " + e.what());
    }
    module.eval();

    std::vector<at::Tensor> bases_batch;
    std::vector<at::Tensor> quals_batch;
    std::vector<int> lengths;
    std::vector<int64_t> sizes;
    std::vector<at::Tensor> indices_batch;
    std::vector<WindowFeatures> wfs;
    // If there are any windows > 5120, then reduce batch size by 1
    int remaining_batch_slots = batch_size;

    auto decode_preds = [](const at::Tensor& preds) {
        std::vector<char> bases;
        bases.reserve(preds.sizes()[0]);
        static std::array<char, 5> decoder = {'A', 'C', 'G', 'T', '*'};
        for (int i = 0; i < preds.sizes()[0]; i++) {
            auto base_idx = preds[i].item<int>();
            bases.push_back(decoder[base_idx]);
        }
        return bases;
    };

    auto batch_infer = [&]() {
        utils::ScopedProfileRange infer("infer", 1);
        // Run inference on batch
        auto length_tensor =
                at::from_blob(lengths.data(), {(int)lengths.size()},
                              at::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
        auto batched_bases = collate<int>(bases_batch, (int)11, torch::kInt32);
        auto batched_quals = collate<float>(quals_batch, 0.f, torch::kFloat32);

        std::unique_lock<std::mutex> lock(m_gpu_mutexes[mtx_idx]);
        std::vector<torch::jit::IValue> inputs;
        {
            utils::ScopedProfileRange move_to_device("move_to_device", 1);
            inputs.push_back(batched_bases.to(device));
            inputs.push_back(batched_quals.to(device));
            inputs.push_back(length_tensor.to(device));
            std::for_each(indices_batch.begin(), indices_batch.end(),
                          [device](at::Tensor& t) { t.to(device); });
            inputs.push_back(indices_batch);
        }

        c10::IValue output;
        try {
            output = module.forward(inputs);
        } catch (std::runtime_error& e) {
#if DORADO_CUDA_BUILD
            spdlog::warn("Caught Torch error '{}', clearing CUDA cache and retrying.", e.what());
            c10::cuda::CUDACachingAllocator::emptyCache();
            output = module.forward(inputs);
#else
            throw e;
#endif
        }
        lock.unlock();
        if (!output.isTuple()) {
            throw std::runtime_error("Expected inference result to be tuple.");
        }
        auto base_logits = output.toTuple()->elements()[1].toTensor();
        auto preds = base_logits.argmax(1, false).to(torch::kCPU);
        auto split_preds = preds.split_with_sizes(sizes);
        for (size_t w = 0; w < split_preds.size(); w++) {
            auto decoded_output = decode_preds(split_preds[w]);
            wfs[w].inferred_bases = decoded_output;
        }

        for (auto& wf : wfs) {
            m_inferred_features_queue.try_push(std::move(wf));
        }

        bases_batch.clear();
        quals_batch.clear();
        lengths.clear();
        sizes.clear();
        wfs.clear();
        indices_batch.clear();
        remaining_batch_slots = batch_size;
    };

    WindowFeatures item;
    auto last_chunk_reserve_time = std::chrono::system_clock::now();
    while (true) {
        const auto pop_status = m_features_queue.try_pop_until(
                item, last_chunk_reserve_time + std::chrono::milliseconds(10000));

        if (pop_status == utils::AsyncQueueStatus::Terminate) {
            break;
        }

        if (pop_status == utils::AsyncQueueStatus::Timeout) {
            // Ended with a timeout, so run inference if there are samples.
            if (bases_batch.size() > 0) {
                batch_infer();
            }
            last_chunk_reserve_time = std::chrono::system_clock::now();
            continue;
        }

        utils::ScopedProfileRange spr("collect_features", 1);
        int required_batch_slots = ((int)item.bases.sizes()[1] / 5120) + 1;
        if (required_batch_slots > remaining_batch_slots) {
            batch_infer();
        }
        wfs.push_back(std::move(item));
        auto& wf = wfs.back();

        auto b = wf.bases.transpose(0, 1);
        auto q = wf.quals.transpose(0, 1);

        bases_batch.push_back(b);
        quals_batch.push_back(q);
        lengths.push_back(wf.length);
        sizes.push_back(wf.length);
        indices_batch.push_back(wf.indices);
        remaining_batch_slots -= required_batch_slots;
        last_chunk_reserve_time = std::chrono::system_clock::now();
    }

    if (bases_batch.size() > 0) {
        batch_infer();
    }

    m_num_active_infer_threads--;
    if (m_num_active_infer_threads.load() == 0) {
        m_inferred_features_queue.terminate();
    }
}

void CorrectionNode::input_thread_fn() {
    auto thread_id = m_num_active_feature_threads++;

    auto fastx_reader = std::make_unique<hts_io::FastxRandomReader>(m_fastq);

    if (thread_id == 0) {
        total_reads_in_input = fastx_reader->num_entries();
    }

    Message message;
    while (get_input_message(message)) {
        if (std::holds_alternative<CorrectionAlignments>(message)) {
            utils::ScopedProfileRange spr("input_loop", 1);

            auto alignments = std::get<CorrectionAlignments>(std::move(message));
            auto tname = alignments.read_name;
            if (!populate_alignments(alignments, fastx_reader.get())) {
                continue;
            }

            size_t n_windows = (alignments.read_seq.length() + m_window_size - 1) / m_window_size;
            LOG_TRACE("num windows {} for read {}", n_windows, alignments.read_name);
            // Get the windows
            std::vector<std::vector<OverlapWindow>> windows;
            windows.resize(n_windows);
            if (!extract_windows(windows, alignments, m_window_size)) {
                continue;
            }
            // Get the features
            auto wfs = extract_features(windows, alignments, m_window_size);

            std::vector<std::string> corrected_seqs;
            corrected_seqs.resize(wfs.size());

            // Move windows that don't need inferring into an output
            // vector for later use.
            std::vector<WindowFeatures> features_to_infer;
            for (size_t w = 0; w < wfs.size(); w++) {
                if (wfs[w].n_alns > 1 && wfs[w].supported.size() > 0) {
                    features_to_infer.push_back(std::move(wfs[w]));
                } else {
                    corrected_seqs[w] = decode_window(wfs[w]);
                }
            }
            if (features_to_infer.empty()) {
                num_early_reads++;
                concat_features_and_send(corrected_seqs, tname);
            } else {
                std::lock_guard<std::mutex> lock(m_features_mutex);
                if (m_features_by_id.find(tname) == m_features_by_id.end()) {
                    m_features_by_id.insert({tname, std::move(corrected_seqs)});
                    m_pending_features_by_id.insert({tname, (int)features_to_infer.size()});
                } else {
                    spdlog::error("Features for {} already exist! Skipping.", tname);
                    continue;
                }
            }
            // Push the ones that need inference to another thread.
            for (auto& wf : features_to_infer) {
                LOG_TRACE("Pushing window idx {} to features queue", wf.window_idx);
                m_features_queue.try_push(std::move(wf));
            }
            num_reads++;

            // TODO: Remove this and move to ProgressTracker
            if (num_reads.load() % 10000 == 0) {
                spdlog::debug("Corrected {} reads, decoded {} reads early, ", num_reads.load(),
                              num_early_reads.load());
            }
        } else {
            send_message_to_sink(std::move(message));
            continue;
        }
    }

    m_num_active_feature_threads--;
    if (m_num_active_feature_threads.load() == 0) {
        m_features_queue.terminate();
    }
}

CorrectionNode::CorrectionNode(const std::string& fastq,
                               int threads,
                               const std::string& device,
                               int infer_threads,
                               const int batch_size,
                               const std::filesystem::path& model_dir)
        : MessageSink(1000, threads),
          m_fastq(fastq),
          m_model_config(parse_model_config(model_dir / "config.toml")),
          m_features_queue(1000),
          m_inferred_features_queue(500),
          m_bases_manager(batch_size),
          m_quals_manager(batch_size) {
    m_window_size = m_model_config.window_size;

    std::vector<std::string> devices;
    if (device == "cpu") {
        infer_threads = 1;
        devices.push_back(device);
    }
#if DORADO_CUDA_BUILD
    else if (utils::starts_with(device, "cuda")) {
        devices = dorado::utils::parse_cuda_device_string(device);
        if (devices.empty()) {
            throw std::runtime_error("CUDA device requested but no devices found.");
        }
    }
#else
    else {
        throw std::runtime_error("Unsupported device: " + device);
    }
#endif
    for (size_t d = 0; d < devices.size(); d++) {
        const auto& dev = devices[d];
        for (int i = 0; i < infer_threads; i++) {
            int device_batch_size = batch_size;
            if (batch_size == 0) {
                device_batch_size = calculate_batch_size(dev, 0.8f);
                if (device_batch_size == 0) {
                    throw std::runtime_error("Insufficient memory to run inference on " + dev);
                }
            }
            spdlog::debug("Using batch size {} on device {}", device_batch_size, dev);
            m_infer_threads.push_back(
                    std::thread(&CorrectionNode::infer_fn, this, dev, (int)d, device_batch_size));
        }
    }
    for (int i = 0; i < 4; i++) {
        m_decode_threads.push_back(std::thread(&CorrectionNode::decode_fn, this));
    }
    // Create index for fastq file.
    char* idx_name = fai_path(fastq.c_str());
    spdlog::debug("Looking for idx {}", idx_name);
    if (!std::filesystem::exists(idx_name)) {
        if (fai_build(fastq.c_str()) != 0) {
            spdlog::error("Failed to build index for file {}", fastq);
            throw std::runtime_error("");
        }
        spdlog::debug("Created fastq index.");
    }
    hts_free(idx_name);
    start_input_processing(&CorrectionNode::input_thread_fn, this);
}

void CorrectionNode::terminate(const FlushOptions&) {
    stop_input_processing();
    for (auto& infer_thread : m_infer_threads) {
        if (infer_thread.joinable()) {
            infer_thread.join();
        }
    }
    m_infer_threads.clear();
    for (auto& decode_thread : m_decode_threads) {
        if (decode_thread.joinable()) {
            decode_thread.join();
        }
    }
    m_decode_threads.clear();
}

stats::NamedStats CorrectionNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    stats["num_reads_corrected"] = double(num_reads.load());
    stats["total_reads_in_input"] = total_reads_in_input;
    return stats;
}

}  // namespace dorado
