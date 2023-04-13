#include "bam_utils.h"

#include "Version.h"
#include "htslib/bgzf.h"
#include "htslib/kroundup.h"
#include "htslib/sam.h"
#include "minimap.h"
//todo: mmpriv.h is a private header from mm2 for the mm_event_identity function.
//Ask lh3 t  make some of these funcs publicly available?
#include "mmpriv.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/duplex_utils.h"

#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
// seq_nt16_str is referred to in the hts-3.lib stub on windows, but has not been declared dllimport for
//  client code, so it comes up as an undefined reference when linking the stub.
const char seq_nt16_str[] = "=ACMGRSVTWYHKDBN";

// These special wrapped memory management function are needed
// because on Windows, memory allocated and returned from one
// DLL cannot be freed or managed from another DLL/exe unless
// they share the runtime library. With htslib that was not
// possible, so these wrapped functions had to be exposed.
// TODO: Find a cleaner solution for this.
#include "htslib/ont_defs.h"
#define _HTSLIB_REALLOC htslib_wrapped_realloc
#else
#define _HTSLIB_REALLOC realloc
#endif  // _WIN32

namespace dorado::utils {

static constexpr std::array<uint8_t, 90> generate_nt16_seq_map() {
    std::array<uint8_t, 90> nt16_seq_map = {0};
    nt16_seq_map['A'] = 0b1;
    nt16_seq_map['C'] = 0b10;
    nt16_seq_map['G'] = 0b100;
    nt16_seq_map['T'] = 0b1000;
    nt16_seq_map['N'] = 0b1111;
    return nt16_seq_map;
}

const std::array<uint8_t, 90> nt16_seq_map = generate_nt16_seq_map();

Aligner::Aligner(MessageSink& sink, const std::string& filename, int threads)
        : MessageSink(10000), m_sink(sink), m_threads(threads) {
    mm_set_opt(0, &m_idx_opt, &m_map_opt);
    // Setting options to map-ont default till relevant args are exposed.
    mm_set_opt("map-ont", &m_idx_opt, &m_map_opt);

    // Set batch sizes large enough to not require chunking since that's
    // not supported yet.
    m_idx_opt.batch_size = 4000000000;
    m_idx_opt.mini_batch_size = 16000000000;

    // Force cigar generation.
    m_map_opt.flag |= MM_F_CIGAR;

    mm_check_opt(&m_idx_opt, &m_map_opt);

    m_index_reader = mm_idx_reader_open(filename.c_str(), &m_idx_opt, 0);
    m_index = mm_idx_reader_read(m_index_reader, m_threads);
    mm_mapopt_update(&m_map_opt, m_index);

    //std::cerr << "Using k:" << m_index->k << ", w:" << m_index->w << ", flag:" << m_index->flag << std::endl;

    if (mm_verbose >= 3) {
        mm_idx_stat(m_index);
    }

    for (int i = 0; i < m_threads; i++) {
        m_tbufs.push_back(mm_tbuf_init());
    }

    for (size_t i = 0; i < m_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&Aligner::worker_thread, this, i)));
    }
}

Aligner::~Aligner() {
    terminate();
    for (auto& m : m_workers) {
        m->join();
    }
    for (int i = 0; i < m_threads; i++) {
        mm_tbuf_destroy(m_tbufs[i]);
    }
    mm_idx_reader_close(m_index_reader);
    mm_idx_destroy(m_index);
    // Adding for thread safety in case worker thread throws exception.
    m_sink.terminate();
}

std::vector<std::pair<char*, uint32_t>> Aligner::sq() {
    std::vector<std::pair<char*, uint32_t>> records;
    for (int i = 0; i < m_index->n_seq; ++i) {
        records.push_back(std::make_pair(m_index->seq[i].name, m_index->seq[i].len));
    }
    return records;
}

void Aligner::worker_thread(size_t tid) {
    m_active++;  // Track active threads.

    Message message;
    while (m_work_queue.try_pop(message)) {
        bam1_t* read = std::get<bam1_t*>(message);
        auto records = align(read, m_tbufs[tid]);
        for (auto& record : records) {
            m_sink.push_message(std::move(record));
        }
        // Free the bam alignment read from the input.
        // Not used anymore after this.
        bam_destroy1(read);
    }

    int num_active = --m_active;
    if (num_active == 0) {
        terminate();
        m_sink.terminate();
    }
}

// Function to add auxiliary tags to the alignment record.
// These are added to maintain parity with mm2.
void Aligner::add_tags(bam1_t* record, const mm_reg1_t* aln, const std::vector<char>& seq) {
    if (aln->p) {
        // NM
        int32_t nm = aln->blen - aln->mlen + aln->p->n_ambi;
        bam_aux_append(record, "NM", 'i', sizeof(nm), (uint8_t*)&nm);

        // ms
        int32_t ms = aln->p->dp_max;
        bam_aux_append(record, "ms", 'i', sizeof(nm), (uint8_t*)&ms);

        // AS
        int32_t as = aln->p->dp_score;
        bam_aux_append(record, "AS", 'i', sizeof(nm), (uint8_t*)&as);

        // nn
        int32_t nn = aln->p->n_ambi;
        bam_aux_append(record, "nn", 'i', sizeof(nm), (uint8_t*)&nn);

        if (aln->p->trans_strand == 1 || aln->p->trans_strand == 2) {
            bam_aux_append(record, "ts", 'A', 2, (uint8_t*)&("?+-?"[aln->p->trans_strand]));
        }
    }

    // de / dv
    if (aln->p) {
        float div;
        div = 1.0 - mm_event_identity(aln);
        bam_aux_append(record, "de", 'f', sizeof(div), (uint8_t*)&div);
    } else if (aln->div >= 0.0f && aln->div <= 1.0f) {
        bam_aux_append(record, "dv", 'f', sizeof(aln->div), (uint8_t*)&aln->div);
    }

    // tp
    char type;
    if (aln->id == aln->parent) {
        type = aln->inv ? 'I' : 'P';
    } else {
        type = aln->inv ? 'i' : 'S';
    }
    bam_aux_append(record, "tp", 'A', sizeof(type), (uint8_t*)&type);

    // cm
    bam_aux_append(record, "cm", 'i', sizeof(aln->cnt), (uint8_t*)&aln->cnt);

    // s1
    bam_aux_append(record, "s1", 'i', sizeof(aln->score), (uint8_t*)&aln->score);

    // s2
    if (aln->parent == aln->id) {
        bam_aux_append(record, "s2", 'i', sizeof(aln->subsc), (uint8_t*)&aln->subsc);
    }

    // MD
    char* md = NULL;
    int max_len = 0;
    int md_len = mm_gen_MD(NULL, &md, &max_len, m_index, aln, seq.data());
    if (md_len > 0) {
        bam_aux_append(record, "MD", 'Z', md_len + 1, (uint8_t*)md);
    }

    // zd
    if (aln->split) {
        uint32_t split = uint32_t(aln->split);
        bam_aux_append(record, "zd", 'i', sizeof(split), (uint8_t*)&split);
    }
}

std::vector<bam1_t*> Aligner::align(bam1_t* irecord, mm_tbuf_t* buf) {
    // some where for the hits
    std::vector<bam1_t*> results;

    // get the sequence to map from the record
    auto seqlen = irecord->core.l_qseq;

    auto bseq = bam_get_seq(irecord);
    std::vector<char> seq(seqlen);
    for (int i = 0; i < seqlen; i++) {
        seq[i] = seq_nt16_str[bam_seqi(bseq, i)];
    }
    std::vector<char> seq_rev(seq);
    reverse_complement(seq_rev);

    std::vector<uint8_t> qual(bam_get_qual(irecord), bam_get_qual(irecord) + seqlen);
    std::vector<uint8_t> qual_rev(qual.rbegin(), qual.rend());

    // do the mapping
    int hits = 0;
    mm_reg1_t* reg = mm_map(m_index, seq.size(), seq.data(), &hits, buf, &m_map_opt, 0);

    // just return the input record
    if (hits == 0) {
        results.push_back(bam_dup1(irecord));
    }

    for (int j = 0; j < hits; j++) {
        // new output record
        auto record = bam_dup1(irecord);

        // mapping region
        auto aln = &reg[j];

        uint16_t flag = 0x0;

        if (aln->rev) {
            flag |= 0x10;
        }
        if (aln->parent != aln->id) {
            flag |= 0x100;
        } else if (!aln->sam_pri) {
            flag |= 0x800;
        }

        record->core.flag = flag;
        record->core.tid = aln->rid;
        record->core.pos = aln->rs;
        record->core.qual = aln->mapq;
        record->core.n_cigar = aln->p ? aln->p->n_cigar : 0;

        // Note: max_bam_cigar_op doesn't need to handled specially when
        // using htslib since the sam_write1 method already takes care
        // of moving the CIGAR string to the tags if the length
        // exceeds 65535.
        if (record->core.n_cigar != 0) {
            uint32_t clip_len[2] = {0};
            clip_len[0] = aln->rev ? record->core.l_qseq - aln->qe : aln->qs;
            clip_len[1] = aln->rev ? aln->qs : record->core.l_qseq - aln->qe;

            if (clip_len[0]) {
                record->core.n_cigar++;
            }
            if (clip_len[1]) {
                record->core.n_cigar++;
            }
            int offset = clip_len[0] ? 1 : 0;
            int cigar_size = record->core.n_cigar * sizeof(uint32_t);
            uint32_t new_m_data = record->l_data + cigar_size;
            kroundup32(new_m_data);
            // todo: this part could possibly by re-written without using realloc.
            uint8_t* data = (uint8_t*)_HTSLIB_REALLOC(record->data, new_m_data);

            // shift existing data to make room for the new cigar field
            memmove(data + record->core.l_qname + cigar_size, data + record->core.l_qname,
                    record->l_data - record->core.l_qname);

            record->data = data;

            // write the left softclip
            if (clip_len[0]) {
                auto clip = bam_cigar_gen(clip_len[0], BAM_CSOFT_CLIP);
                memcpy(bam_get_cigar(record), &clip, sizeof(uint32_t));
            }

            // write the cigar
            memcpy(bam_get_cigar(record) + offset, aln->p->cigar,
                   aln->p->n_cigar * sizeof(uint32_t));

            // write the right softclip
            if (clip_len[1]) {
                auto clip = bam_cigar_gen(clip_len[1], BAM_CSOFT_CLIP);
                memcpy(bam_get_cigar(record) + offset + aln->p->n_cigar, &clip, sizeof(uint32_t));
            }

            // update the data length
            record->l_data += cigar_size;
            record->m_data = new_m_data;

            if (aln->rev) {
                for (int i = 0; i < seqlen; i++) {
                    bam_set_seqi(bam_get_seq(record), i, nt16_seq_map[seq_rev[i]]);
                }
                memcpy(bam_get_qual(record), (void*)qual_rev.data(), seqlen);
            }
        }

        add_tags(record, aln, seq);

        free(aln->p);
        results.push_back(record);
    }

    free(reg);
    return results;
}

BamReader::BamReader(const std::string& filename) {
    m_file = hts_open(filename.c_str(), "r");
    if (!m_file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    format = hts_format_description(hts_get_format(m_file));
    header = sam_hdr_read(m_file);
    if (!header) {
        throw std::runtime_error("Could not read header from file: " + filename);
    }
    is_aligned = header->n_targets > 0;
    record = bam_init1();
}

BamReader::~BamReader() {
    hts_free(format);
    sam_hdr_destroy(header);
    bam_destroy1(record);
    hts_close(m_file);
}

bool BamReader::read() { return sam_read1(m_file, header, record) >= 0; }

void BamReader::read(MessageSink& read_sink, int max_reads) {
    int num_reads = 0;
    while (this->read()) {
        read_sink.push_message(bam_dup1(record));
        if (++num_reads >= max_reads) {
            break;
        }
    }
    read_sink.terminate();
}

BamWriter::BamWriter(const std::string& filename, size_t threads) : MessageSink(1000) {
    m_file = hts_open(filename.c_str(), "w");
    if (!m_file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    auto res = bgzf_mt(m_file->fp.bgzf, threads, 128);
    if (res < 0) {
        throw std::runtime_error("Could not enable multi threading for BAM generation.");
    }
    m_worker = std::make_unique<std::thread>(std::thread(&BamWriter::worker_thread, this));
}

BamWriter::~BamWriter() {
    // Adding for thread safety in case worker thread throws exception.
    terminate();
    if (m_worker->joinable()) {
        join();
    }
    sam_hdr_destroy(m_header);
    hts_close(m_file);
}

void BamWriter::join() { m_worker->join(); }

void BamWriter::worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        bam1_t* aln = std::get<bam1_t*>(message);
        write(aln);
        // Free the bam alignment that's already written
        // out to disk.
        bam_destroy1(aln);
    }
}

int BamWriter::write(bam1_t* record) {
    // track stats
    total++;
    if (record->core.flag & BAM_FUNMAP) {
        unmapped++;
    }
    if (record->core.flag & BAM_FSECONDARY) {
        secondary++;
    }
    if (record->core.flag & BAM_FSUPPLEMENTARY) {
        supplementary++;
    }
    primary = total - secondary - supplementary - unmapped;

    auto res = sam_write1(m_file, m_header, record);
    if (res < 0) {
        throw std::runtime_error("Failed to write SAM record, error code " + std::to_string(res));
    }
    return res;
}

int BamWriter::write_header(const sam_hdr_t* header, const sq_t seqs) {
    m_header = sam_hdr_dup(header);
    write_hdr_pg();
    for (auto pair : seqs) {
        write_hdr_sq(std::get<0>(pair), std::get<1>(pair));
    }
    auto res = sam_hdr_write(m_file, m_header);
    return res;
}

int BamWriter::write_hdr_pg() {
    // todo: add CL Writer node
    return sam_hdr_add_line(m_header, "PG", "ID", "aligner", "PN", "dorado", "VN", DORADO_VERSION,
                            "DS", MM_VERSION, NULL);
}

int BamWriter::write_hdr_sq(char* name, uint32_t length) {
    return sam_hdr_add_line(m_header, "SQ", "SN", name, "LN", std::to_string(length).c_str(), NULL);
}

read_map read_bam(const std::string& filename, const std::set<std::string>& read_ids) {
    BamReader reader(filename);

    read_map reads;

    while (reader.read()) {
        std::string read_id = bam_get_qname(reader.record);

        if (read_ids.find(read_id) == read_ids.end()) {
            continue;
        }

        uint8_t* qstring = bam_get_qual(reader.record);
        uint8_t* sequence = bam_get_seq(reader.record);

        uint32_t seqlen = reader.record->core.l_qseq;
        std::vector<uint8_t> qualities(seqlen);
        std::vector<char> nucleotides(seqlen);

        // Todo - there is a better way to do this.
        for (int i = 0; i < seqlen; i++) {
            qualities[i] = qstring[i] + 33;
            nucleotides[i] = seq_nt16_str[bam_seqi(sequence, i)];
        }

        auto tmp_read = std::make_shared<Read>();
        tmp_read->read_id = read_id;
        tmp_read->seq = std::string(nucleotides.begin(), nucleotides.end());
        tmp_read->qstring = std::string(qualities.begin(), qualities.end());
        reads[read_id] = tmp_read;
    }

    return reads;
}

}  // namespace dorado::utils
