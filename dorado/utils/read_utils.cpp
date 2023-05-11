#include "read_utils.h"

namespace dorado::utils {
std::shared_ptr<Read> shallow_copy_read(const Read& read) {
    auto copy = std::make_shared<Read>();
    copy->raw_data = read.raw_data;
    copy->digitisation = read.digitisation;
    copy->range = read.range;
    copy->offset = read.offset;
    copy->sample_rate = read.sample_rate;

    copy->shift = read.shift;
    copy->scale = read.scale;

    copy->scaling = read.scaling;

    copy->num_chunks = read.num_chunks;
    copy->num_modbase_chunks = read.num_modbase_chunks;

    copy->model_stride = read.model_stride;

    copy->read_id = read.read_id;
    copy->seq = read.seq;
    copy->qstring = read.qstring;
    copy->moves = read.moves;
    copy->run_id = read.run_id;
    copy->model_name = read.model_name;

    copy->base_mod_probs = read.base_mod_probs;
    copy->base_mod_info = read.base_mod_info;

    copy->num_trimmed_samples = read.num_trimmed_samples;

    copy->attributes = read.attributes;

    copy->start_sample = read.start_sample;
    copy->end_sample = read.end_sample;
    copy->run_acquisition_start_time_ms = read.run_acquisition_start_time_ms;
    return copy;
}

std::vector<uint64_t> move_cum_sums(const std::vector<uint8_t>& moves) {
    std::vector<uint64_t> ans(moves.size(), 0);
    if (!moves.empty()) {
        ans[0] = moves[0];
    }
    for (size_t i = 1, n = moves.size(); i < n; i++) {
        ans[i] = ans[i - 1] + moves[i];
    }
    return ans;
}

}  // namespace dorado::utils
