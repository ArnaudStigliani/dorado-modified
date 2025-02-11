#include "dna_poly_tail_calculator.h"

#include "read_pipeline/messages.h"
#include "utils/math_utils.h"
#include "utils/sequence_utils.h"

#include <edlib.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>

namespace dorado::poly_tail {

SignalAnchorInfo DNAPolyTailCalculator::determine_signal_anchor_and_strand(const SimplexRead& read) const {
    // const std::string& front_primer = m_config.front_primer;
    // const std::string& front_primer_rc = m_config.rc_front_primer;
    const std::string& rear_primer = m_config.rear_primer;
    const std::string& rear_primer_rc = m_config.rc_rear_primer;
    const float threshold = m_config.flank_threshold;
    const int primer_window = m_config.primer_window;
    int trailing_Ts = static_cast<int>(dorado::utils::count_trailing_chars(rear_primer, 'T'));

    // const int kMinSeparation = 10;
    std::string_view seq_view = std::string_view(read.read_common.seq);
    std::string_view read_top = seq_view.substr(0, primer_window);
    // spdlog::debug("read_top {}", read_top);
    auto bottom_start = std::max(0, (int)seq_view.length() - primer_window);
    std::string_view read_bottom = seq_view.substr(bottom_start, primer_window);
    // spdlog::debug("read_bottom {}", read_bottom);
    
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_LOC;
    align_config.mode = EDLIB_MODE_HW;

    // Check for forward strand.
    // EdlibAlignResult top_v1 = edlibAlign(front_primer.data(), int(front_primer.length()),
    //                                      read_top.data(), int(read_top.length()), align_config);
    EdlibAlignResult bottom_v1 =
            edlibAlign(rear_primer_rc.data(), int(rear_primer_rc.length()), read_bottom.data(),
                       int(read_bottom.length()), align_config);

    int dist_v1 =  bottom_v1.editDistance;
    // int dist_v1 = top_v1.editDistance + bottom_v1.editDistance;

    // Check for reverse strand.
    EdlibAlignResult top_v2 = edlibAlign(rear_primer.data(), int(rear_primer.length()),
                                         read_top.data(), int(read_top.length()), align_config);
    // EdlibAlignResult bottom_v2 =
    //         edlibAlign(front_primer_rc.data(), int(front_primer_rc.length()), read_bottom.data(),
    //                    int(read_bottom.length()), align_config);

    // int dist_v2 = top_v2.editDistance + bottom_v2.editDistance;
    int dist_v2 = top_v2.editDistance;
    spdlog::trace("v1 dist {}, v2 dist {}", dist_v1, dist_v2);
    spdlog::debug("bottom_v1 {}, top_v2 {}",  bottom_v1.editDistance, top_v2.editDistance);

    const bool fwd = dist_v1 < dist_v2;
    // const float flank_score = 1.f - static_cast<float>(std::min(dist_v1, dist_v2)) /
    //                                         (front_primer.length() + rear_primer.length());
    const float flank_score = 1.f - static_cast<float>(std::min(dist_v1, dist_v2)) /
                                            (rear_primer.length());

    // const bool proceed = flank_score >= threshold && std::abs(dist_v1 - dist_v2) > kMinSeparation;
    const bool proceed = flank_score >= threshold;
    // spdlog::debug("flank score {}, threshold {}",  flank_score,  threshold );

    SignalAnchorInfo result = {false, -1, trailing_Ts, false};

    if (proceed) {
        int base_anchor = 0;
        if (fwd) {
            base_anchor = bottom_start + bottom_v1.endLocations[0];
        } else {
            base_anchor = top_v2.startLocations[0];
        }
	// spdlog::debug("base_anchor  {}", base_anchor);

        const auto stride = read.read_common.model_stride;
        const auto seq_to_sig_map = dorado::utils::moves_to_map(
                read.read_common.moves, stride, read.read_common.get_raw_data_samples(),
                read.read_common.seq.size() + 1);
        int signal_anchor = int(seq_to_sig_map[base_anchor]);
	// spdlog::debug("signal_anchor  {}", signal_anchor);
	
        result = {fwd, signal_anchor, trailing_Ts, false};
    } else {
        spdlog::trace("{} primer edit distance too high {}", read.read_common.read_id,
                      std::min(dist_v1, dist_v2));
    }

    // edlibFreeAlignResult(top_v1);
    edlibFreeAlignResult(bottom_v1);
    edlibFreeAlignResult(top_v2);
    // edlibFreeAlignResult(bottom_v2);

    return result;
}

float DNAPolyTailCalculator::average_samples_per_base(const std::vector<float>& sizes) const {
    auto quantiles = dorado::utils::quantiles(sizes, {0.5});
    return static_cast<float>(quantiles[0]);
}

std::pair<int, int> DNAPolyTailCalculator::signal_range(int signal_anchor,
                                                        int signal_len,
                                                        float samples_per_base) const {
    const int kSpread = int(std::round(samples_per_base * max_tail_length()));
    return {std::max(0, signal_anchor - kSpread), std::min(signal_len, signal_anchor + kSpread)};
}

}  // namespace dorado::poly_tail
