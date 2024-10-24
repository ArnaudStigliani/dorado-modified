#pragma once

#include "poly_tail_calculator.h"

namespace dorado::poly_tail {

class RNAPolyTailCalculator : public PolyTailCalculator {
public:
 RNAPolyTailCalculator(PolyTailConfig config, const std::string& debug_path) : PolyTailCalculator(std::move(config),debug_path) {}
    SignalAnchorInfo determine_signal_anchor_and_strand(const SimplexRead& read) const override;

protected:
    float average_samples_per_base(const std::vector<float>& sizes) const override;
    int signal_length_adjustment(int signal_len) const override;
    float min_avg_val() const override { return 0.0f; }
    std::pair<int, int> signal_range(int signal_anchor,
                                     int signal_len,
                                     float samples_per_base) const override;
};

}  // namespace dorado::poly_tail
