#pragma once

#include "dna_poly_tail_calculator.h"

namespace dorado::poly_tail {

class PlasmidPolyTailCalculator : public DNAPolyTailCalculator {
public:
 PlasmidPolyTailCalculator(PolyTailConfig config, const std::string& debug_path) : DNAPolyTailCalculator(std::move(config), debug_path) {}
    SignalAnchorInfo determine_signal_anchor_and_strand(const SimplexRead& read) const override;
};

}  // namespace dorado::poly_tail
