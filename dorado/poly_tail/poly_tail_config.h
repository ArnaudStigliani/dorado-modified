#pragma once

#include <istream>
#include <string>

namespace dorado::poly_tail {
  
  struct PolyTailConfig {
    /* std::string front_primer = "TTTCTGTTGGTGCTGATATTGCTTT";                          // SSP */
    /* std::string rear_primer = "ACTTGCCTGTCGCTCTATCTTCAGAGGAGAGTCCGCCGCCCGCAAGTTTT";  // VNP */
    std::string front_primer = "";                          // SSP
    /* std::string rear_primer = "GACATACCGTGAATCGCTATCCTATGTTCATCCGCACAAAGACACCGACAACTTTCTT";  // VNP */
    std::string rear_primer = "ATCCGCACAAAGACACCGACAACTTTCTT";  // VNP
    std::string rc_front_primer;
    std::string rc_rear_primer;
    std::string plasmid_front_flank;
    std::string plasmid_rear_flank;
    std::string rc_plasmid_front_flank;
    std::string rc_plasmid_rear_flank;
    int primer_window = 150;
    float flank_threshold = 0.55f;
    bool is_plasmid = false;
    int tail_interrupt_length = 10;
    int min_base_count = 10;
};

// Prepare the PolyA configuration struct. If a configuration
// file is available, parse it to extract parameters. Otherwise
// prepare the default configuration.
PolyTailConfig prepare_config(const std::string& config_file);

// Overloaded function that parses the configuration passed
// in as an input stream.
PolyTailConfig prepare_config(std::istream& is);

}  // namespace dorado::poly_tail
