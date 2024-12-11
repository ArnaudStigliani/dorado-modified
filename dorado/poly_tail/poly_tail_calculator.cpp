#include "poly_tail_calculator.h"

#include "dna_poly_tail_calculator.h"
#include "plasmid_poly_tail_calculator.h"
#include "poly_tail_config.h"
#include "read_pipeline/messages.h"
#include "rna_poly_tail_calculator.h"
#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

namespace dorado::poly_tail {
  
namespace {
    const int kMaxTailLength = PolyTailCalculator::max_tail_length();
}
  
  float PolyTailCalculator::estimate_samples_per_base(const dorado::SimplexRead& read) const {
    const size_t num_bases = read.read_common.seq.length();
    const auto num_samples = read.read_common.get_raw_data_samples();
    const auto stride = read.read_common.model_stride;
    const auto seq_to_sig_map =
      dorado::utils::moves_to_map(read.read_common.moves, stride, num_samples, num_bases + 1);
    // Store the samples per base in float to use the quantile calcuation function.
    std::vector<float> sizes(seq_to_sig_map.size() - 1, 0.f);
    for (int i = 1; i < int(seq_to_sig_map.size()); i++) {
      sizes[i - 1] = static_cast<float>(seq_to_sig_map[i] - seq_to_sig_map[i - 1]);
    }
    
    return average_samples_per_base(sizes);
}
  
  
  /* Here is the begining of my personnal code
     ####################################### */

std::vector<c10::Half> rectify(const std::vector<c10::Half>& data) {
    std::vector<c10::Half> rectified_data = data;
    std::transform(rectified_data.begin(), rectified_data.end(), rectified_data.begin(), 
                   [](c10::Half val) { return val < 0 ? -val : val; });
    return rectified_data;
}

std::vector<c10::Half> truncate_spikes(const std::vector<c10::Half>& data, c10::Half threshold) {
    std::vector<c10::Half> truncated_data = data;
    std::for_each(truncated_data.begin(), truncated_data.end(), 
                  [threshold](c10::Half& val) { if (val > threshold) val = threshold; });
    return truncated_data;
}


double standard_deviation(const c10::Half* data, int start, int end) {

    /* cast  double to keep the precision */
    // Compute mean
    double sum=0.0;
    for(int i=start ; i< end ; ++i)
      {
	sum += static_cast<double>(data[i]);
      }
    double mean = sum /(end -start) ;

    
  // Calculate the sum of squared differences
    double sq_sum=0.0;
    for(int i=start ; i< end ; ++i)
      {
	double value = static_cast<double>(data[i]);
	sq_sum += value*value;
      }
    double variance = sq_sum / (end -start) - mean * mean;
    double std_dev = std::sqrt(variance);
  
  // Compute the standard deviation and cast to c10::Half
  return std_dev;
}

  std::pair<int,int> checkTailConfidence(std::pair<int,int>& interval, 
				     const c10::Half* raw_data,
				     int& data_size, bool& fwd) {
    int tail_start = interval.first;
    int tail_end = interval.second;
  if (tail_end != 0 && tail_start != 0) { // using -1 for NA equivalent in C++
    int mid_notail=0;
    int length_no_tail = 0;
    double tail_prox_transcript_variance = 0.0;
    if (fwd){
      length_no_tail =  tail_start;
      mid_notail = length_no_tail / 2;
      tail_prox_transcript_variance = standard_deviation(raw_data, 0 , mid_notail);
    }else {
      length_no_tail = data_size - tail_end;
      mid_notail = tail_end + length_no_tail / 2;
      tail_prox_transcript_variance = standard_deviation(raw_data, mid_notail , data_size -1 );
    }
    double tail_variance = standard_deviation(raw_data, tail_start, tail_end);
    int tail_length = tail_end - tail_start;
    // more stringent requirement of low variance for shorter tails
    if (tail_length > 20) {
      if (tail_variance > tail_prox_transcript_variance / 2) 	{ // fake tail
	tail_start = 0;
	tail_end = 0;
	tail_length = 0;
	}
    } else {
      if (tail_variance > tail_prox_transcript_variance / 2) { // fake tail
	tail_start = 0;
	tail_end = 0;
	tail_length = 0;
      }
    }
    return {tail_start, tail_end};
  }
  return {0, 0};
}

std::vector<c10::Half> z_normalize(const c10::Half* data, int data_size) {
    // Check if data is empty to avoid division by zero
    if (data_size == 0 ) {
        return {};
    }
    /* cast  double to keep the precision */
    // Compute mean
    double sum=0.0;
    for(int i=0 ; i< data_size ; ++i)
      {
	sum += static_cast<double>(data[i]);
      }
    double mean = sum / data_size;

    // c10::Half sq_sum=0.0;
    double sq_sum=0.0;
    for(int i=0 ; i< data_size ; ++i)
      {
	double value = static_cast<double>(data[i]);
	sq_sum += value*value;
      }
    double variance = sq_sum / data_size - mean * mean;
    double std_dev = std::sqrt(variance);
    
    // Compute standard deviation

    // Handle edge case where standard deviation is zero
    if (std_dev == 0) {
      return std::vector<c10::Half>(data_size, 0.0);
    }

    // Z-normalize data
    std::vector<double> norm_data;
    norm_data.reserve(data_size);
    for(int i=0 ; i< data_size ; ++i)
      {
	double value = static_cast<double>(data[i]);
        norm_data.push_back((value - mean) / std_dev);	
      }
  std::vector<c10::Half> norm_data_converted(norm_data.size());
  std::transform(norm_data.begin(), norm_data.end(), norm_data_converted.begin(), [](c10::Half value) {return static_cast<c10::Half>(value);});
  return norm_data_converted;
  }

std::vector<c10::Half> left_to_right_sliding_window_dna(const std::vector<c10::Half>& data, int window_size, int step = 1) {
  int data_size = data.size();
  std::vector<c10::Half> result(data_size, 0.0);

  for (int i = 0; i <= data_size - window_size; i += step) {
    c10::Half sum = std::accumulate(data.begin() + i, data.begin() + i + window_size, 0.0);
    c10::Half mean = sum / window_size;
        
    // Set the moving average value for the window
    for (int j = 0; j < window_size; ++j) {
      result[i + j] = mean;
    }
  }

  return result;
}

  std::vector<c10::Half> right_to_left_sliding_window_dna(const std::vector<c10::Half>& data, int window_size, int step = 1) {
    int data_size = data.size();
  std::vector<c10::Half> result(data_size, 0.0);
  
  for (int i = data_size - 1; i >= window_size - 1; i -= step) {
    c10::Half sum = std::accumulate(data.begin() + i - window_size + 1, data.begin() + i + 1, 0.0);
    c10::Half mean = sum / window_size;
        
    // Set the moving average value for the window
    for (int j = 0; j < window_size; ++j) {
      result[i - j] = mean;
    }
  }

  return result;
}

  
  // std::vector<int> find_dna_tail_per_read(std::vector<c10::Half> raw_data,bool fwd, int anchor) {

std::pair<int, int> PolyTailCalculator::determine_signal_bounds(int signal_anchor,
                                                                bool fwd,
                                                                const dorado::SimplexRead& read,
                                                                float num_samples_per_base) const {
  //
  // Parameters
  const int samples_per_nt = int(num_samples_per_base);
  const c10::Half SPIKE_THRESHOLD = 2.0;
  const int MOVING_WINDOW_SIZE = 30;
  const int MAX_GAP_BETWEEN_TAILS = 20 * samples_per_nt; 
  const c10::Half SLOPE_THRESHOLD = 0.20;
  const int signal_len = int(read.read_common.get_raw_data_samples());
  const c10::Half*  raw_data= static_cast<c10::Half*>(read.read_common.raw_data.data_ptr());
  std::vector<c10::Half> norm_data = z_normalize(raw_data, signal_len );   // Z-normalize the data

  if (fwd) {
    std::reverse(norm_data.begin(), norm_data.end());   // Reverse polyA reads if necessary
  }
  
  int read_length = norm_data.size();
  int tail_start = (fwd) ? (read_length - signal_anchor) : signal_anchor;  
  std::vector<c10::Half> rectified_data = rectify(norm_data); //abs value of data
  std::vector<c10::Half> truncated_data = truncate_spikes(rectified_data, SPIKE_THRESHOLD); //cut everything > threshold
  std::vector<c10::Half> smoothed_data_lr = left_to_right_sliding_window_dna(truncated_data, MOVING_WINDOW_SIZE, 1);
  std::vector<c10::Half> smoothed_data_rl = right_to_left_sliding_window_dna(truncated_data, MOVING_WINDOW_SIZE, 1);
  std::vector<c10::Half> smoothed_data(smoothed_data_lr.size(), 0); //initialize vector with 0
  
  for (size_t i = 0; i < smoothed_data.size(); ++i) {
    smoothed_data[i] = std::min(smoothed_data_lr[i], smoothed_data_rl[i]);
  }

  int window_size = 10;
  int step_size = 10;
  int start = tail_start;
  int end =  read_length;
  std::vector<c10::Half> data(truncated_data.begin() + start, truncated_data.begin() + end);
  int total = data.size();
  std::vector<c10::Half> mean_data(total / step_size, 0.0);
  std::vector<c10::Half> slope(mean_data.size() - 1, 0.0);

  for (size_t i = 0; i < mean_data.size() - 1; ++i) {
    mean_data[i] = std::accumulate(data.begin() + i * step_size, data.begin() + (i + 1) * step_size, 0.0) / window_size;
    mean_data[i + 1] = std::accumulate(data.begin() + (i + 1) * step_size, data.begin() + (i + 2) * step_size, 0.0) / window_size;
    slope[i] = mean_data[i + 1] - mean_data[i];
  }

  
  std::optional<int> precise_tail_start = std::nullopt;
  int k = 0;
  std::vector<int> pass;
  int poly_start_seach_window = 60;
  
  while (k < poly_start_seach_window && k + 7 < int(slope.size())) {

    bool slope_in_range = std::abs(slope[k]) < SLOPE_THRESHOLD &&
					       std::abs(slope[k + 1]) < SLOPE_THRESHOLD &&
									std::abs(slope[k + 2]) < SLOPE_THRESHOLD &&
												 std::abs(slope[k + 3]) < SLOPE_THRESHOLD && 
															  std::abs(slope[k + 4]) < SLOPE_THRESHOLD &&
																		   std::abs(slope[k + 5]) < SLOPE_THRESHOLD &&
																					    std::abs(slope[k + 6]) < SLOPE_THRESHOLD &&
																								     std::abs(slope[k + 7]) < SLOPE_THRESHOLD ;
    spdlog::debug("k = {} ; slope_in_range = {} ; slope = {}" , k, slope_in_range, slope[k]);
    bool mean_in_range=true;
    // bool mean_in_range =  mean_data[k] < SLOPE_THRESHOLD + 0.2 &&
    // 					 mean_data[k + 1] < SLOPE_THRESHOLD + 0.2; // &&
      // mean_data[k + 2] < SLOPE_THRESHOLD + 0.2 &&
      // mean_data[k + 3] < SLOPE_THRESHOLD + 0.20;
    bool mean_way_out = false;
    // bool mean_way_out =  mean_data[k] > SLOPE_THRESHOLD + 0.5 &&
    // 															  mean_data[k + 1] > SLOPE_THRESHOLD + 0.5 &&
    // 												 mean_data[k + 2] > SLOPE_THRESHOLD + 0.5 &&
    // 									mean_data[k + 3] > SLOPE_THRESHOLD + 0.5 &&
    // 					       mean_data[k + 4] > SLOPE_THRESHOLD + 0.5 ; //this is to avoid fake starts
    if (slope_in_range && mean_in_range) { //if the slope and mean are in the range
      pass.push_back(0);
    }
    else if (mean_way_out){ // if the mean is way out of range
      pass.push_back(2);
    }else { // if the slope is out of range
      pass.push_back(1);
    }
    ++k;
  }
  // for(size_t i=0;i<pass.size(); i++)
  //   {
  //     spdlog::debug("i = {} ; pass = {} ", i, pass[i]);
  //   }
  
      
  /* Here, we find the best start. If there is a mean value way out of range
  we will start the polyA/T  search  start from its position */  
  auto last_two_pos = std::find(pass.rbegin(), pass.rend(), 2);
  std::vector<int> pass2(last_two_pos.base(), pass.end());
  auto zero_pos = std::find(pass2.begin(), pass2.end(), 0);
  pass2.erase(pass2.begin(), zero_pos);
  // get index of the precise start (to extend the polyA/T end search window)
  k = poly_start_seach_window - pass2.size();
  // If there is no 0 in the vector, we will start the search from the beginning,
  // whether there is a mean value way out of range or not//
  if(pass2.size() ==0)
    {
      zero_pos =  std::find(pass.begin(), pass.end(), 0);
      std::vector<int> pass3(pass.begin(), zero_pos);
      k=pass3.size();
    }
  // k=54;
  if (k != poly_start_seach_window) {
    precise_tail_start = k * window_size + tail_start; // get precise start
  }

  int tail_end=0;
  int width = 0;
  int last_good_end = 0;
  if (precise_tail_start) { //else, we will return tail_start=0;
    tail_start = *precise_tail_start;
    // Determine tail end
    int i = k; //start the search from the precise boundaries (determined above)
    c10::Half sm_data_threshold = fwd ? 5 : 5;
    
    while (i < int(slope.size())) {
      if (std::abs(slope[i]) < SLOPE_THRESHOLD && smoothed_data[tail_start + i * window_size] < sm_data_threshold) { 
	tail_end = i;
	++i;
      } else { // if there is gap in the tail
	last_good_end = i; 
	int small_glitch_count = 0;
	bool quit_searching = false;
	while (i < int(slope.size()))
	  {
	    if (i - last_good_end  > (MAX_GAP_BETWEEN_TAILS / window_size)){
	      quit_searching = true;
	      break;
	    }
 	    // if (std::abs(mean_data[i]) > SLOPE_THRESHOLD + 0.1) {
	    if (std::abs(slope[i]) > SLOPE_THRESHOLD) {
	      small_glitch_count = 0;
	    } else if (++small_glitch_count > 7) {
	      break;
	    }
	    ++i;
	  }
	if (quit_searching) {
	  width =  last_good_end - k;
	  break;
	}
      }
    }
  }else {
    tail_start=0;
    tail_end=0;
  }
  if(tail_start != 0 &&  tail_end !=0){
    tail_end =  tail_start + width  * window_size;
    tail_start = tail_start;
    int final_tail_start = fwd ? (read_length - tail_end) : tail_start;
    int final_tail_end = fwd ? (read_length - tail_start) : tail_end;
    std::pair<int,int> best_interval = {final_tail_start, final_tail_end};
    best_interval = checkTailConfidence(best_interval, raw_data, read_length, fwd);
    return best_interval; 
  }
  return {0,0};
}

  
  /* Here is the end of my personnal code
     ####################################### */
  

  

int PolyTailCalculator::calculate_num_bases(const SimplexRead& read,
                                            const SignalAnchorInfo& signal_info
					    ) const {
    spdlog::debug("{} Strand {}; poly A/T signal anchor {}", read.read_common.read_id,
                  signal_info.is_fwd_strand ? '+' : '-', signal_info.signal_anchor);
    
    auto num_samples_per_base = estimate_samples_per_base(read);

    // Walk through signal. Require a minimum of length 10 poly-A since below that
    // the current algorithm returns a lot of false intervals.
    auto [signal_start, signal_end] = determine_signal_bounds(
            signal_info.signal_anchor, signal_info.is_fwd_strand, read, num_samples_per_base);

    auto signal_len = signal_end - signal_start;
    signal_len -= signal_length_adjustment(signal_len);

    int num_bases = int(std::round(static_cast<float>(signal_len) / num_samples_per_base)) -
                    signal_info.trailing_adapter_bases;

    spdlog::debug(
            "{} PolyA bases {}, signal anchor {} Signal range is {} {} Signal length "
            "{}, samples/base {} trim {} read len {}",
            read.read_common.read_id, num_bases, signal_info.signal_anchor, signal_start,
            signal_end, signal_len, num_samples_per_base, read.read_common.num_trimmed_samples,
            read.read_common.seq.length());
    spdlog::debug("{}", read.read_common.get_raw_data_samples());
    if (! m_debug_path.empty())
      {
	std::string strand = signal_info.is_fwd_strand ? "plus" : "minus";
	fs::create_directories(m_debug_path);
	const c10::Half* sig = static_cast<c10::Half*>(read.read_common.raw_data.data_ptr());
	std::ofstream myfile;
	std::string file_name = read.read_common.read_id + "_" + std::to_string(num_bases) + "_" +
	  std::to_string(signal_info.signal_anchor)+ "_" + std::to_string(signal_start) + "_" +std::to_string(signal_end)+ "_" +
	  std::to_string(static_cast<int>(num_samples_per_base))+ "_"+   std::to_string(read.read_common.num_trimmed_samples) + "_" +
	  std::to_string(read.read_common.seq.length()) + "_" +  strand  +".txt";
	const auto seq_to_sig_map = dorado::utils::moves_to_map(read.read_common.moves, read.read_common.model_stride,
								read.read_common.get_raw_data_samples(), read.read_common.seq.length() + 1);
	file_name = m_debug_path + "/" + file_name ;
	myfile.open (file_name);
	int i;
	int j = 0;
	std::string base_called;
	  for(i=0 ; i < int(read.read_common.get_raw_data_samples()) ; i++)
	  {
	    base_called = ".";
	    if (i == int(seq_to_sig_map[j]))
	      {
		base_called = read.read_common.seq[j];
		j+=1;
	      }
	    myfile  << i << "\t" << sig[i] << "\t" << base_called << std::endl;
	  }
	myfile.close();
      }
    return num_bases;
}
  
  std::shared_ptr<const PolyTailCalculator> PolyTailCalculatorFactory::create(
									      bool is_rna,
									      const std::string& config_file,
									      const std::string& debug_path) {
    auto config = prepare_config(config_file);
    if (is_rna) {
      return std::make_unique<RNAPolyTailCalculator>(std::move(config),"");
    }
    if (config.is_plasmid) {
      return std::make_unique<PlasmidPolyTailCalculator>(std::move(config),"");
    }
    return std::make_unique<DNAPolyTailCalculator>(std::move(config), debug_path);
  }
  
}  // namespace dorado::poly_tail
