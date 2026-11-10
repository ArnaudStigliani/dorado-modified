#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <numeric>
#include <optional>
#include <iterator>
#include <fstream>
#include <c10/util/Half.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <thread>
#include <future>

namespace fs = std::filesystem;


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



std::vector<c10::Half> z_normalize(const std::vector<c10::Half>& data) {
    // Check if data is empty to avoid division by zero
    if (data.empty()) {
        return {};
    }

    // Compute mean
    float sum = float(std::accumulate(data.begin(), data.end(), 0.0));
    float mean = sum / int(data.size());
    // Compute standard deviation
    float sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    float variance = sq_sum / int(data.size()) - mean * mean;
    float std_dev = std::sqrt(variance);

    // Handle edge case where standard deviation is zero
    if (std_dev == 0) {
        return std::vector<c10::Half>(data.size(), 0.0);
    }

    // Z-normalize data
    std::vector<c10::Half> norm_data;
    norm_data.reserve(data.size());
    for (const auto& value : data) {
        norm_data.push_back((value - mean) / std_dev);
    }

    return norm_data;
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


double standard_deviation(const std::vector<c10::Half>& data, int start, int end) {
  double mean = std::accumulate(data.begin() + start, data.begin() + end, 0.0,
				[](double sum, c10::Half val) { return sum + static_cast<double>(val); }) 
    / (end - start);
  // Calculate the sum of squared differences
  double sum_sq_diff = 0.0;
  for (int i = start; i < end; ++i) {
    sum_sq_diff += std::pow(static_cast<double>(data[i]) - mean, 2);
  }
  
  // Compute the standard deviation and cast to c10::Half
  double stddev = std::sqrt(sum_sq_diff / (end - start));
  return stddev;
}

std::vector<int> checkTailConfidence(int& tail_start, int& tail_end, 
				     const std::vector<c10::Half>& raw_data,
				     int& data_size, bool& fwd) {
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
    bool is_a_low_confidence_tail = false;
    double tail_variance = standard_deviation(raw_data, tail_start, tail_end);
    int tail_length = tail_end - tail_start;
    // more stringent requirement of low variance for shorter tails
    if (tail_length > 20) {
      if (tail_variance > tail_prox_transcript_variance / 2.5) 	{ // fake tail
	tail_start = 0;
	tail_end = 0;
	tail_length = 0;
	is_a_low_confidence_tail = true;
	}
    } else {
      if (tail_variance > tail_prox_transcript_variance / 3.5) { // fake tail
	tail_start = 0;
	tail_end = 0;
	tail_length = 0;
	is_a_low_confidence_tail = true;
      }
    }
    // std::cout <<  "tail vatiance: " << tail_variance << "\t" << "tail prox variance:" << tail_prox_transcript_variance << "\n";
    if(is_a_low_confidence_tail){
      std::cout  << "this is a low confidence tail\n" ;
      // std::cout << "mid_notail: " << mid_notail << "\n";
    }
      return {tail_start, tail_end};
  }
  return {0, 0};
}



std::vector<int> find_dna_tail_per_read(std::vector<c10::Half> raw_data,bool fwd, int anchor) {

  // Parameters
  const int samples_per_nt = 10;
  const c10::Half SPIKE_THRESHOLD = 2.0;
  const int MOVING_WINDOW_SIZE = 30;
  const int MAX_GAP_BETWEEN_TAILS = 20 * samples_per_nt; 
  const c10::Half SLOPE_THRESHOLD = 0.20;

  

  std::vector<c10::Half> norm_data = z_normalize(raw_data);   // Z-normalize the data
  if (fwd) {
    std::reverse(norm_data.begin(), norm_data.end());   // Reverse polyA reads if necessary
  }
  
  int read_length = norm_data.size();
  int tail_start = (fwd) ? (read_length - anchor) : anchor;  
  std::vector<c10::Half> rectified_data = rectify(norm_data); //abs value of data
  std::vector<c10::Half> truncated_data = truncate_spikes(rectified_data, SPIKE_THRESHOLD); //cut everything > threshold
  std::vector<c10::Half> smoothed_data_lr = left_to_right_sliding_window_dna(truncated_data, MOVING_WINDOW_SIZE, 1);
  std::vector<c10::Half> smoothed_data_rl = right_to_left_sliding_window_dna(truncated_data, MOVING_WINDOW_SIZE, 1);
  std::vector<c10::Half> smoothed_data(smoothed_data_lr.size(), 0); //initialize vector with 0

  for (size_t i = 0; i < smoothed_data.size(); ++i) {
    smoothed_data[i] = std::min(smoothed_data_lr[i], smoothed_data_rl[i]);
  }

  int window_size = 20;
  int step_size = 20;
  int start = tail_start;
  int end =  read_length;
  std::vector<c10::Half> data(truncated_data.begin() + start, truncated_data.begin() + end); //initialize the data vector with the truncated data
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
  int j = 0;
  std::vector<int> pass;
  int poly_start_seach_window = 30;
  
  while (k < poly_start_seach_window && k + 3 < slope.size()) {
    bool slope_in_range = std::abs(slope[k]) < SLOPE_THRESHOLD &&
      std::abs(slope[k + 1]) < SLOPE_THRESHOLD; // &&
    // std::abs(slope[k + 2]) < SLOPE_THRESHOLD &&
    // std::abs(slope[k + 3]) < SLOPE_THRESHOLD;
    bool mean_in_range =  mean_data[k] < SLOPE_THRESHOLD + 0.3 &&
					 mean_data[k + 1] < SLOPE_THRESHOLD + 0.3; // &&
							    // mean_data[k + 2] < SLOPE_THRESHOLD + 0.3 &&
    // mean_data[k + 3] < SLOPE_THRESHOLD + 0.3;
    bool mean_way_out =  mean_data[k] > SLOPE_THRESHOLD + 0.4 &&
      mean_data[k + 1] > SLOPE_THRESHOLD + 0.4; // &&
      // mean_data[k + 2] > SLOPE_THRESHOLD + 0.4 ;// &&
      // mean_data[k + 3] > SLOPE_THRESHOLD + 0.4; //this is to avoid fake starts
    
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

  // for(int i=0; i< pass.size(); i++){
  //   std::cout << "pass: " << pass[i] << "\n";
  // }

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
      std::vector<int> pass2(pass.begin(), zero_pos);
      k=pass2.size();
    }
  // k=54;
  if (k != poly_start_seach_window) {
    precise_tail_start = k * window_size + tail_start; // get precise start
  }
  // std::cout << "k: " <<  k << "\n" ;
  // std::cout << "index: " <<  k * window_size + tail_start << "\n" ;
  int tail_end=0;
  int width = 0;
  int last_good_end = 0;
  if (precise_tail_start) { //else, we will return tail_start=0;
    tail_start = *precise_tail_start;
    // Determine tail end
    int i = k; //start the search from the precise boundaries (determined above)
    c10::Half sm_data_threshold = fwd ? 0.6 : 0.4;
    
    while (i < slope.size()) {
      if (std::abs(slope[i]) < SLOPE_THRESHOLD && smoothed_data[tail_start + i * window_size] < sm_data_threshold) {
	tail_end = i;
	++i;
      } else { // if there is gap in the tail
	last_good_end = i;
	// std::cout << slope[i] << " < "  << SLOPE_THRESHOLD << " && " <<  smoothed_data[tail_start + i * window_size] << " < "  <<  sm_data_threshold << "\n";
	// std::cout << "last good end = " <<  read_length - ((i-k) * window_size  + tail_start)<< "\n" ;
	int small_glitch_count = 0;
	bool quit_searching = false;
	while (i < slope.size())
	  {
	    // std::cout << "small_glitch_count: " <<  small_glitch_count << "\n";
	    if (i - last_good_end  > (MAX_GAP_BETWEEN_TAILS / window_size)){
	      quit_searching = true;
	      break;
	    }
 	    if (std::abs(mean_data[i]) > SLOPE_THRESHOLD + 0.3) { // if 6 small glitches in a row, come back into the main loop 
	      small_glitch_count = 0;
	      // std::cout << "mean_data = " << std::abs(mean_data[i]) << " > " << SLOPE_THRESHOLD + 0.3 <<  "\n";
	    } else if (++small_glitch_count > 3) {
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
    std::vector<int> best_interval = {final_tail_start, final_tail_end};
    // std::cout << "final_tail_start: " << final_tail_start << "   final_tail_end: " << final_tail_end << "";
    best_interval = checkTailConfidence(final_tail_start, final_tail_end, raw_data, read_length, fwd);
    return best_interval;
  }
  return {0,0};
}



typedef struct ReadData{
  std::string read_name;
  int anchor;
  std::vector<c10::Half> raw_data;
  std::vector<int> index;
  std::vector<std::string> moves;
  int samples_per_nt;
  bool fwd;
} ReadData;

ReadData read_tsv(std::string file_name){
  std::ifstream file(file_name); 
    if (!file.is_open()) {
      throw std::runtime_error("Error: filename is empty");
    }
    std::vector<int> column1; 
    std::vector<c10::Half> column2;
    std::vector<std::string> column3; 
    
    std::string line;
    int i=0;
    while (std::getline(file, line)) {
      // Stop if  empty line
      i++;
      std::istringstream iss(line);
      std::string value;
      
      // Read each column value for the current line
      int col1_val;
      double col2_val;
      std::string col3_val;
      
      // Assuming three columns; adjust as needed
      if (std::getline(iss, value, '\t')) col1_val = std::stoi(value);
      if (std::getline(iss, value, '\t')) col2_val = std::stod(value);
      if (std::getline(iss, value, '\t')) col3_val = value;
      // std::cout << col1_val  << "\n";
      
      // Store each value in its corresponding vector
      column1.push_back(col1_val);
      column2.push_back(col2_val);
      column3.push_back(col3_val);
    }
    std::size_t pos_start = file_name.rfind('/') + 1;
    std::size_t pos_end_uuid = file_name.find('_', pos_start + 1);
    std::size_t pos_start_value = file_name.find('_', pos_end_uuid + 1);
    std::size_t pos_end_value = file_name.find('_', pos_start_value + 1);
    // Extract the samples per nt
    std::size_t pos_start_samplesNt =  pos_end_value  ;
    std::size_t pos_end_samplesNt =  file_name.find('_', pos_start_samplesNt + 1);
    pos_start_samplesNt = file_name.find('_', pos_end_samplesNt + 1 );
    pos_end_samplesNt = file_name.find('_', pos_start_samplesNt + 1);
    std::string samples_per_nt  = file_name.substr(pos_start_samplesNt  + 1 , pos_end_samplesNt  - pos_start_samplesNt -1  );
    // Extract the samples per nt
    std::size_t pos_start_strand = file_name.rfind('_') + 1;
    std::size_t pos_end_strand = file_name.rfind('.');
    // Extract the UUID
    std::string read_name = file_name.substr(pos_start, pos_end_uuid - pos_start);
    // Extract the value after the UUID
    std::string value = file_name.substr(pos_start_value +1, pos_end_value - pos_start_value -1);
    // Extract the "strand direction" part
    std::string strand = file_name.substr(pos_start_strand, pos_end_strand - pos_start_strand);
    bool fwd;
    strand == "plus" ? fwd = true : fwd = false;
    int samples_per_nt_int = std::stoi(samples_per_nt);
    int anchor = std::stoi(value);
    ReadData read_data = {read_name, anchor, column2, column1, column3, samples_per_nt_int, fwd};
    return read_data;
}

void write_tsv(ReadData read_data, std::vector<int> interval, std::string dir_out){
  std::string strand = read_data.fwd ? "plus" : "minus";
  std::string file_name = dir_out + "/" +
    read_data.read_name + "_" +
    std::to_string(int(interval[1] - interval[0])/read_data.samples_per_nt) + "_" +
    std::to_string(read_data.anchor) +  "_" +
    std::to_string(interval[0]) + "_" +
    std::to_string(interval[1]) + "_" +
    std::to_string(read_data.samples_per_nt) + "_0_0_"  + strand + ".tsv";
  std::cout << file_name << "\n";
  if (!fs::exists(dir_out)) {
    fs::create_directory(dir_out);
  }
  std::ofstream file(file_name);
  for (int i = 0; i < read_data.raw_data.size(); i++) {
    file << read_data.index[i] << "\t" << read_data.raw_data[i] << "\t" << read_data.moves[i] << "\n";
  }
  file.close();
}




// int main(int argc, char** argv)
// {
//   std::string file_name;
//   std::string dir_name = argv[1];
//   std::string dir_out = argv[2];
//   for (const auto & entry : fs::directory_iterator(dir_name)) {
//     file_name = entry.path();
//     ReadData read_data = read_tsv(file_name);
//     std::string read_name = read_data.read_name;
//     int anchor = read_data.anchor;
//     std::vector<c10::Half> raw_data = read_data.raw_data;
//     bool fwd = read_data.fwd;
//     std::vector<int> interval = find_dna_tail_per_read( raw_data, fwd, anchor);
//     write_tsv(read_data, interval, dir_out);
//     // std::cout  << read_name  << "\t;\t start tail: " << interval[0] << "\t;\tend tail: " << interval[1] << "\t;\ttail_length: " << interval[1]-interval[0]  << "\t;\tanchor: "  << anchor << "\n" ;
//   }
// }



void process_file(const std::string &file_name, const std::string &dir_out) {
    ReadData read_data = read_tsv(file_name);
    std::string read_name = read_data.read_name;
    int anchor = read_data.anchor;
    std::vector<c10::Half> raw_data = read_data.raw_data;
    bool fwd = read_data.fwd;
    std::vector<int> interval = find_dna_tail_per_read(raw_data, fwd, anchor);
    write_tsv(read_data, interval, dir_out);
}

int main(int argc, char** argv) {
    std::string dir_name = argv[1];
    std::string dir_out = argv[2];
    const int max_threads = 40;
    std::vector<std::future<void>> futures;

    for (const auto &entry : fs::directory_iterator(dir_name)) {
        std::string file_name = entry.path();
        
        // Launch a new task in async mode for each file
        futures.push_back(std::async(std::launch::async, process_file, file_name, dir_out));        
        // Limit concurrent tasks
        if (futures.size() >= max_threads) {
            // Wait for the first batch of tasks to complete
            for (auto &f : futures) {
                f.get();
            }
            futures.clear(); // Clear futures to start new batch
        }
    }
    // Ensure all remaining tasks complete
    for (auto &f : futures) {
        f.get();
    }

    return 0;
}

