#pragma once

#include <ATen/core/TensorBody.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace dorado::utils {

// Serialise Torch tensor to disk.
void serialise_tensor(const at::Tensor& t, const std::string& path);
// Load serialised tensor from disk.
std::vector<at::Tensor> load_tensors(const std::filesystem::path& dir,
                                     const std::vector<std::string>& tensors);

// Computes the q-th quantiles of each row of the input tensor `t`
// using a partial sort as opposed a full sort per torch::quantiles
// Only `interpolation='lower'` is currently implemented.
at::Tensor quantile(const at::Tensor& t, const at::Tensor& q);

// Computes the q-th quantiles of each row of the input tensor `t`
// using a counting sort which is extremely fast for low range integers.
// Only `interpolation='lower'` is currently implemented.
at::Tensor quantile_counting(const at::Tensor& t, const at::Tensor& q);

// Converts count float elements pointed to by src to half precision, with
// the result pointed to by dest.
void convert_f32_to_f16(c10::Half* dest, const float* src, std::size_t count);

// Copies count elements from src_offset elements into src to
// dest_elements into dst.  The tensors must be contiguous.
void copy_tensor_elems(at::Tensor& dest_tensor,
                       size_t dest_offset,
                       const at::Tensor& src_tensor,
                       std::size_t src_offset,
                       std::size_t count);

// Quantize a tensor to int8, returning a pair of tensors `{scales, quantized_tensor}`, where:
// `scales` is the same size as `tensor` with dimension 0 dropped, dtype float
// `quantized_tensor` is the same size as `tensor`, dtype int8
// such that `quantized_tensor / scales ~= tensor`
std::pair<at::Tensor, at::Tensor> quantize_tensor(const at::Tensor& tensor);

// Helper function to print tensor size.
std::string print_size(const at::Tensor& t, const std::string& name);

}  // namespace dorado::utils
