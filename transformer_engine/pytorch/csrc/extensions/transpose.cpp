/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind.h>

#include <optional>
#include <vector>

#include "../extensions.h"
#include "pybind.h"

namespace transformer_engine {
namespace pytorch {

at::Tensor fp8_transpose(at::Tensor input, DType otype, std::optional<at::Tensor> output) {
  init_extension();

  const auto dim = input.dim();
  NVTE_CHECK(dim >= 2, "Need at least 2D tensor to transpose.");

  if (input.dim() > 2) {
    input = input.view({-1, input.size(dim - 1)});
  }

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  at::Tensor out;
  if (output.has_value()) {
    out = *output;
  } else {
    out = allocateTorchTensor(input.size(1), input.size(0), DType::kByte);
  }
  if (M == 0 || N == 0) return out;

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), std::vector<size_t>{M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(out.data_ptr(), std::vector<size_t>{N, M}, otype);

  nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return out;
}

py::object fp8_blockwise_transpose(py::object tensor, py::object quantizer) {
  init_extension();
  // Basic checks
  NVTE_CHECK(!tensor.is_none(), "Tensor has not been provided");
  NVTE_CHECK(detail::IsFloat8BlockwiseQuantizers(quantizer.ptr()),
             "Quantizer must be a Float8BlockwiseQuantizer");

  // Get intermediate dtype
  torch::Tensor torch_tensor = py::cast<torch::Tensor>(tensor);
  auto torch_dtype = torch_tensor.scalar_type();
  auto te_dtype = DType::kBFloat16;
  switch (torch_dtype) {
      case c10::ScalarType::Float:
          te_dtype = DType::kFloat32;
          break;  
      case c10::ScalarType::Half:
          te_dtype = DType::kFloat16;
          break;
      case c10::ScalarType::BFloat16:
          te_dtype = DType::kBFloat16;
          break;
      default:
          NVTE_ERROR("Unsupported dtype");
  }

  // Create TE tensor
  TensorWrapper te_tensor = makeTransformerEngineTensor(tensor, quantizer);
  NVTE_CHECK(nvte_tensor_scaling_mode(te_tensor.data()) == NVTE_BLOCK_SCALING_1D,
             "Only rowwise block scaling is supported for fp8 blockwise transpose");

  // Create Quantizer
  auto my_quantizer = static_cast<Float8BlockQuantizer*>(convert_quantizer(quantizer).get());
  TORCH_CHECK(my_quantizer->force_pow_2_scales,
              "Only power-of-2 scaling is supported for fp8 blockwise transpose");

  // Create QuantizationConfig
  QuantizationConfigWrapper quant_config;
  quant_config.set_force_pow_2_scales(my_quantizer->force_pow_2_scales);
  quant_config.set_amax_epsilon(my_quantizer->amax_epsilon);

  // Launch TE kernel
  nvte_transpose_blockwise(te_tensor.data(), quant_config, te_dtype, at::cuda::getCurrentCUDAStream());

  return tensor;
}

}  // namespace pytorch
}  // namespace transformer_engine
