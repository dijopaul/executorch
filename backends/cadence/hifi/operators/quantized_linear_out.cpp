/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>
#include <cmath>

namespace impl {
namespace HiFi {
namespace native {

using Tensor = exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

void quantized_linear_out(
    KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    int64_t src_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    const exec_aten::optional<Tensor>& offset,
    Tensor& out) {
  // input comes in shape [leading_dims, in_dim]
  // weight comes in shape [out_dim, in_dim]
  // output comes in empty with shape [leading_dims, out_dim]
  // Perform matrix multiply (M x N) x (N x P)' => M x P
  int64_t leading_dims = getLeadingDims(src, src.dim() - 1);
  int64_t out_dim = weight.size(0); // = out_dim
  int64_t in_dim = weight.size(1); // = in_dim

  if (src.scalar_type() == exec_aten::ScalarType::Byte) {
    const uint8_t* __restrict__ in_data = src.const_data_ptr<uint8_t>();
    const uint8_t* __restrict__ weight_data = weight.const_data_ptr<uint8_t>();
    const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
    uint8_t* __restrict__ out_data = out.mutable_data_ptr<uint8_t>();

    // The nnlib kernel to compute quantized linear via matmul.
    xa_nn_matmul_asym8uxasym8u_asym8u(
        out_data,
        weight_data,
        in_data,
        bias_data,
        out_dim,
        in_dim,
        in_dim,
        leading_dims,
        in_dim,
        out_dim,
        1,
        -weight_zero_point.const_data_ptr<int32_t>()[0],
        -src_zero_point,
        out_multiplier.const_data_ptr<int32_t>()[0],
        out_shift.const_data_ptr<int32_t>()[0],
        out_zero_point);
  } else {
    const int8_t* __restrict__ in_data = src.const_data_ptr<int8_t>();
    const int8_t* __restrict__ weight_data = weight.const_data_ptr<int8_t>();
    const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
    int8_t* __restrict__ out_data = out.mutable_data_ptr<int8_t>();

    xa_nn_matmul_asym8sxasym8s_asym8s(
        out_data,
        weight_data,
        in_data,
        bias_data,
        out_dim,
        in_dim,
        in_dim,
        leading_dims,
        in_dim,
        out_dim,
        1,
        -weight_zero_point.const_data_ptr<int32_t>()[0],
        -src_zero_point,
        out_multiplier.const_data_ptr<int32_t>()[0],
        out_shift.const_data_ptr<int32_t>()[0],
        out_zero_point);
  }
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl
