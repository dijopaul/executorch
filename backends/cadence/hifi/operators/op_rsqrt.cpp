/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>

using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::aten::RuntimeContext;

namespace impl {
namespace HiFi {
namespace native {
namespace {

double rsqrt(double x) {
  return 1.0 / std::sqrt(x);
}

} // namespace

Tensor& rsqrt_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  bool optimized = 1;

  if (out.scalar_type() != ScalarType::Float)
    optimized = 0;

  if (!optimized) {
    WORD32 num_elm = out.numel();

    FLOAT32* __restrict__ p_out =
        (FLOAT32* __restrict__)out.mutable_data_ptr<float>();
    const FLOAT32* __restrict__ p_inp =
        (const FLOAT32* __restrict__)in.const_data_ptr<float>();

    xa_nn_elm_rsqrt_f32_f32(p_out, p_inp, num_elm);
    return out;
  } else
    return torch::executor::native::internal::unary_ufunc_realhb_to_floath(
        rsqrt, ctx, in, out);
}

} // namespace native
} // namespace HiFi
} // namespace impl
