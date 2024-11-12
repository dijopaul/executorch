/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/math_constants.h>
#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <stdio.h>
#include <cmath>
#include <executorch/kernels/portable/cpu/util/activation_ops_util.cpp>

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using string_view = exec_aten::string_view;
using executorch::aten::RuntimeContext;
using torch::executor::Error;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

Tensor& gelu_out(
    RuntimeContext& ctx,
    const Tensor& in,
    string_view approximate,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      torch::executor::check_gelu_args(in, approximate, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  ScalarType out_type = out.scalar_type();

  int optimized = 1;

  if (out_type != ScalarType::Float)
    optimized = 0;

  if (optimized == 1) {
    int approx;
    if (approximate == "tanh")
      approx = 1;
    else if (approximate == "none")
      approx = 0;
    else
      printf("Invalid approximation format: %.*s for gelu", approximate);

    float* p_in = (float*)in.const_data_ptr<float>();
    float* p_out = out.mutable_data_ptr<float>();
    xa_nn_vec_gelu_f32_f32(p_out, p_in, in.numel(), approx);
    return out;
  }

  ET_SWITCH_FLOAT_TYPES(in.scalar_type(), ctx, "gelu", CTYPE, [&]() {
    if (approximate == "tanh") {
      torch::executor::apply_unary_map_fn(
          [](const CTYPE x) {
            const CTYPE kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
            const CTYPE kKappa = static_cast<float>(0.044715);

            const CTYPE x_cubed = x * x * x;
            const CTYPE inner = kBeta * (x + kKappa * x_cubed);
            const CTYPE ret = 0.5 * x * (1 + std::tanh(inner));

            return ret;
          },
          in.const_data_ptr<CTYPE>(),
          out.mutable_data_ptr<CTYPE>(),
          in.numel());
    } else if (approximate == "none") {
      torch::executor::apply_unary_map_fn(
          [](const CTYPE x) { return 0.5 * x * (1 + std::erf(x * M_SQRT1_2)); },
          in.const_data_ptr<CTYPE>(),
          out.mutable_data_ptr<CTYPE>(),
          in.numel());
    } else {
      ET_CHECK_MSG(
          false,
          "Invalid approximation format: %.*s for gelu",
          static_cast<int>(approximate.length()),
          approximate.data());
    }
  });

  return out;
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl
}; // namespace cadence