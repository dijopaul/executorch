/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using exec_aten::ScalarType;
using exec_aten::Tensor;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using torch::executor::KernelRuntimeContext;
using torch::executor::Error;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

Tensor& masked_fill_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& mask,
    const Scalar& value,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, torch::executor::check_masked_fill_args(in, mask, value, out), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  ScalarType val_type = torch::executor::native::utils::get_scalar_dtype(value);

  ET_KERNEL_CHECK(
      ctx,
      torch::executor::resize_to_broadcast_target_size(in, mask, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, executorch::runtime::tensors_have_same_dim_order(in, mask, out), InvalidArgument, out);

  ET_SWITCH_REAL_TYPES_AND(
      Bool, in_type, ctx, "masked_fill.Scalar_out", CTYPE, [&]() {
        ET_SWITCH_REAL_TYPES_AND(
            Bool, val_type, ctx, "masked_fill.Scalar_out", CTYPE_VAL, [&]() {
              CTYPE_VAL value_v;
              torch::executor::native::utils::extract_scalar(value, &value_v);
              CTYPE val = static_cast<CTYPE>(value_v);

              torch::executor::apply_binary_elementwise_fn<CTYPE, bool, CTYPE>(
                  [val](const CTYPE val_in, const bool val_mask) {
                    return val_mask ? val : val_in;
                  },
                  in,
                  mask,
                  out);
            });
      });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence