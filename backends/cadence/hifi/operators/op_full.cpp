/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <stdio.h>

namespace impl {
namespace HiFi {
namespace native {

using exec_aten::IntArrayRef;
using exec_aten::RuntimeContext;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::Error;
using torch::executor::native::utils::extract_scalar;
using torch::executor::native::utils::get_scalar_dtype;

Tensor& full_out(
    RuntimeContext& ctx,
    const IntArrayRef sizes,
    const Scalar& fill_value,
    Tensor& out) {
  (void)ctx;

  ScalarType val_type = get_scalar_dtype(fill_value);
  ScalarType out_type = out.scalar_type();

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, sizes) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  constexpr auto name = "full.out";

  bool optimized = 0;
  if (out_type == ScalarType::Long || out_type == ScalarType::Float ||
      out_type == ScalarType::Byte || out_type == ScalarType::Char)
    optimized = 1;

  if (optimized) {
    if (out_type == ScalarType::Long) {
      int* data_out = out.mutable_data_ptr<int>();
      int64_t val = fill_value.to<int64_t>();
      int val_casted = static_cast<int>(val);
      for (size_t i = 0; i < out.numel(); ++i) {
        data_out[i] = val_casted;
      }
    } else if (out_type == ScalarType::Float) {
      float* data_out = out.mutable_data_ptr<float>();
      double val = fill_value.to<double>();
      float val_casted = static_cast<float>(val);
      xa_nn_memset_f32_f32(data_out, val_casted, out.numel());
    } else if (out_type == ScalarType::Byte || out_type == ScalarType::Char) {
      char* data_out = out.mutable_data_ptr<char>();
      int val = fill_value.to<int64_t>();
      memset((void*)data_out, val, out.numel());
    }

    return out;
  }

  ET_SWITCH_SCALAR_OBJ_TYPES(val_type, ctx, name, CTYPE_VAL, [&] {
    CTYPE_VAL val;
    extract_scalar(fill_value, &val);

    ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, "full", CTYPE_OUT, [&] {
      CTYPE_OUT val_casted = static_cast<CTYPE_OUT>(val);
      auto data_out = out.mutable_data_ptr<CTYPE_OUT>();
      for (size_t i = 0; i < out.numel(); ++i) {
        data_out[i] = val_casted;
      }
    });
  });
  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
