/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using torch::executor::Error;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

// view_copy.out(Tensor self, int[] size, *, Tensor(a!) out) -> Tensor(a!)
Tensor& view_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    exec_aten::ArrayRef<int64_t> size_int64_t,
    Tensor& out) {
  (void)ctx;

  Tensor::SizesType expected_output_size[16];
  ET_KERNEL_CHECK(
      ctx,
      torch::executor::get_view_copy_target_size(
          self, size_int64_t, out.dim(), expected_output_size),
      InvalidArgument,
      out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      executorch::runtime::resize_tensor(
          out, {expected_output_size, static_cast<size_t>(out.dim())}) ==
          Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(self, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensor_is_default_dim_order(self),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      torch::executor::check_view_copy_args(self, size_int64_t, out),
      InvalidArgument,
      out);

  if (self.nbytes() > 0) {
    memcpy(out.mutable_data_ptr(), self.const_data_ptr(), self.nbytes());
  }
  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence