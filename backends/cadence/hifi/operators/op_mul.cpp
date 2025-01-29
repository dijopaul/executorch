/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/dtype_util.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

using executorch::aten::RuntimeContext;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::can_cast;
using executorch::runtime::CppTypeToScalarType;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::Error;
using torch::executor::native::utils::apply_unitensor_elementwise_fn;
using torch::executor::native::utils::get_compute_type;
using torch::executor::native::utils::promote_type_with_scalar;
using torch::executor::native::utils::scalar_to;
using torch::executor::native::utils::SupportedTensorDtypes;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

namespace {
template <
    bool can_cast,
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct MulInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct MulInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void run(const Tensor& a, const Tensor& b, Tensor& out) {
    torch::executor::apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [](const CTYPE_A val_a, const CTYPE_B val_b) {
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value = a_casted * b_casted;

          return static_cast<CTYPE_OUT>(value);
        },
        a,
        b,
        out);
  }
};

struct ReportCanCastBug {
  static void run(const Tensor&, const Tensor&, Tensor&) {
    ET_DCHECK_MSG(false, "BUG: canCast should have been checked above");
  }
};

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct MulInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug {};
} // namespace

Tensor&
mul_out(RuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      torch::executor::resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensor_is_realhb_type(out),
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type =
      executorch::runtime::promoteTypes(a_type, b_type, /*half_to_float*/ true);
  ScalarType out_type = out.scalar_type();
  constexpr int kNnlibMaxDim = 4; /*fallback if broadcast and dim > 4 */

  int a_dim = a.dim(), b_dim = b.dim(), out_dim = out.dim();
  bool optimized = 1;
  /*find broadcast*/
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool broadcast = (a_is_broadcasted || b_is_broadcasted);
  int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
  max_dim = out.dim() > max_dim ? out.dim() : max_dim;

  if ((a_type != ScalarType::Float) || (b_type != ScalarType::Float))
    optimized = 0;

  if ((a_dim == 0) || (b_dim == 0))
    optimized = 0;

  if ((broadcast == 1) && (max_dim > kNnlibMaxDim))
    optimized = 0;

  if (optimized) {
    float* a_data = a.mutable_data_ptr<float>();
    float* b_data = b.mutable_data_ptr<float>();
    float* out_data = out.mutable_data_ptr<float>();

    if (broadcast == 1) {
      int out_shape[kNnlibMaxDim];
      int inp1_shape[kNnlibMaxDim];
      int inp2_shape[kNnlibMaxDim];
      for (int i = 0; i < kNnlibMaxDim; i++) {
        out_shape[i] = 1;
        inp1_shape[i] = 1;
        inp2_shape[i] = 1;
      }
      int off_o = kNnlibMaxDim - out.dim();
      int off_a = kNnlibMaxDim - a.dim();
      int off_b = kNnlibMaxDim - b.dim();
      for (int i = 0; i < out.dim(); i++)
        out_shape[i + off_o] = out.size(i);
      for (int i = 0; i < a.dim(); i++)
        inp1_shape[i + off_a] = a.size(i);
      for (int i = 0; i < b.dim(); i++)
        inp2_shape[i + off_b] = b.size(i);

      WORD32 ret_val = xa_nn_elm_mul_broadcast_4D_f32xf32_f32(
          out_data, out_shape, a_data, inp1_shape, b_data, inp2_shape);

      ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

    } else {
      WORD32 ret_val =
          xa_nn_elm_mul_f32xf32_f32(out_data, a_data, b_data, out.numel());

      ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
    }

    return out;
  }

  // Compute Dtype
  ScalarType compute_type = get_compute_type(common_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "mul.Scalar_out";

  ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    torch::executor::native::utils::
        apply_bitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
            [](const CTYPE_COMPUTE val_a, const CTYPE_COMPUTE val_b) {
              return val_a * val_b;
            },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            b,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBBF16);
  });

  return out;
}

Tensor& mul_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = promote_type_with_scalar(a.scalar_type(), b);

  // Check Common Dtype
  ET_KERNEL_CHECK(ctx, common_type == out.scalar_type(), InvalidArgument, out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // Compute Dtype
  ScalarType compute_type = get_compute_type(common_type);

  constexpr int kNnlibMaxDim = 4; /*fallback if broadcast and dim > 4 */

  int a_dim = a.dim();
  bool optimized = 1;

  if (common_type != ScalarType::Float)
    optimized = 0;

  if ((a_dim == 0) || (a_dim > kNnlibMaxDim))
    optimized = 0;

  if (optimized) {
    float* a_data = a.mutable_data_ptr<float>();
    const float val_b = scalar_to<float>(b);
    float* b_data = (float*)&(val_b);
    float* out_data = out.mutable_data_ptr<float>();

    int out_shape[kNnlibMaxDim];
    int inp1_shape[kNnlibMaxDim];
    int inp2_shape[kNnlibMaxDim];
    for (int i = 0; i < kNnlibMaxDim; i++) {
      out_shape[i] = 1;
      inp1_shape[i] = 1;
      inp2_shape[i] = 1;
    }
    int off_o = kNnlibMaxDim - out.dim();
    int off_a = kNnlibMaxDim - a.dim();
    int off_b = kNnlibMaxDim;
    for (int i = 0; i < out.dim(); i++)
      out_shape[i + off_o] = out.size(i);
    for (int i = 0; i < a.dim(); i++)
      inp1_shape[i + off_a] = a.size(i);

    WORD32 ret_val = xa_nn_elm_mul_broadcast_4D_f32xf32_f32(
        out_data, out_shape, a_data, inp1_shape, b_data, inp2_shape);

    ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

    return out;
  }
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "mul.Scalar_out";

  ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    const CTYPE_COMPUTE val_b = scalar_to<CTYPE_COMPUTE>(b);
    apply_unitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
        [val_b](const CTYPE_COMPUTE val_a) { return val_a * val_b; },
        ctx,
        a,
        SupportedTensorDtypes::REALHBBF16,
        out,
        SupportedTensorDtypes::SAME_AS_COMMON);
  });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
