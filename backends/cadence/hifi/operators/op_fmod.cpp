/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include "kernels.h"

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {
template <
    bool can_cast,
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct FmodInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct FmodInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void
  run(const Tensor& a, const Tensor& b, Tensor& out, bool& div_by_zero_error) {
    apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [&div_by_zero_error](const CTYPE_A val_a, const CTYPE_B val_b) {
          if (is_integral_type<CTYPE_IN, /*includeBool=*/true>::value) {
            if (val_b == 0) {
              div_by_zero_error = true;
              return static_cast<CTYPE_OUT>(0);
            }
          }
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value = std::fmod(a_casted, b_casted);

          return static_cast<CTYPE_OUT>(value);
        },
        a,
        b,
        out);
  }
};

struct ReportCanCastBug {
  static void run(const Tensor&, const Tensor&, Tensor&, bool&) {
    ET_DCHECK_MSG(false, "BUG: canCast should have been checked above");
  }
};

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct FmodInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug {};

} // namespace

Tensor& fmod_Tensor_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {

  // Determine output size and resize for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);
      
  if((a.scalar_type() == ScalarType::Float)||(b.scalar_type() == ScalarType::Float))
  {

    const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
    const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
    const bool any_is_broadcasted = (a_is_broadcasted || b_is_broadcasted);

    if(any_is_broadcasted)
    {
      FLOAT32 * __restrict__ p_out = (FLOAT32 * __restrict__ )out.mutable_data_ptr<float>();
      const FLOAT32 * __restrict__ p_inp1 = (const FLOAT32 * __restrict__)a.const_data_ptr<float>();
      const FLOAT32 * __restrict__ p_inp2 = (const FLOAT32 * __restrict__)b.const_data_ptr<float>();
      
      WORD32 p_out_shape[4];
      WORD32 p_inp1_shape[4];
      WORD32 p_inp2_shape[4];
      
      for(int i = 0; i < 4; i++)
      {
        p_inp1_shape[i] = 1;
        p_inp2_shape[i] = 1;
        p_out_shape[i] = 1;
      }
        
      int off_o = 4 - out.dim();        
      int off_a = 4 - a.dim();
      int off_b = 4 - b.dim();

      for(int i = 0; i < out.dim(); i++)
        p_out_shape[i+off_o] = out.size(i);
      for(int i = 0; i < a.dim(); i++)
            p_inp1_shape[i+off_a] = a.size(i);
      for(int i = 0; i < b.dim(); i++)
          p_inp2_shape[i+off_b] = b.size(i);
      
      WORD32 val = xa_nn_elm_fmod_broadcast_4D_f32xf32_f32(p_out,
                                                          p_out_shape,
                                                          p_inp1,
                                                          p_inp1_shape,
                                                          p_inp2,
                                                          p_inp2_shape);
    }
    else
    {
      FLOAT32 * __restrict__ p_out = (FLOAT32 * __restrict__ )out.mutable_data_ptr<float>();
      const FLOAT32 * __restrict__ p_inp1 = (const FLOAT32 * __restrict__)a.const_data_ptr<float>();
      const FLOAT32 * __restrict__ p_inp2 = (const FLOAT32 * __restrict__)b.const_data_ptr<float>();
      
      WORD32 num_elm = out.numel();
        
      WORD32 val = xa_nn_elm_fmod_f32xf32_f32(p_out,
                                            p_inp1,
                                            p_inp2,
                                            num_elm);
    }
  }
  else
  {
    ScalarType a_type = a.scalar_type();
    ScalarType b_type = b.scalar_type();
    ScalarType common_type = promoteTypes(a_type, b_type);
    ScalarType out_type = out.scalar_type();
  
    ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);
  
    auto div_by_zero_error = false;
  
    ET_SWITCH_REAL_TYPES_AND(
        Bool, a_type, ctx, "fmod.Tensor_out", CTYPE_A, [&]() {
          ET_SWITCH_REAL_TYPES_AND(
              Bool, b_type, ctx, "fmod.Tensor_out", CTYPE_B, [&]() {
                using CTYPE_IN = typename torch::executor::
                    promote_types<CTYPE_A, CTYPE_B>::type;
                ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
                ET_SWITCH_REAL_TYPES(
                    out_type, ctx, "fmod.Tensor_out", CTYPE_OUT, [&]() {
                      FmodInner<
                          !std::is_same<CTYPE_IN, bool>::value &&
                              can_cast<CTYPE_IN, CTYPE_OUT>::value,
                          CTYPE_A,
                          CTYPE_B,
                          CTYPE_IN,
                          CTYPE_OUT>::run(a, b, out, div_by_zero_error);
                    });
              });
        });
  
    ET_KERNEL_CHECK_MSG(
        ctx,
        !div_by_zero_error,
        InvalidArgument,
        out,
        "Fmod operation encountered integer division by zero");
  }

  return out;
}

Tensor& fmod_Scalar_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, a.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType common_type = utils::promote_type_with_scalar(a_type, b);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  // Check for integer division by zero
  if (isIntegralType(common_type, /*includeBool=*/true)) {
    auto is_zero = false;
    ET_SWITCH_REAL_TYPES_AND(
        Bool, b_type, ctx, "fmod.Scalar_out", CTYPE_B, [&]() {
          CTYPE_B val_b = 0;
          utils::extract_scalar(b, &val_b);
          is_zero = (val_b == 0);
        });

    ET_KERNEL_CHECK_MSG(
        ctx,
        !is_zero,
        InvalidArgument,
        out,
        "Fmod operation encountered integer division by zero");
  }

  ET_SWITCH_REAL_TYPES_AND(
      Bool, a_type, ctx, "fmod.Scalar_out", CTYPE_A, [&]() {
        ET_SWITCH_SCALAR_OBJ_TYPES(
            b_type, ctx, "fmod.Scalar_out", CTYPE_B, [&]() {
              CTYPE_B val_b = 0;
              utils::extract_scalar(b, &val_b);
              ET_SWITCH_REAL_TYPES(
                  common_type, ctx, "fmod.Scalar_out", CTYPE_IN, [&]() {
                    ET_SWITCH_REAL_TYPES(
                        out_type, ctx, "fmod.Scalar_out", CTYPE_OUT, [&]() {
                          apply_unary_map_fn(
                              [val_b](const CTYPE_A val_a) {
                                CTYPE_IN a_casted =
                                    static_cast<CTYPE_IN>(val_a);
                                CTYPE_IN b_casted =
                                    static_cast<CTYPE_IN>(val_b);
                                CTYPE_IN value = std::fmod(a_casted, b_casted);

                                return static_cast<CTYPE_OUT>(value);
                              },
                              a.const_data_ptr<CTYPE_A>(),
                              out.mutable_data_ptr<CTYPE_OUT>(),
                              out.numel());
                        });
                  });
            });
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
