/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <cmath>

#include "kernels.h"

namespace torch {
namespace executor {
namespace native {

namespace {

ScalarType get_compute_type(ScalarType a_type, ScalarType b_type) {
  if (isFloatingType(a_type) && isFloatingType(b_type)) {
    return promoteTypes(a_type, b_type);
  } else if (isFloatingType(a_type)) {
    return a_type;
  } else if (isFloatingType(b_type)) {
    return b_type;
  }
  return ScalarType::Float;
}

} // namespace

Tensor&
div_out(RuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();

  ET_KERNEL_CHECK(
      ctx,
      !isComplexType(a_type) && !isQIntType(a_type) && !isBitsType(a_type),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      !isComplexType(b_type) && !isQIntType(b_type) && !isBitsType(b_type),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(ctx, tensor_is_real_type(out), InvalidArgument, out);
  
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
      
      WORD32 val = xa_nn_elm_div_broadcast_4D_f32xf32_f32(p_out,
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
        
      WORD32 val = xa_nn_elm_div_f32xf32_f32(p_out,
                                            p_inp1,
                                            p_inp2,
                                            num_elm);
    }
  }
  else
  {
    ScalarType common_type = get_compute_type(a_type, b_type);
    ScalarType out_type = out.scalar_type();
  
    ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);
  
    ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "div.out", CTYPE_A, [&]() {
      ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, "div.out", CTYPE_B, [&]() {
        ET_SWITCH_FLOAT_TYPES(common_type, ctx, "div.out", CTYPE_IN, [&]() {
          ET_SWITCH_FLOAT_TYPES(out_type, ctx, "div.out", CTYPE_OUT, [&]() {
            apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
                [](const CTYPE_A val_a, const CTYPE_B val_b) {
                  CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                  CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                  CTYPE_IN value = a_casted / b_casted;
  
                  return static_cast<CTYPE_OUT>(value);
                },
                a,
                b,
                out);
          });
        });
      });
    });
  }

  return out;
}

Tensor& div_out_mode(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    exec_aten::optional<exec_aten::string_view> mode,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = get_compute_type(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, tensor_is_real_type(out), InvalidArgument, out);

  // Allow casting float -> integral here
  // non-bool -> bool is still disallowed
  ET_KERNEL_CHECK(
      ctx,
      !(common_type != ScalarType::Bool && out_type == ScalarType::Bool),
      InvalidArgument,
      out);
      
  if(common_type == ScalarType::Float)
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
      
      WORD32 val = xa_nn_elm_floor_div_broadcast_4D_f32xf32_f32(p_out,
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
        
      WORD32 val = xa_nn_elm_floor_div_f32xf32_f32(p_out,
                                            p_inp1,
                                            p_inp2,
                                            num_elm);
    }
  }
  else
  {
    ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "div.out_mode", CTYPE_A, [&]() {
      ET_SWITCH_REAL_TYPES_AND(Bool, b_type, ctx, "div.out_mode", CTYPE_B, [&]() {
        ET_SWITCH_FLOAT_TYPES(common_type, ctx, "div.out_mode", CTYPE_IN, [&]() {
          ET_SWITCH_REAL_TYPES(out_type, ctx, "div.out_mode", CTYPE_OUT, [&]() {
            apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
                [mode](const CTYPE_A val_a, const CTYPE_B val_b) {
                  CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                  CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                  CTYPE_IN value = a_casted / b_casted;
                  if (mode.has_value() && mode.value() == "trunc") {
                    value = std::trunc(value);
                  } else if (mode.has_value() && mode.value() == "floor") {
                    value = std::floor(value);
                  }
                  return static_cast<CTYPE_OUT>(value);
                },
                a,
                b,
                out);
          });
        });
      });
    });
  }

  return out;
}

Tensor& div_scalar_out(
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
  ScalarType common_type = isFloatingType(a_type) ? a_type : ScalarType::Float;
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "div.Scalar_out", CTYPE_A, [&]() {
    ET_SWITCH_SCALAR_OBJ_TYPES(b_type, ctx, "div.Scalar_out", CTYPE_B, [&]() {
      ET_SWITCH_FLOAT_TYPES(out_type, ctx, "div.Scalar_out", CTYPE, [&]() {
        CTYPE_B b_val;
        utils::extract_scalar(b, &b_val);
        CTYPE b_casted = static_cast<CTYPE>(b_val);

        apply_unary_map_fn(
            [b_casted](const CTYPE_A val_a) {
              CTYPE a_casted = static_cast<CTYPE>(val_a);
              CTYPE value = a_casted / b_casted;
              return static_cast<CTYPE>(value);
            },
            a.const_data_ptr<CTYPE_A>(),
            out.mutable_data_ptr<CTYPE>(),
            out.numel());
      });
    });
  });

  return out;
}

Tensor& div_scalar_mode_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    exec_aten::optional<exec_aten::string_view> mode,
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
  
  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);
  
  constexpr auto name = "div.Scalar_mode_out";
  
  ET_SWITCH_REALB_TYPES(a_type, ctx, name, CTYPE_A, [&]() {
    ET_SWITCH_SCALAR_OBJ_TYPES(b_type, ctx, name, CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES(out_type, ctx, name, CTYPE, [&]() {
        CTYPE_B b_val;
        utils::extract_scalar(b, &b_val);
        CTYPE b_casted = static_cast<CTYPE>(b_val);
  
        apply_unary_map_fn(
            [b_casted, mode](const CTYPE_A val_a) {
              CTYPE a_casted = static_cast<CTYPE>(val_a);
              CTYPE value = a_casted / b_casted;
              if (mode.has_value() && mode.value() == "trunc") {
                value = std::trunc(value);
              } else if (mode.has_value() && mode.value() == "floor") {
                value = utils::floor_divide(a_casted, b_casted);
              }
              return value;
            },
            a.const_data_ptr<CTYPE_A>(),
            out.mutable_data_ptr<CTYPE>(),
            out.numel());
      });
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
