/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include "kernels.h"

namespace torch {
namespace executor {
namespace native {

#define NNLIB_MAX_DIM 4

Tensor& where_out(
    RuntimeContext& ctx,
    const Tensor& cond,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ScalarType cond_type = cond.scalar_type();
  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  // Determine output size and resize for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, cond, out) == Error::Ok,
      InvalidArgument,
      out);

  constexpr auto name = "where.self_out";

  ET_CHECK_MSG(
      cond_type == ScalarType::Bool || cond_type == ScalarType::Byte,
      "Unhandled dtype %s for where.self_out",
      torch::executor::toString(cond_type));

      /*logic to find broadcast*/
      const int a_is_broadcasted = !out.sizes().equals(a.sizes());
      const int b_is_broadcasted = !out.sizes().equals(b.sizes());
      const int cond_is_broadcasted = !out.sizes().equals(cond.sizes());
      const int broadcast = (a_is_broadcasted || b_is_broadcasted || cond_is_broadcasted);

      int max_dim = a.dim() > b.dim() ? a.dim() : b.dim();
      max_dim = out.dim() > max_dim ? out.dim() : max_dim;
      bool fall_back = 0;
      if((a_type != ScalarType::Float) || (b_type != ScalarType::Float))
        fall_back = 1;
    
      if((broadcast == 1) && (max_dim > NNLIB_MAX_DIM))
          fall_back = 1;

if(!fall_back)
{
      const float* a_data = a.const_data_ptr<float>();
      const float* b_data = b.const_data_ptr<float>();
      float* out_data = out.mutable_data_ptr<float>();
      const unsigned char* con = cond.const_data_ptr<uint8_t>();

      if(broadcast == 1)
      {
         int out_shape[NNLIB_MAX_DIM];
         int inp1_shape[NNLIB_MAX_DIM];
         int inp2_shape[NNLIB_MAX_DIM];
         int con_shape[NNLIB_MAX_DIM];
         
         for(int i = 0; i < NNLIB_MAX_DIM; i++)
         {
            con_shape[i] = 1;
            out_shape[i] = 1;
            inp1_shape[i] = 1;
            inp2_shape[i] = 1;
         }
         
         int off_o = NNLIB_MAX_DIM - out.dim();
         int off_a = NNLIB_MAX_DIM - a.dim();
         int off_b = NNLIB_MAX_DIM - b.dim();
         int off_c = NNLIB_MAX_DIM - cond.dim();

         for(int i = 0; i < out.dim(); i++){
            out_shape[i+off_o] = out.size(i);}

         for(int i = 0; i < a.dim(); i++)
            inp1_shape[i+off_a] = a.size(i);
         for(int i = 0; i < b.dim(); i++)
            inp2_shape[i+off_b] = b.size(i);
         for(int i = 0; i < cond.dim(); i++)
            con_shape[i+off_c] = cond.size(i);

        /* Add fallback if broadcast and condition dimension are larger than inputs dimension */

        if(con_shape[0] != out_shape[0] || con_shape[1] != out_shape[1] || con_shape[2] != out_shape[2] || con_shape[3] != out_shape[3])
        {
            void* p_scratch =  malloc(out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]);
            const unsigned char *p_brd_cond = (const unsigned char*)p_scratch;
            xa_nn_broadcast_8_8((WORD8* __restrict__) p_brd_cond, out_shape, (const  WORD8* __restrict__) con, con_shape, 4);

            for(int i = 0; i < 4; i++)
            {
                con_shape[i] = out_shape[i];
            }
            xa_nn_elm_where_broadcast_4D_f32xf32_f32(out_data, out_shape, a_data, inp1_shape,
                                                    b_data, inp2_shape, p_brd_cond, con_shape);
            free(p_scratch);
        }
        else
        {
            xa_nn_elm_where_broadcast_4D_f32xf32_f32(out_data, out_shape, a_data, inp1_shape, b_data, inp2_shape, con, con_shape);
        }
      }
      else
      {
          xa_nn_elm_where_f32xf32_f32(out_data, a_data, b_data, con, out.numel());
      }
}
else
{
  ET_SWITCH_REALHB_TYPES(a_type, ctx, name, CTYPE_A, [&]() {
    ET_SWITCH_REALHB_TYPES(b_type, ctx, name, CTYPE_B, [&]() {
      using CTYPE_OUT =
          typename torch::executor::promote_types<CTYPE_A, CTYPE_B>::type;
      apply_ternary_elementwise_fn<CTYPE_A, CTYPE_B, uint8_t, CTYPE_OUT>(
          [](const CTYPE_A val_a, const CTYPE_B val_b, const uint8_t val_c) {
            CTYPE_OUT a_casted = static_cast<CTYPE_OUT>(val_a);
            CTYPE_OUT b_casted = static_cast<CTYPE_OUT>(val_b);
            return val_c ? a_casted : b_casted;
          },
          a,
          b,
          cond,
          out);
    });
  });
}
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
