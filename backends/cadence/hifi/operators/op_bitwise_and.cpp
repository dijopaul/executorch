/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// patternlint-disable-next-line executorch-cpp-nostdinc
#include <functional>

#include <executorch/kernels/portable/cpu/pattern/bitwise_op.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include "kernels.h"

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& bitwise_and_Tensor_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
    
  if(common_type == ScalarType::Bool)
  {
      
    const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
    const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
    const bool both_broadcasted = a_is_broadcasted && b_is_broadcasted;
    
    if(both_broadcasted)
    {
      WORD32 num_elm = out.numel();
      
      WORD8* __restrict__ ptr1 = (WORD8* __restrict__ )malloc(num_elm);
      WORD8* __restrict__ ptr2 = (WORD8* __restrict__ )malloc(num_elm);

      const   WORD8 * __restrict__ pin1 = (const   WORD8 * __restrict__)a.const_data_ptr<float>();
      const   WORD8 * __restrict__ pin2 = (const   WORD8 * __restrict__)b.const_data_ptr<float>();
      
      WORD8 * __restrict__ p_out = (WORD8 * __restrict__)out.mutable_data_ptr<float>();
      
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
  
      xa_nn_broadcast_8_8(  
                              ptr1,      
                              p_out_shape,             
                              pin1,        
                              p_inp1_shape,             
                              4);
                              
      xa_nn_broadcast_8_8(  
                              ptr2,      
                              p_out_shape,             
                              pin2,        
                              p_inp2_shape,             
                              4);
                              
      const   WORD8 * __restrict__ p_inp1 = (const   WORD8 * __restrict__)ptr1;
      const   WORD8 * __restrict__ p_inp2 = (const   WORD8 * __restrict__)ptr2;
      
      xa_nn_elm_logicaland_boolxbool_bool(
                                p_out,
                                p_inp1,
                                p_inp2,
                                num_elm);
    }
    else if(a_is_broadcasted && !b_is_broadcasted)
    {
      WORD32 num_elm = out.numel();
      
      WORD8* __restrict__ ptr1 = (WORD8* __restrict__ )malloc(num_elm);

      const   WORD8 * __restrict__ pin1 = (const   WORD8 * __restrict__)a.const_data_ptr<float>();
      const   WORD8 * __restrict__ p_inp2 = (const   WORD8 * __restrict__)b.const_data_ptr<float>();
      
      WORD8 * __restrict__ p_out = (WORD8 * __restrict__)out.mutable_data_ptr<float>();
      
      WORD32 p_out_shape[4];
      WORD32 p_inp1_shape[4];
      
      for(int i = 0; i < 4; i++)
      {
        p_inp1_shape[i] = 1;
        p_out_shape[i] = 1;
      }
        
      int off_o = 4 - out.dim();        
      int off_a = 4 - a.dim();
  
      for(int i = 0; i < out.dim(); i++)
        p_out_shape[i+off_o] = out.size(i);
      for(int i = 0; i < a.dim(); i++)
        p_inp1_shape[i+off_a] = a.size(i);
  
      xa_nn_broadcast_8_8(ptr1,      
                              p_out_shape,             
                              pin1,        
                              p_inp1_shape,             
                              4);
                              
      const   WORD8 * __restrict__ p_inp1 = (const   WORD8 * __restrict__)ptr1;
      
      xa_nn_elm_logicaland_boolxbool_bool(
                                p_out,
                                p_inp1,
                                p_inp2,
                                num_elm);
    }
    else if(!a_is_broadcasted && b_is_broadcasted)
    {
      WORD32 num_elm = out.numel();
      
      WORD8* __restrict__ ptr1 = (WORD8* __restrict__ )malloc(num_elm);

      const   WORD8 * __restrict__ p_inp1 = (const   WORD8 * __restrict__)a.const_data_ptr<float>();
      const   WORD8 * __restrict__ pinp2 = (const   WORD8 * __restrict__)b.const_data_ptr<float>();
      
      WORD8 * __restrict__ p_out = (WORD8 * __restrict__)out.mutable_data_ptr<float>();
      
      WORD32 p_out_shape[4];
      WORD32 p_inp2_shape[4];
      
      for(int i = 0; i < 4; i++)
      {
        p_inp2_shape[i] = 1;
        p_out_shape[i] = 1;
      }
        
      int off_o = 4 - out.dim();        
      int off_b = 4 - b.dim();
  
      for(int i = 0; i < out.dim(); i++)
        p_out_shape[i+off_o] = out.size(i);
      for(int i = 0; i < b.dim(); i++)
        p_inp2_shape[i+off_b] = b.size(i);
  
      xa_nn_broadcast_8_8(
                              ptr1,      
                              p_out_shape,             
                              pinp2,        
                              p_inp2_shape,             
                              4);
                              
      const   WORD8 * __restrict__ p_inp2 = (const   WORD8 * __restrict__)ptr1;
      
      xa_nn_elm_logicaland_boolxbool_bool(
                                p_out,
                                p_inp1,
                                p_inp2,
                                num_elm);
    }
    else
    {
      const   WORD8 * __restrict__ p_inp1 = (const   WORD8 * __restrict__)a.const_data_ptr<float>();
      const   WORD8 * __restrict__ p_inp2 = (const   WORD8 * __restrict__)b.const_data_ptr<float>();
      
      WORD8 * __restrict__ p_out = (WORD8 * __restrict__)out.mutable_data_ptr<float>();
      
      WORD32 num_elm = out.numel();
      
      xa_nn_elm_logicaland_boolxbool_bool(
                                p_out,
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
  
    ET_SWITCH_INT_TYPES_AND(
        Bool, a_type, ctx, "bitwise_and.Tensor_out", CTYPE_A, [&]() {
          ET_SWITCH_INT_TYPES_AND(
              Bool, b_type, ctx, "bitwise_and.Tensor_out", CTYPE_B, [&]() {
                using CTYPE_IN = typename torch::executor::
                    promote_types<CTYPE_A, CTYPE_B>::type;
                ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
                ET_SWITCH_REAL_TYPES_AND(
                    Bool,
                    out_type,
                    ctx,
                    "bitwise_and.Tensor_out",
                    CTYPE_OUT,
                    [&]() {
                      internal::BitwiseOpInner<
                          can_cast<CTYPE_IN, CTYPE_OUT>::value,
                          std::bit_and,
                          CTYPE_A,
                          CTYPE_B,
                          CTYPE_IN,
                          CTYPE_OUT>::run(a, b, out);
                    });
              });
        });
  }

  return out;
}

Tensor& bitwise_and_Scalar_out(
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

  ET_SWITCH_INT_TYPES_AND(
      Bool, a_type, ctx, "bitwise_and.Scalar_out", CTYPE_A, [&]() {
        ET_SWITCH_SCALAR_OBJ_INTB_TYPES(
            b_type, ctx, "bitwise_and.Scalar_out", CTYPE_B, [&]() {
              CTYPE_B val_b = 0;
              utils::extract_scalar(b, &val_b);
              ET_SWITCH_INT_TYPES_AND(
                  Bool,
                  common_type,
                  ctx,
                  "bitwise_and.Scalar_out",
                  CTYPE_IN,
                  [&]() {
                    ET_SWITCH_REAL_TYPES_AND(
                        Bool,
                        out_type,
                        ctx,
                        "bitwise_and.Scalar_out",
                        CTYPE_OUT,
                        [&]() {
                          apply_unary_map_fn(
                              [val_b](const CTYPE_A val_a) {
                                CTYPE_IN a_casted =
                                    static_cast<CTYPE_IN>(val_a);
                                CTYPE_IN b_casted =
                                    static_cast<CTYPE_IN>(val_b);
                                CTYPE_IN value = std::bit_and<CTYPE_IN>()(
                                    a_casted, b_casted);

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
