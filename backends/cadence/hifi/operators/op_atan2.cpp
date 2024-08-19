/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>

#include"kernels.h"

#define NNLIB_OPT 0

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor&
atan2_out(RuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out) {
  // Determine output size and resize for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);
      
#if NNLIB_OPT
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool both_is_broadcasted = (a_is_broadcasted && b_is_broadcasted);

  WORD32 num_elm = out.numel();
  
  if(both_is_broadcasted)
  {
    WORD32* __restrict__ ptr1 = (WORD32* __restrict__ )malloc(num_elm * sizeof(WORD32));
    WORD32* __restrict__ ptr2 = (WORD32* __restrict__ )malloc(num_elm * sizeof(WORD32));
        
    WORD32* __restrict__ pin1 = (WORD32* __restrict__)a.const_data_ptr<float>();
    WORD32* __restrict__ pin2 = (WORD32* __restrict__)b.const_data_ptr<float>();
    
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

    WORD32 val = xa_nn_broadcast_32_32(ptr1,      
                            p_out_shape,             
                            pin1,        
                            p_inp1_shape,             
                            out.dim());
                          
    val = xa_nn_broadcast_32_32(ptr2,      
                            p_out_shape,             
                            pin2,        
                            p_inp2_shape,             
                            out.dim());
            
    FLOAT32 * __restrict__ p_out = (FLOAT32 * __restrict__ )out.mutable_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp1 = (const FLOAT32 * __restrict__)ptr1;
    const FLOAT32 * __restrict__ p_inp2 = (const FLOAT32 * __restrict__)ptr2;
    
    vecatan2f (p_out, 
              p_inp1, 
              p_inp2, 
              num_elm ); 
    
    free(ptr1);
    free(ptr2);
  }
  else if(a_is_broadcasted && (!b_is_broadcasted))
  {
    FLOAT32* __restrict__ ptr1 = (FLOAT32* __restrict__ )malloc((num_elm + 2) * sizeof(WORD32));
        
    FLOAT32* __restrict__ pin1 = (FLOAT32* __restrict__)a.const_data_ptr<float>();
    
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

    WORD32 val = xa_nn_broadcast_32_32((WORD32 *)ptr1,      
                            p_out_shape,             
                            (WORD32 *)pin1,        
                            p_inp1_shape,             
                            4);
            
    FLOAT32 * __restrict__ p_out = (FLOAT32 * __restrict__ )out.mutable_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp1 = (const FLOAT32 * __restrict__)ptr1;
    const FLOAT32 * __restrict__ p_inp2 = (const FLOAT32 * __restrict__)b.const_data_ptr<float>();
    
    vecatan2f (p_out, 
              p_inp1, 
              p_inp2, 
              num_elm); 
    
    free(ptr1);
  }
  else if(b_is_broadcasted && (!a_is_broadcasted))
  {
    WORD32* __restrict__ ptr1 = (WORD32* __restrict__ )malloc(num_elm * sizeof(WORD32));
        
    WORD32* __restrict__ pin1 = (WORD32* __restrict__)b.const_data_ptr<float>();
    
    WORD32 p_out_shape[4];
    WORD32 p_inp1_shape[4];
    
    for(int i = 0; i < 4; i++)
    {
      p_inp1_shape[i] = 1;
      p_out_shape[i] = 1;
    }
      
    int off_o = 4 - out.dim();        
    int off_a = 4 - b.dim();

    for(int i = 0; i < out.dim(); i++)
      p_out_shape[i+off_o] = out.size(i);
    for(int i = 0; i < a.dim(); i++)
      p_inp1_shape[i+off_a] = b.size(i);

    WORD32 val = xa_nn_broadcast_32_32(ptr1,      
                            p_out_shape,             
                            pin1,        
                            p_inp1_shape,             
                            out.dim());
            
    FLOAT32 * __restrict__ p_out = (FLOAT32 * __restrict__ )out.mutable_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp1 = (const FLOAT32 * __restrict__)a.const_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp2 = (const FLOAT32 * __restrict__)ptr1;
    
    vecatan2f (p_out, 
              p_inp1, 
              p_inp2, 
              num_elm ); 
    
    free(ptr1);
  }
  else
  {
    FLOAT32 * __restrict__ p_out = (FLOAT32 * __restrict__ )out.mutable_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp1 = (const FLOAT32 * __restrict__)a.const_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp2 = (const FLOAT32 * __restrict__)b.const_data_ptr<float>();
    
    vecatan2f (p_out, 
              p_inp1, 
              p_inp2, 
              num_elm );  
  }
#else
  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_SWITCH_REALHB_TYPES(a_type, ctx, "atan2.out", CTYPE_A, [&]() {
    ET_SWITCH_REALHB_TYPES(b_type, ctx, "atan2.out", CTYPE_B, [&]() {
      ET_SWITCH_FLOATH_TYPES(out_type, ctx, "atan2.out", CTYPE_OUT, [&]() {
        apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
            [](const CTYPE_A val_a, const CTYPE_B val_b) {
              CTYPE_OUT casted_a = static_cast<CTYPE_OUT>(val_a);
              CTYPE_OUT casted_b = static_cast<CTYPE_OUT>(val_b);
              return static_cast<CTYPE_OUT>(std::atan2(casted_a, casted_b));
            },
            a,
            b,
            out);
      });
    });
  });
  
#endif
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
