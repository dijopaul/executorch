/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>

#include <cstring>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.cpp>

#include "kernels.h"
#include "stdio.h"

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& cat_out(
    RuntimeContext& ctx,
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  
  if(out.scalar_type()== ScalarType::Float){
    WORD32 num_inp = tensors.size();
    WORD32 num_inp_dims = tensors[0].dim();
    WORD32 num_out_dims = num_inp_dims;
    WORD32 axis = dim;
    
    WORD32 inp_shape[16][16];
    WORD32 p_out_shape[16] = {0};
    
    WORD32 *ptr_shape[16];
    const WORD32 *ptr[16];
    
    for(int i = 0; i < num_inp; i++)
    {
      ptr[i] = (const WORD32 *)tensors[i].const_data_ptr<float>();
      for(int j = 0; j < num_inp_dims; j++)
      {
        inp_shape[i][j] = tensors[i].size(j);
        if(j == axis)
          p_out_shape[j] += inp_shape[i][j];
        else
          p_out_shape[j] = inp_shape[i][j];
      }
      
      ptr_shape[i] = inp_shape[i];
    }
    
    const WORD32 **pp_inps = &ptr[0];
    
    WORD32 * p_out = (WORD32 *)out.mutable_data_ptr<float>();
    
    const WORD32 *const *pp_inps_shape = (const WORD32 *const *)&ptr_shape[0];
    
    WORD32 val = xa_nn_concat_32_32(p_out
                                  ,p_out_shape
                                  ,pp_inps
                                  ,pp_inps_shape
                                  ,num_out_dims
                                  ,num_inp
                                  ,num_inp_dims
                                  ,axis);   
    
    return out;
  }
  else if(out.scalar_type() == ScalarType::Char){
    WORD32 num_inp = tensors.size();
    WORD32 num_inp_dims = tensors[0].dim();
    WORD32 num_out_dims = num_inp_dims;
    WORD32 axis = dim;
    
    WORD32 inp_shape[16][16];
    WORD32 p_out_shape[16] = {0};
    
    WORD32 *ptr_shape[16];
    const WORD8 *ptr[16];
    
    for(int i = 0; i < num_inp; i++)
    {
      ptr[i] = (const WORD8 *)tensors[i].const_data_ptr<char>();
      for(int j = 0; j < num_inp_dims; j++)
      {
        inp_shape[i][j] = tensors[i].size(j);
        if(j == axis)
          p_out_shape[j] += inp_shape[i][j];
        else
          p_out_shape[j] = inp_shape[i][j];
      }
      
      ptr_shape[i] = inp_shape[i];
    }
    
    const WORD8 **pp_inps = &ptr[0];
    
    WORD8 * p_out = (WORD8 *)out.mutable_data_ptr<char>();
    
    const WORD32 *const *pp_inps_shape = (const WORD32 *const *)&ptr_shape[0];
    
    WORD32 val = xa_nn_concat_8_8(p_out
                                  ,p_out_shape
                                  ,pp_inps
                                  ,pp_inps_shape
                                  ,num_out_dims
                                  ,num_inp
                                  ,num_inp_dims
                                  ,axis);   
    
    return out;    
  }
  else {
  
    if (dim < 0) {
      dim += out.dim();
    }
  
    ET_KERNEL_CHECK(ctx, check_cat_args(tensors, dim, out), InvalidArgument, out);
  
    Tensor::SizesType expected_out_size[kTensorDimensionLimit];
    size_t expected_out_dim = 0;
    get_cat_out_target_size(tensors, dim, expected_out_size, &expected_out_dim);
    ET_CHECK(
        resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok);
  
    // Special handling when all inputs are 1D-empty tensors for aten consistency
    // In that case, just return an 1D-empty tensor without checking dim
    bool all_1d_empty = true;
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (tensors[i].numel() != 0 || tensors[i].dim() != 1) {
        all_1d_empty = false;
        break;
      }
    }
    if (all_1d_empty) {
      return out;
    }
  
    const size_t outer = getLeadingDims(out, dim);
    const size_t dim_stride = getTrailingDims(out, dim);
    const size_t ninputs = tensors.size();
    
    const auto out_type = out.scalar_type();
  
    ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, "cat", CTYPE_OUT, [&] {
      CTYPE_OUT* out_ptr = out.mutable_data_ptr<CTYPE_OUT>();
      for (size_t i = 0; i < outer; ++i) {
        for (size_t j = 0; j < ninputs; ++j) {
          const auto in_type = tensors[j].scalar_type();
          ET_SWITCH_REAL_TYPES_AND(Bool, in_type, ctx, "cat", CTYPE_IN, [&] {
            if (tensors[j].numel() == 0) {
              return;
            }
            size_t inner = tensors[j].size(dim) * dim_stride;
            
            const CTYPE_IN* const in_ptr =
                tensors[j].const_data_ptr<CTYPE_IN>() + i * inner;
  
            for (size_t k = 0; k < inner; ++k) {
              out_ptr[k] = static_cast<CTYPE_OUT>(in_ptr[k]);
            }
            out_ptr += inner;
          });
        }
      }
    });
  }  

  return out;
#endif
}

} // namespace native
} // namespace executor
} // namespace torch
