/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include "kernels.h"

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& mean_dim_out(
    RuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {

  ET_KERNEL_CHECK(
      ctx,
      check_mean_dim_args(in, dim_list, keepdim, dtype, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim_list, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);
      
  if(in.scalar_type() == ScalarType::Float)
  {

    FLOAT32 * __restrict__ p_out = out.mutable_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp = (const FLOAT32 * __restrict__)in.const_data_ptr<float>();
    
    WORD32 num_elm = in.numel();
      
    WORD32 num_inp_dims = in.dim();
    WORD32 num_out_dims = out.dim();
      
    WORD32 *p_inp_shape = (WORD32 *)malloc(num_inp_dims * sizeof(WORD32));
    
    WORD32 *p_out_shape = (WORD32 *)malloc(num_out_dims * sizeof(WORD32));
    
    for(int i = 0; i < num_inp_dims; i++)
    {
      p_inp_shape[i] = in.size(i);
    }
  
    for(int i = 0; i < num_out_dims; i++)
    {
      p_out_shape[i] = out.size(i);
    }
    
    WORD32 * __restrict__ p_axis = (WORD32 * __restrict__)malloc(dim_list.value().size() * sizeof(WORD32));
  
    WORD32 num_axis_dims = 0;
    for (const auto& d : dim_list.value())
    {
      if(d < 0)
      {
        p_axis[num_axis_dims] = num_inp_dims + d;
        num_axis_dims++;
      }
      else
      {
        p_axis[num_axis_dims] = d;
        num_axis_dims++;
      }
    }
    
    if(num_axis_dims == num_inp_dims)
    {
      num_out_dims = 1;
      p_out_shape[0] = 1;
    }
    
    WORD32 scratch_size = xa_nn_reduce_getsize_nhwc(-3,
                                  p_inp_shape,
                                  num_inp_dims,
                                  p_axis,
                                  num_axis_dims,
                                  1);
    
    void * __restrict__ p_scratch_in = (void * __restrict__)malloc(scratch_size * sizeof(WORD32));
  
    xa_nn_reduce_mean_4D_f32_f32(p_out,
                                p_out_shape,
                                p_inp,
                                p_inp_shape,
                                p_axis,
                                num_out_dims,
                                num_inp_dims,
                                num_axis_dims,
                                p_scratch_in);
  }
  else
  {
    ET_SWITCH_REALHB_TYPES(in.scalar_type(), ctx, "mean.out", CTYPE_IN, [&] {
      ET_SWITCH_FLOATH_TYPES(out.scalar_type(), ctx, "mean.out", CTYPE_OUT, [&] {
        CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
        const size_t num = get_reduced_dim_product(in, dim_list);
        
        for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
          CTYPE_OUT sum = 0;
          if (in.numel() > 0) {
            sum = map_reduce_over_dim_list<CTYPE_IN, CTYPE_OUT>(
                [](CTYPE_IN v) { return static_cast<CTYPE_OUT>(v); },
                [](CTYPE_OUT outv, CTYPE_OUT acc) { return acc + outv; },
                in,
                dim_list,
                out_ix);
          }
          out_data[out_ix] = sum / static_cast<float>(num);
        }
      });
    });
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
