/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include "kernels.h"

namespace torch {
namespace executor {
namespace native {

using SizesType = exec_aten::SizesType;
using Tensor = exec_aten::Tensor;
using IntArrayRef = exec_aten::ArrayRef<int64_t>;

namespace {

void increment_coordinate_permuted(
    const Tensor& tensor,
    size_t* const coordinate,
    IntArrayRef dims) {
  for (int i = dims.size() - 1; i >= 0; i--) {
    size_t d = dims[i] >= 0 ? dims[i] : dims[i] + tensor.dim();
    coordinate[d]++;
    if (coordinate[d] == tensor.size(d)) {
      coordinate[d] = 0;
    } else {
      return;
    }
  }
}

} // namespace

Tensor& permute_copy_out(
    RuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef dims,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_permute_copy_args(in, dims, out), InvalidArgument, out);

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_permute_copy_out_target_size(
      in, dims, expected_out_size, &expected_out_dim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  const auto in_type = out.scalar_type();

  if(in_type == ScalarType::Float)
  {
    WORD32 * p_inp = (WORD32 *)in.const_data_ptr<float>();
    WORD32 * p_out = (WORD32 *)out.mutable_data_ptr<float>();
    
    WORD32 num_inp_dims = in.dim();
    WORD32 num_out_dims = num_inp_dims;
    
    WORD32 p_inp_shape[5];
    WORD32 p_out_shape[5];
    WORD32 p_permute_vec[5];
    
    for(int i = 0; i < num_inp_dims; i++)
    {
      p_inp_shape[i] = in.size(i);
      p_out_shape[i] = in.size(dims[i]);
      p_permute_vec[i] = dims[i];
    }
    
    WORD32 val = xa_nn_transpose_32_32(p_out
                                      ,p_out_shape
                                      ,p_inp
                                      ,p_inp_shape
                                      ,p_permute_vec
                                      ,num_out_dims
                                      ,num_inp_dims);
                                      
  }
  else if(in_type == ScalarType::Char)
  {
    WORD8 * p_inp = (WORD8 *)in.const_data_ptr<char>();
    WORD8 * p_out = (WORD8 *)out.mutable_data_ptr<char>();
    
    WORD32 num_inp_dims = in.dim();
    WORD32 num_out_dims = num_inp_dims;
    
    WORD32 p_inp_shape[5];
    WORD32 p_out_shape[5];
    WORD32 p_permute_vec[5];
    
    for(int i = 0; i < num_inp_dims; i++)
    {
      p_inp_shape[i] = in.size(i);
      p_out_shape[i] = in.size(dims[i]);
      p_permute_vec[i] = dims[i];
    }
    
    p_inp_shape[num_inp_dims] = 4;
    p_out_shape[num_inp_dims] = 4;
    
    
    WORD32 val = xa_nn_transpose_8_8(p_out
                                      ,p_out_shape
                                      ,p_inp
                                      ,p_inp_shape
                                      ,p_permute_vec
                                      ,num_out_dims
                                      ,num_inp_dims);
      
  }
  else
  {
        // in and out must be the same dtype
    ET_SWITCH_ALL_TYPES(in_type, ctx, "permute_copy.out", CTYPE, [&] {
      const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
      CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

      size_t in_coord[kTensorDimensionLimit] = {0};

      for (size_t i = 0; i < out.numel(); ++i) {
        out_data[i] = in_data[coordinateToIndex(in, in_coord)];
        increment_coordinate_permuted(in, in_coord, dims);
      }
    });
      
  }
  
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
