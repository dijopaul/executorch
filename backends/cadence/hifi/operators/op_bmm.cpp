/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kernels.h"

#include <executorch/kernels/portable/cpu/util/matmul_ops_util.cpp>
#include <executorch/kernels/portable/cpu/vec_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#define NNLIB_OPT 1

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& bmm_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const Tensor& mat2,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, check_bmm_args(in, mat2, out), InvalidArgument, out);
  
  
#if NNLIB_OPT
  
  if(in.scalar_type() == ScalarType::Float)
   {
      const float* in_data = in.const_data_ptr<float>();
      const float* mat2_data = mat2.const_data_ptr<float>();
      float* out_data = out.mutable_data_ptr<float>();
      
      int64_t batch_size = in.size(0);
      int64_t m = in.size(1);
      int64_t n = in.size(2);
      int64_t p = mat2.size(2);
      
      float *tmp = (float*) calloc((batch_size * m * p), sizeof(float));
      
      WORD32 rows = m;
      WORD32 cols1 = n;
      WORD32 row_stride1 = n;
      WORD32 vec_count = p;
      WORD32 vec_offset = n;
      WORD32 out_offset = 1;
      WORD32 out_stride = p;
      
      WORD8 *p_o = (WORD8 *) calloc((batch_size * m * p), sizeof(float));
      
      for (int i = 0; i < batch_size; ++i) {
          const FLOAT32 * __restrict__ p_mat1 = in_data + i * m * n;
          const FLOAT32 * __restrict__ p_vec1 = mat2_data + i * n * p;
          FLOAT32 * __restrict__ p_out = out_data + i * m * p;
          const FLOAT32 * __restrict__ p_bias = (const FLOAT32 * __restrict__)tmp;
          
          WORD8 *p_inp = (WORD8 *)p_vec1;
          
          
          WORD32 p_inp_shape[3];
          p_inp_shape[0] = n;
          p_inp_shape[1] = p;
          p_inp_shape[2] = 4;
          
          WORD32 p_out_shape[3];
          p_out_shape[0] = p;
          p_out_shape[1] = n;
          p_out_shape[2] = 4;
          
          WORD32 p_permute_vec[3] = {1, 0, 2};
          
          WORD32 num_out_dims = 3;
          WORD32 num_inp_dims = 3;
          
          WORD32 t = xa_nn_transpose_8_8(p_o
                            ,p_out_shape
                            ,p_inp
                            ,p_inp_shape
                            ,p_permute_vec
                            ,num_out_dims
                            ,num_inp_dims);
                            
          const FLOAT32 * __restrict__ p_vec = (const FLOAT32 * __restrict__)p_o;

          WORD32 val = xa_nn_matmul_f32xf32_f32(
                                          p_out,          
                                          p_mat1,   
                                          p_vec,   
                                          p_bias,   
                                          rows,
                                          cols1,
                                          row_stride1,                    
                                          vec_count,                      
                                          vec_offset,
                                          out_offset,
                                          out_stride);
       }
       
       free(tmp);
       free(p_o);
 }
  
#else
  size_t output_ndim = 0;
  exec_aten::SizesType output_sizes[kTensorDimensionLimit];
  get_bmm_out_target_size(in, mat2, output_sizes, &output_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ET_SWITCH_REAL_TYPES_AND(
      Half, in.scalar_type(), ctx, "bmm.out", CTYPE, [&]() {
        const CTYPE* in_data = in.const_data_ptr<CTYPE>();
        const CTYPE* mat2_data = mat2.const_data_ptr<CTYPE>();
        CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

        int64_t batch_size = in.size(0);
        int64_t m = in.size(1);
        int64_t n = in.size(2);
        int64_t p = mat2.size(2);

        for (int i = 0; i < batch_size; ++i) {
          const CTYPE* in_data_offset = in_data + i * m * n;
          const CTYPE* mat2_data_offset = mat2_data + i * n * p;
          CTYPE* out_data_offset = out_data + i * m * p;
          
          vec_matmul<CTYPE>(
              out_data_offset, in_data_offset, mat2_data_offset, m, n, p);
        }
      });
      
#endif
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
