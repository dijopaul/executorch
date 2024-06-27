/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "inttypes.h"
#include "stddef.h"
#include "xa_type_def.h"

                                     
extern "C" WORD32 xa_nn_elm_quantize_f32_asym8s(WORD8 * __restrict__ p_out,
                                     const FLOAT32 * __restrict__ p_inp,
                                     FLOAT32 out_scale,
                                     WORD32  out_zero_bias,
                                     WORD32  num_elm);
    
extern "C" WORD32 xa_nn_elm_quantize_f32_asym8u(UWORD8 * __restrict__ p_out,
                                     const FLOAT32 * __restrict__ p_inp,
                                     FLOAT32 out_scale,
                                     WORD32  out_zero_bias,
                                     WORD32  num_elm); 
    
extern "C" WORD32 xa_nn_elm_dequantize_asym8s_f32(FLOAT32 * __restrict__ p_out,
                                       const WORD8 * __restrict__ p_inp,
                                       WORD32  inp_zero_bias,
                                       FLOAT32 inp_scale,
                                       WORD32  num_elm);
    
extern "C" WORD32 xa_nn_elm_dequantize_asym8u_f32(FLOAT32 * __restrict__ p_out,
                                       const UWORD8 * __restrict__ p_inp,
                                       WORD32   inp_zero_bias,
                                       FLOAT32  inp_scale,
                                       WORD32   num_elm);

extern "C" WORD32 xa_nn_elm_add_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp1,
                               const FLOAT32 * __restrict__ p_inp2,
                               WORD32 num_elm);

extern "C"  WORD32 xa_nn_elm_add_broadcast_4D_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                                const WORD32 *const p_out_shape,
                                const FLOAT32 * __restrict__ p_inp1,
                                const WORD32 *const p_inp1_shape,
                                const FLOAT32 * __restrict__ p_inp2,
                                const WORD32 *const p_inp2_shape);                               

extern "C" WORD32 xa_nn_elm_sub_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                                const FLOAT32 * __restrict__ p_inp1,
                                const FLOAT32 * __restrict__ p_inp2,
                                WORD32 num_elm);
                                
extern "C"  WORD32 xa_nn_elm_sub_broadcast_4D_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                                const WORD32 *const p_out_shape,
                                const FLOAT32 * __restrict__ p_inp1,
                                const WORD32 *const p_inp1_shape,
                                const FLOAT32 * __restrict__ p_inp2,
                                const WORD32 *const p_inp2_shape);
                               
extern "C" WORD32 xa_nn_elm_mul_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp1,
                               const FLOAT32 * __restrict__ p_inp2,
                               WORD32 num_elm);

extern "C" WORD32 xa_nn_elm_mul_broadcast_4D_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                      const FLOAT32 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const FLOAT32 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape);
                               
extern "C" WORD32 xa_nn_elm_div_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                                const FLOAT32 * __restrict__ p_inp1,
                                const FLOAT32 * __restrict__ p_inp2,
                                WORD32 num_elm);

extern "C" WORD32 xa_nn_elm_div_mode_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                                   const FLOAT32 * __restrict__ p_inp1,
                                   const FLOAT32 * __restrict__ p_inp2,
                                   WORD32 mode,
                                   WORD32 num_elm);                                

extern "C" WORD32 xa_nn_vec_softmax_f32_f32( FLOAT32 * __restrict__ p_out, 
                                const FLOAT32 * __restrict__ p_vec, WORD32 vec_length);

extern "C" WORD32 xa_nn_vec_tanh_f32_f32(FLOAT32 * __restrict__ p_out, 
                                const FLOAT32 * __restrict__ p_vec, WORD32 vec_length);

extern "C" WORD32 xa_nn_vec_sigmoid_f32_f32(FLOAT32  * __restrict__ p_out,
                                      const FLOAT32 * __restrict__ p_vec, WORD32 vec_length);

extern "C" void xa_nn_layer_norm_asym8_asym8(const UWORD8 *in_data, float input_scale, int64_t input_zero_point, WORD32 last_dim,
                                WORD32 leading_dims, const float *weight_data, const float *bias_data,double eps,
                                double output_scale, int64_t output_zero_point, UWORD8 *out_data);
                                
                                
extern "C" WORD32 xa_nn_vec_relu_8u_8u_custom(UWORD8 * __restrict__ p_out, const UWORD8  * __restrict__ p_vec,
                                               UWORD8 q_zero_point, WORD32 vec_length);

extern "C" WORD32 xa_nn_vec_relu_8_8_custom(WORD8 * __restrict__ p_out, const WORD8  * __restrict__ p_vec,
                                             WORD8 q_zero_point, WORD32 vec_length);
                                             
extern "C" WORD32 xa_nn_transpose_8_8(WORD8 * __restrict__ p_out, const WORD32 *const p_out_shape
                    ,const WORD8 * __restrict__ p_inp, const WORD32 *const p_inp_shape
                    ,const WORD32 * __restrict__ p_permute_vec, WORD32 num_out_dims, WORD32 num_inp_dims);

extern "C" WORD32 xa_nn_transpose_32_32(WORD32 * __restrict__ p_out
                    ,const WORD32 *const p_out_shape
                    ,const WORD32 * __restrict__ p_inp
                    ,const WORD32 *const p_inp_shape
                    ,const WORD32 * __restrict__ p_permute_vec
                    ,WORD32 num_out_dims
                    ,WORD32 num_inp_dims);

extern "C" WORD32 xa_nn_matmul_f32xf32_f32(FLOAT32  * __restrict__ p_out, 
                    const FLOAT32  * __restrict__ p_mat1,
                    const FLOAT32  * __restrict__ p_mat2,
                    const FLOAT32  * __restrict__ p_bias,
                    WORD32 rows,
                    WORD32 cols,
                    WORD32 row_stride,
                    WORD32 vec_count,
                    WORD32 vec_offset,
                    WORD32 out_offset,
                    WORD32 out_stride);
namespace impl {
namespace HiFi {
namespace kernels {

void memcpy(void* dst, const void* src, size_t num_bytes);

WORD32 matmul_asym8uxasym8u_asym8u(
    UWORD8* __restrict__ p_out, // output uint8 matrix
    const UWORD8* __restrict__ p_mat1, // weight uint8 matrix
    const UWORD8* __restrict__ p_vec1, // input uint8 matrix
    const WORD32* __restrict__ p_bias, // bias int32 vec
    WORD32 rows, // rows of p_mat1
    WORD32 cols1, // columns of p_mat1
    WORD32 row_stride1, // row stride of p_mat1
    WORD32 vec_count, // rows of p_mat2
    WORD32 vec_offset, // vec_offset of p_mat2.
    WORD32 out_offset, // out_offset, i.e., offset of next output element
    WORD32 out_stride, // out_stride, i.e., stride to go to next output row
    WORD32 mat1_zero_bias, // zero_point of p_mat1
    WORD32 vec1_zero_bias, // zero_point of p_vec1
    const WORD32* __restrict__ out_multiplier,
    const WORD32* __restrict__ out_shift,
    WORD32 out_zero_bias,
    bool per_channel_quantized = false); // per-channel quantized weight

template <typename T>
T quantize(const float x, float scale, int32_t zero_point);

template <typename T>
float dequantize(const T x, float scale, int32_t zero_point);

template <typename T>
void quantize(
    T* __restrict__ y,
    const float* __restrict__ x,
    float scale,
    int32_t zero_point,
    size_t size);

// Deuantize an int8_t/uint8_t/int16_t array to an fp32 array
template <typename T>
void dequantize(
    float* __restrict__ y,
    const T* __restrict__ x,
    float scale,
    int32_t zero_point,
    size_t size);

}; // namespace kernels
}; // namespace HiFi
}; // namespace impl
