/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kernels.h"

#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>
#include <cmath>


#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))
#define NNLIB_OPT 0

namespace impl {
namespace HiFi {
namespace native {

using Tensor = exec_aten::Tensor;
using RuntimeContext = torch::executor::RuntimeContext;
using ScalarType = exec_aten::ScalarType;


// This implements a generic 2d conv kernel that operates on raw pointers.
// The version handles both quantized and fp32 convolutions.
// The input is of shape [n x c x h x w]
// The weight is of shape [oc x wc x wh x ww], where wc == c
// The output is of shape [n x oc x oh x ow]
// The bias is of shape [oc]
template <typename IT, typename WT, typename BT, typename OT, bool quantized>
__attribute__((noinline)) void conv2d_nchw_core_generic(
    // All the arrays
    const IT* __restrict__ p_in,
    const WT* __restrict__ p_weight,
    const BT* __restrict__ p_bias,
    OT* __restrict__ p_out,
    // The array sizes
    int32_t n,
    int32_t c,
    int32_t h,
    int32_t w,
    int32_t oc,
    int32_t wc,
    int32_t wh,
    int32_t ww,
    int32_t oh,
    int32_t ow,
    // Stride
    int16_t s0,
    int16_t s1,
    // Padding
    int16_t p0,
    int16_t p1,
    // Dilation
    int16_t d0,
    int16_t d1,
    // Group for depthwise conv
    int16_t groups,
    // Optional args that are only relevant for quantized convolution
    // input zero point
    IT in_zero_point = 0,
    // weight zero point
    const int32_t* __restrict__ weight_zero_point = nullptr,
    const float* __restrict__ bias_scale = nullptr,
    float out_scale = 1,
    OT out_zero_point = 0,
    bool per_tensor_quantized = true) {
  float inv_out_scale = 1. / out_scale;
  bool zero_pad_unit_dilation = d0 == 1 && d1 == 1 && p0 == 0 && p1 == 0;

  // Compute the number of in and out channels per group
  const int ocpg = oc / groups;
  const int icpg = c / groups;

  // Iterate over all the output batches (i.e., n)
  for (int _n = 0; _n < n; ++_n) {
    const IT* in_batch = p_in + _n * c * h * w;
    OT* out_batch = p_out + _n * oc * oh * ow;
    // Compute separable convolution for each group
    for (int _g = 0; _g < groups; ++_g) {
      // Identify the input and output channels involved in the computation
      // of this group
      int sic = _g * icpg;
      int soc = _g * ocpg;
      // Populate all the output channels in the group
      for (int _oc = soc; _oc < soc + ocpg; ++_oc) {
        OT* out_plane = out_batch + _oc * oh * ow;
        const WT* weight_batch = p_weight + _oc * wc * wh * ww;
        // We compute one output channel at a time. The computation can be
        // thought of as a stencil computation: we iterate over an input of size
        // icpg x h x w, with a stencil of size icpg x wh x ww, to compute an
        // output channel of size 1 x oh x ow.
        for (int _h = 0, _oh = 0; _oh < oh; _h += s0, ++_oh) {
          for (int _w = 0, _ow = 0; _ow < ow; _w += s1, ++_ow) {
            float acc = p_bias[_oc];
            // Below is the stencil computation that performs the hadamard
            // product+accumulation of each input channel (contributing to the
            // output channel being computed) with the corresponding weight
            // channel.
            // If the padding is 0, and dilation is 1, then we can remove the
            // unnecessary checks, and simplify the code so that it can be
            // vectorized by Tensilica compiler.
            if (zero_pad_unit_dilation) {
              for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                const IT* in_plane = in_batch + _ic * h * w;
                const WT* weight_plane = weight_batch + (_ic - sic) * wh * ww;
                for (int _wh = 0; _wh < wh; ++_wh) {
                  for (int _ww = 0; _ww < ww; ++_ww) {
                    int ioff = (_h + _wh) * w + (_w + _ww);
                    int woff = _wh * ww + _ww;
                    float lhs = in_plane[ioff] - in_zero_point;
                    float rhs = weight_plane[woff] -

                      (quantized ? 0 : 0);
                    /*float rhs = weight_plane[woff] -
                    (quantized ? weight_zero_point[0] : 0);*/

                    acc += lhs * rhs;
                  }
                }
              }
            } else {
              for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                const IT* in_plane = in_batch + _ic * h * w;
                const WT* weight_plane = weight_batch + (_ic - sic) * wh * ww;
                for (int _wh = 0; _wh < wh; ++_wh) {
                  for (int _ww = 0; _ww < ww; ++_ww) {
                    if (((_h + d0 * _wh - p0) >= 0) &&
                        ((_h + d0 * _wh - p0) < h) &&
                        ((_w + d1 * _ww - p1) >= 0) &&
                        ((_w + d1 * _ww - p1) < w)) {
                        //((_w + d1 * _ww - p1 < w))) {

                      int ioff =
                          (_h + d0 * _wh - p0) * w + (_w + d1 * _ww - p1);
                      int woff = _wh * ww + _ww;
                      float lhs = in_plane[ioff] - in_zero_point;
                      float rhs = weight_plane[woff] -

                      (quantized ? 0 : 0);
                      /*float rhs = weight_plane[woff] -
                      (quantized ? weight_zero_point[0] : 0);*/

                      acc += lhs * rhs;
                    }
                  }
                }
              }
            }
            if (quantized) {
              float val =
                  (per_tensor_quantized ? bias_scale[0] : bias_scale[_oc]) *
                  acc;
              out_plane[_oh * ow + _ow] =
                  kernels::quantize<OT>(val, inv_out_scale, out_zero_point);
            } else {
              out_plane[_oh * ow + _ow] = acc;
            }
          }
        }
      }
    }
  }
}

// The quantized convolution kernel. in_scale and weight_scale are implicit in
// bias_scale, since it is a product of the two. The kernel will branch to
// quantized::conv1d or quantized::conv2d based on the dimensionality of
// activation tensor.
void quantized_conv_out(
    RuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    exec_aten::IntArrayRef stride,
    exec_aten::IntArrayRef padding,
    exec_aten::IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& bias_scale,
    double output_scale,
    int64_t output_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    bool channel_last,
    Tensor& out) {
  bool conv1d = input.dim() == 3;
  
#if NNLIB_OPT
    
  if(input.scalar_type() == ScalarType::Char)
  {
    WORD8* __restrict__ p_out = (WORD8* __restrict__ )out.mutable_data_ptr<int8_t>();
    WORD8* __restrict__ p_inp = (WORD8* __restrict__ )input.const_data_ptr<int8_t>();
    WORD8* __restrict__ p_kernel = (WORD8* __restrict__ )weight.const_data_ptr<int8_t>();
    WORD32* __restrict__ p_bias = (WORD32* __restrict__ )bias.const_data_ptr<int32_t>();
    
    WORD32 input_height = conv1d ? 1 : input.size(2);
    WORD32 input_width = conv1d ? input.size(2) : input.size(3);
    WORD32 input_channels = input.size(1);
    WORD32 kernel_height = conv1d ? 1 : weight.size(2);
    WORD32 kernel_width = conv1d ? weight.size(2) : weight.size(3);
    WORD32 kernel_channels = weight.size(1);
    WORD32 out_channels = weight.size(0);
    WORD32 out_height = conv1d ? 1 : out.size(2);
    WORD32 out_width = conv1d ? out.size(2) : out.size(3);
    WORD32 batches = input.size(0);
    
    WORD32 x_stride = stride[1];
    WORD32 y_stride = stride[0];
    WORD32 x_padding = padding[1];
    WORD32 y_padding = padding[0];
    WORD32 dilation_width = 1;
    WORD32 dilation_height = 1;
    
    WORD32 * kernel_bias_ptr = (WORD32 *)weight_zero_point.const_data_ptr<int32_t>();
    
    WORD32 input_zero_bias = -in_zero_point;
    WORD32 kernel_zero_bias = -kernel_bias_ptr[0];

    WORD32 out_multiplier32[out_channels];
    WORD32 out_shift32[out_channels];
    
    float out_scale = 1. / output_scale;

    for(int i = 0; i < out_channels; i++)
    {
        out_multiplier32[i] = bias_scale.const_data_ptr<float>()[0] * out_scale * 2147483648;
        out_shift32[i] = 0;
    }

    WORD32 out_zero_bias = output_zero_point;
    WORD32 inp_precision = 8;
    WORD32 kernel_precision = 8;
    pVOID p_scratch = nullptr;
    WORD8 *ptr_scratch;
    
    WORD32 scratch_size = 0;
    
    WORD32 out_data_format = 1;
    WORD32 inp_data_format = 0;
    
    WORD8 *ptr1 = (WORD8 *)malloc(((input.size(0) * input_channels * input_height * input_width) + 8) * sizeof(WORD8));
    WORD8 *ptr2 = (WORD8 *)malloc(((out_channels * kernel_channels * kernel_height * kernel_width) + 8) * sizeof(WORD8));
    
    WORD8 *pin = (WORD8 *)ALIGN_PTR(ptr1, 8);
    WORD8 *pkernel = (WORD8 *)ALIGN_PTR(ptr2, 8);
    
    WORD32 p_inp_shape[4];
    p_inp_shape[0] = input.size(0);
    p_inp_shape[1] = input_channels;
    p_inp_shape[2] = input_height;
    p_inp_shape[3] = input_width;
    
    WORD32 p_out_shape[4];
    p_out_shape[0] = input.size(0);
    p_out_shape[1] = input_height;
    p_out_shape[2] = input_width;
    p_out_shape[3] = input_channels;
    
    WORD32 p_permute_vec[4] = {0, 2, 3, 1};
    
    WORD32 num_out_dims = 4;
    WORD32 num_inp_dims = 4;
    
    WORD32 t = xa_nn_transpose_8_8(pin
                      ,p_out_shape
                      ,p_inp
                      ,p_inp_shape
                      ,p_permute_vec
                      ,num_out_dims
                      ,num_inp_dims);
                      
    WORD32 p_inp_shape1[4];
    p_inp_shape1[0] = out_channels;
    p_inp_shape1[1] = kernel_channels;
    p_inp_shape1[2] = kernel_height;
    p_inp_shape1[3] = kernel_width;
    
    WORD32 p_out_shape1[4];
    p_out_shape1[0] = out_channels;
    p_out_shape1[1] = kernel_height;
    p_out_shape1[2] = kernel_width;
    p_out_shape1[3] = kernel_channels;
    
    WORD32 p_permute_vec1[4] = {0, 2, 3, 1};
    
    WORD32 num_out_dims1 = 4;
    WORD32 num_inp_dims1 = 4;
    
    WORD32 t1 = xa_nn_transpose_8_8(pkernel
                      ,p_out_shape1
                      ,p_kernel
                      ,p_inp_shape1
                      ,p_permute_vec1
                      ,num_out_dims1
                      ,num_inp_dims1);   
    
    scratch_size = xa_nn_conv2d_getsize(
      input_height,
      input_width,
      input_channels,
      kernel_height,
      kernel_width,
      kernel_channels,
      dilation_height,
      dilation_width,
      y_stride,
      y_padding,
      x_stride,
      x_padding,
      out_height,
      out_width,
      out_channels,
      inp_precision,
      kernel_precision,
      out_data_format);
                                            
    scratch_size=scratch_size<0?0:scratch_size;

    ptr_scratch = (WORD8 *)malloc(scratch_size + 16);
    
    p_scratch = (xa_codec_handle_t)ALIGN_PTR(ptr_scratch, 8);
    
    for (int _n = 0; _n < batches; ++_n) {
      WORD8 *in_batch = pin + _n * input_channels * input_height * input_width;
      WORD8 *out_batch = p_out + _n * out_channels * out_height * out_width;
    
      WORD32 val = xa_nn_conv2d_per_chan_sym8sxasym8s
        (out_batch
        ,in_batch
        ,pkernel
        ,p_bias
        ,input_height
        ,input_width
        ,input_channels
        ,kernel_height
        ,kernel_width
        ,kernel_channels
        ,dilation_height
        ,dilation_width
        ,out_channels
        ,x_stride
        ,y_stride
        ,x_padding
        ,y_padding
        ,out_height
        ,out_width
        ,input_zero_bias
        ,out_multiplier32
        ,out_shift32
        ,out_zero_bias
        ,out_data_format
        ,p_scratch
      );
    }
    
    free(ptr1);
    free(ptr2);
    free(ptr_scratch);
  }
  else if(input.scalar_type() == ScalarType::Byte)
  {
    printf("UINT8 CONV KERNEL");
    UWORD8* __restrict__ p_out = (UWORD8* __restrict__ )out.mutable_data_ptr<uint8_t>();
    UWORD8* __restrict__ p_inp = (UWORD8* __restrict__ )input.const_data_ptr<uint8_t>();
    UWORD8* __restrict__ p_kernel = (UWORD8* __restrict__ )weight.const_data_ptr<uint8_t>();
    WORD32* __restrict__ p_bias = (WORD32* __restrict__ )bias.const_data_ptr<int32_t>();
    
    WORD32 input_height = conv1d ? 1 : input.size(2);
    WORD32 input_width = conv1d ? input.size(2) : input.size(3);
    WORD32 input_channels = input.size(1);
    WORD32 kernel_height = conv1d ? 1 : weight.size(2);
    WORD32 kernel_width = conv1d ? weight.size(2) : weight.size(3);
    WORD32 kernel_channels = weight.size(1);
    WORD32 out_channels = weight.size(0);
    WORD32 out_height = conv1d ? 1 : out.size(2);
    WORD32 out_width = conv1d ? out.size(2) : out.size(3);
    WORD32 batches = input.size(0);
    
    WORD32 x_stride = stride[1];
    WORD32 y_stride = stride[0];
    WORD32 x_padding = padding[1];
    WORD32 y_padding = padding[0];
    WORD32 dilation_width = 1;
    WORD32 dilation_height = 1;
    
    WORD32 * kernel_bias_ptr = (WORD32 *)weight_zero_point.const_data_ptr<int32_t>();
    
    WORD32 input_zero_bias = -in_zero_point;
    WORD32 kernel_zero_bias = -kernel_bias_ptr[0];

    WORD32 out_multiplier32[out_channels];
    WORD32 out_shift32[out_channels];
    
    float out_scale = 1. / output_scale;

    for(int i = 0; i < out_channels; i++)
    {
        out_multiplier32[i] = bias_scale.const_data_ptr<float>()[0] * out_scale * 2147483648;
        out_shift32[i] = 0;
    }

    WORD32 out_zero_bias = output_zero_point;
    WORD32 inp_precision = 8;
    WORD32 kernel_precision = 8;
    pVOID p_scratch = nullptr;
    WORD8 *ptr_scratch;
    
    WORD32 scratch_size = 0;
    
    WORD32 out_data_format = 1;
    WORD32 inp_data_format = 0;

    WORD8 *ptr1 = (WORD8 *)malloc(((input.size(0) * input_channels * input_height * input_width) + 8) * sizeof(WORD8));
    WORD8 *ptr2 = (WORD8 *)malloc(((out_channels * kernel_channels * kernel_height * kernel_width) + 8) * sizeof(WORD8));
    
    WORD8 *pin = (WORD8 *)ALIGN_PTR(ptr1, 8);
    WORD8 *pkernel = (WORD8 *)ALIGN_PTR(ptr2, 8);
    
    WORD32 p_inp_shape[4];
    p_inp_shape[0] = input.size(0);
    p_inp_shape[1] = input_channels;
    p_inp_shape[2] = input_channels;
    p_inp_shape[3] = input_width;
    
    WORD32 p_out_shape[4];
    p_out_shape[0] = input.size(0);
    p_out_shape[1] = input_height;
    p_out_shape[2] = input_width;
    p_out_shape[3] = input_channels;
    
    WORD32 p_permute_vec[4] = {0, 2, 3, 1};
    
    WORD32 num_out_dims = 4;
    WORD32 num_inp_dims = 4;
    
    WORD8 * p_tmp = (WORD8 *)p_inp;
    
    WORD32 t = xa_nn_transpose_8_8(pin
                      ,p_out_shape
                      ,p_tmp
                      ,p_inp_shape
                      ,p_permute_vec
                      ,num_out_dims
                      ,num_inp_dims);
                      
    WORD32 p_inp_shape1[4];
    p_inp_shape1[0] = out_channels;
    p_inp_shape1[1] = kernel_channels;
    p_inp_shape1[2] = kernel_height;
    p_inp_shape1[3] = kernel_width;
    
    WORD32 p_out_shape1[4];
    p_out_shape1[0] = out_channels;
    p_out_shape1[1] = kernel_height;
    p_out_shape1[2] = kernel_width;
    p_out_shape1[3] = kernel_channels;
    
    WORD32 p_permute_vec1[4] = {0, 2, 3, 1};
    
    WORD32 num_out_dims1 = 4;
    WORD32 num_inp_dims1 = 4;
    
    WORD8 * p_tmp1 = (WORD8 *)p_kernel;
    
    WORD32 t1 = xa_nn_transpose_8_8(pkernel
                      ,p_out_shape1
                      ,p_tmp1
                      ,p_inp_shape1
                      ,p_permute_vec1
                      ,num_out_dims1
                      ,num_inp_dims1);      
    
    scratch_size = xa_nn_conv2d_getsize(
      input_height,
      input_width,
      input_channels,
      kernel_height,
      kernel_width,
      kernel_channels,
      dilation_height,
      dilation_width,
      y_stride,
      y_padding,
      x_stride,
      x_padding,
      out_height,
      out_width,
      out_channels,
      inp_precision,
      kernel_precision,
      out_data_format);
                                            
    scratch_size=scratch_size<0?0:(scratch_size);

    ptr_scratch = (WORD8 *)malloc(scratch_size + 16);
    
    p_scratch = (xa_codec_handle_t)ALIGN_PTR(ptr_scratch, 8);
    
    const UWORD8* __restrict__ p_inp1 = (const UWORD8* __restrict__ )pin;
    const UWORD8* __restrict__ p_kernel1 = (const UWORD8* __restrict__ )pkernel;

    for (int _n = 0; _n < batches; _n++) {
      const UWORD8* __restrict__ in_batch = p_inp1 + _n * input_channels * input_height * input_width;
      UWORD8* __restrict__ out_batch = p_out + _n * out_channels * out_height * out_width;
      
      WORD32 val = xa_nn_conv2d_per_chan_asym8xasym8
        (out_batch
        ,in_batch
        ,p_kernel1
        ,p_bias
        ,input_height
        ,input_width
        ,input_channels
        ,kernel_height
        ,kernel_width
        ,kernel_channels
        ,dilation_height
        ,dilation_width
        ,out_channels
        ,x_stride
        ,y_stride
        ,x_padding
        ,y_padding
        ,out_height
        ,out_width
        ,input_zero_bias
        ,out_multiplier32
        ,out_shift32
        ,out_zero_bias
        ,out_data_format
        ,p_scratch
      );
    }
    
    free(ptr1);
    free(ptr2);
    free(ptr_scratch);
  }
  else
  {
    ET_CHECK_MSG(false, "Unhandled input dtype %hhd", out.scalar_type());
  }
  
#else  
  // input = [n, c, h, w]
  const int n = input.size(0);
  const int c = input.size(1);
  const int h = conv1d ? 1 : input.size(2);
  const int w = conv1d ? input.size(2) : input.size(3);
  // weight = [oc, wc, wh, ww]
  const int oc = weight.size(0);
  const int wc = weight.size(1);
  const int wh = conv1d ? 1 : weight.size(2);
  const int ww = conv1d ? weight.size(2) : weight.size(3);
  // output = [n, oc, oh, ow]
  const int oh = conv1d ? 1 : out.size(2);
  const int ow = conv1d ? out.size(2) : out.size(3);

  // Bool flag to check if weight tensor is quantized per-tensor or
  // per-channel
  bool per_tensor_quantized = bias_scale.numel() == 1;

  if(input.scalar_type() == ScalarType::Char)
  {
    conv2d_nchw_core_generic<int8_t, int8_t, int32_t, int8_t, true>(
        input.const_data_ptr<int8_t>(),
        weight.const_data_ptr<int8_t>(),
        bias.const_data_ptr<int32_t>(),
        out.mutable_data_ptr<int8_t>(),
        n,
        c,
        h,
        w,
        oc,
        wc,
        wh,
        ww,
        oh,
        ow,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        1,//dilation[0],
        1,//dilation[1],
        groups,
        in_zero_point,
        weight_zero_point.const_data_ptr<int32_t>(),
        bias_scale.const_data_ptr<float>(),
        output_scale,
        (int8_t)output_zero_point,
        per_tensor_quantized);
      
  }
  else if(input.scalar_type() == ScalarType::Byte)
  {
    conv2d_nchw_core_generic<uint8_t, uint8_t, int32_t, uint8_t, true>(
        input.const_data_ptr<uint8_t>(),
        weight.const_data_ptr<uint8_t>(),
        bias.const_data_ptr<int32_t>(),
        out.mutable_data_ptr<uint8_t>(),
        n,
        c,
        h,
        w,
        oc,
        wc,
        wh,
        ww,
        oh,
        ow,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        1,//dilation[0],
        1,//dilation[1],
        groups,
        in_zero_point,
        weight_zero_point.const_data_ptr<int32_t>(),
        bias_scale.const_data_ptr<float>(),
        output_scale,
        (uint8_t)output_zero_point,
        per_tensor_quantized);
  }
  else
  {
    ET_CHECK_MSG(false, "Unhandled input dtype %hhd", out.scalar_type());
  }
#endif
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl
