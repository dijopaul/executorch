/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include "kernels.h"

#define NNLIB_OPT 1

namespace impl {
namespace HiFi {
namespace native {

using Tensor = exec_aten::Tensor;
using RuntimeContext = torch::executor::RuntimeContext;

// Note: this kernel assumes that the input and output share quantization
// parameters. If that is not the case, it will produce incorrect results.
template <typename T>
void quantized_relu_(
    const Tensor& input,
    const Tensor& in_zero_point,
    Tensor& output) {
  T q_zero_point = in_zero_point.const_data_ptr<T>()[0];
  const T* __restrict__ in = input.const_data_ptr<T>();
  T* __restrict__ out = output.mutable_data_ptr<T>();

  for (size_t i = 0, e = input.numel(); i < e; ++i) {
    out[i] = in[i] > q_zero_point ? in[i] : q_zero_point;
  }
}

void quantized_relu_out(
    RuntimeContext& ctx,
    const Tensor& input,
    const Tensor& in_zero_point,
    Tensor& output) {
        
  if (input.scalar_type() == exec_aten::ScalarType::Byte) {
#if NNLIB_OPT
    const uint8_t *p_in = input.const_data_ptr<uint8_t>();
    uint8_t *p_out =output.mutable_data_ptr<uint8_t>();
    uint8_t q_zero_point = in_zero_point.const_data_ptr<uint8_t>()[0];
    /*xa_nn_vec_relu_asym8u_asym8u( p_out, p_in, 0, 0x7FFFFFFF, 0, 0, 0,
                            (WORD32)q_zero_point,
                            input.numel());*/
    xa_nn_vec_relu_8u_8u_custom(p_out, p_in, q_zero_point, input.numel());                    
#else
    quantized_relu_<uint8_t>(input, in_zero_point, output);
#endif

  } else if (input.scalar_type() == exec_aten::ScalarType::Char) {
#if NNLIB_OPT
    const int8_t *p_in = input.const_data_ptr<int8_t>();
    int8_t *p_out = output.mutable_data_ptr<int8_t>();
    int8_t q_zero_point = in_zero_point.const_data_ptr<int8_t>()[0];
    xa_nn_vec_relu_8_8_custom(p_out, p_in, q_zero_point, input.numel());
#else    
    quantized_relu_<int8_t>(input, in_zero_point, output);
#endif    
  } else {
    ET_CHECK_MSG(false, "Unhandled input dtype %hhd", input.scalar_type());
  }
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl
