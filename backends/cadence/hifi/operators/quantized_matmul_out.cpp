/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::getLeadingDims;
using executorch::runtime::KernelRuntimeContext;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

// The quantized matmul. The quantized matmul accumulates in a wider register,
// whose type is TA.
template <
    typename TZ,
    typename TA = float,
    bool transposed = false,
    typename TX = TZ,
    typename TY = TZ>
__attribute__((noinline)) void qmatmul(
    TZ* __restrict__ Z,
    int32_t Z_multiplier,
    int32_t Z_shift,
    int32_t Z_zero_point,
    const TX* __restrict__ X,
    int32_t X_zero_point,
    const TY* __restrict__ y,
    int32_t Y_zero_point,
    size_t m,
    size_t n,
    size_t p) {
  // Compute the Z_scale from Z_multiplier and Z_shift
  const float Z_scale = -Z_multiplier * 1.0 / (1 << 31) * pow(2, Z_shift);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      TA sum = 0;
      for (size_t k = 0; k < n; ++k) {
        if (transposed) {
          sum += (X[i * n + k] - X_zero_point) * (y[j * n + k] - Y_zero_point);
        } else {
          sum += (X[i * n + k] - X_zero_point) * (y[k * p + j] - Y_zero_point);
        }
      }
      Z[i * p + j] = kernels::quantize<TZ>(sum, Z_scale, Z_zero_point);
    }
  }
}

template <typename T>
void inline _typed_quantized_matmul(
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    const exec_aten::optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out) {
  T* __restrict__ out_data = out.mutable_data_ptr<T>();
  const T* __restrict__ X_data = X.const_data_ptr<T>();
  const T* __restrict__ Y_data = Y.const_data_ptr<T>();
  size_t batch_size = getLeadingDims(X, X.dim() - 2);
  size_t leading_dim = X.size(X.dim() - 2);
  size_t out_dim = Y.size(Y.dim() - 1 - transposed);
  size_t in_dim = X.size(X.dim() - 1);
  const int32_t* __restrict__ bias_data =
      (int32_t*)calloc(leading_dim * in_dim, 4);
  uint8_t* y_data_temp1 = NULL;
  int8_t* y_data_temp2 = NULL;

  if (!transposed) {
    y_data_temp1 = (uint8_t*)malloc(leading_dim * in_dim);
    y_data_temp2 = (int8_t*)malloc(leading_dim * in_dim);
  }

  for (size_t i = 0; i < batch_size; ++i) {
    const T* x = X_data + i * leading_dim * in_dim;
    const T* y = Y_data + i * in_dim * out_dim;
    T* z = out_data + i * leading_dim * out_dim;
    if (transposed) {
      if (out.scalar_type() == exec_aten::ScalarType::Byte) {
        xa_nn_matmul_asym8uxasym8u_asym8u(
            (uint8_t*)z, // p_out
            (uint8_t*)y, // p_mat1,
            (uint8_t*)x, // p_mat2,
            bias_data, // p_bias
            out_dim, // rows of p_mat1
            in_dim, // cols of p_mat1
            in_dim, // row_stride of p_mat1
            leading_dim, // vec_count, i.e., rows of p_mat2
            in_dim, // vec_offset of p_mat2.
            out_dim, // out_offset, i.e., offset of next output element written
            1, // out_stride, i.e., stride to go to next output row
            -(static_cast<int32_t>(Y_zero_point)), // mat1_zero_bias
            -(static_cast<int32_t>(X_zero_point)), // mat2_zero_bias
            static_cast<int32_t>(out_multiplier), // out_multiplier
            static_cast<int32_t>(out_shift), // out_shift
            static_cast<int32_t>(out_zero_point)); // out_zero_bias
      } else if (out.scalar_type() == exec_aten::ScalarType::Char) {
        xa_nn_matmul_asym8sxasym8s_asym8s(
            (int8_t*)z, // p_out
            (int8_t*)y, // p_mat1,
            (int8_t*)x, // p_mat2,
            bias_data, // p_bias
            out_dim, // rows of p_mat1
            in_dim, // cols of p_mat1
            in_dim, // row_stride of p_mat1
            leading_dim, // vec_count, i.e., rows of p_mat2
            in_dim, // vec_offset of p_mat2.
            out_dim, // out_offset, i.e., offset of next output element written
            1, // out_stride, i.e., stride to go to next output row
            -(static_cast<int32_t>(Y_zero_point)), // mat1_zero_bias
            -(static_cast<int32_t>(X_zero_point)), // mat2_zero_bias
            static_cast<int32_t>(out_multiplier), // out_multiplier
            static_cast<int32_t>(out_shift), // out_shift
            static_cast<int32_t>(out_zero_point)); // out_zero_bias
      }

    } else {
      if (out.scalar_type() == exec_aten::ScalarType::Byte) {
        /* Assuming matmul is 2D always */
        WORD32 num_inp_dims = 2;
        WORD32 num_out_dims = 2;

        WORD32 p_inp_shape[2];
        WORD32 p_out_shape[2];
        WORD32 p_permute_vec[2] = {1, 0};

        p_inp_shape[0] = leading_dim;
        p_inp_shape[1] = in_dim;
        p_out_shape[0] = in_dim;
        p_out_shape[1] = leading_dim;

        xa_nn_transpose_8_8(
            (int8_t*)y_data_temp1,
            p_out_shape,
            (int8_t*)y,
            p_inp_shape,
            p_permute_vec,
            num_out_dims,
            num_inp_dims);

        xa_nn_matmul_asym8uxasym8u_asym8u(
            (uint8_t*)z, // p_out
            (uint8_t*)y_data_temp1, // p_mat1,
            (uint8_t*)x, // p_mat2,
            bias_data, // p_bias
            out_dim, // rows of p_mat1
            in_dim, // cols of p_mat1
            in_dim, // row_stride of p_mat1
            leading_dim, // vec_count, i.e., rows of p_mat2
            in_dim, // vec_offset of p_mat2.
            out_dim, // out_offset, i.e., offset of next output element written
            1, // out_stride, i.e., stride to go to next output row
            -(static_cast<int32_t>(Y_zero_point)), // mat1_zero_bias
            -(static_cast<int32_t>(X_zero_point)), // mat2_zero_bias
            static_cast<int32_t>(out_multiplier), // out_multiplier
            static_cast<int32_t>(out_shift), // out_shift
            static_cast<int32_t>(out_zero_point)); // out_zero_bias
      } else if (out.scalar_type() == exec_aten::ScalarType::Char) {
        /* Assuming matmul is 2D always */
        WORD32 num_inp_dims = 2;
        WORD32 num_out_dims = 2;

        WORD32 p_inp_shape[2];
        WORD32 p_out_shape[2];
        WORD32 p_permute_vec[2] = {1, 0};

        p_inp_shape[0] = leading_dim;
        p_inp_shape[1] = in_dim;
        p_out_shape[0] = in_dim;
        p_out_shape[1] = leading_dim;

        xa_nn_transpose_8_8(
            (int8_t*)y_data_temp2,
            p_out_shape,
            (int8_t*)y,
            p_inp_shape,
            p_permute_vec,
            num_out_dims,
            num_inp_dims);

        xa_nn_matmul_asym8sxasym8s_asym8s(
            (int8_t*)z, // p_out
            (int8_t*)y_data_temp2, // p_mat1,
            (int8_t*)x, // p_mat2,
            bias_data, // p_bias
            out_dim, // rows of p_mat1
            in_dim, // cols of p_mat1
            in_dim, // row_stride of p_mat1
            leading_dim, // vec_count, i.e., rows of p_mat2
            in_dim, // vec_offset of p_mat2.
            out_dim, // out_offset, i.e., offset of next output element written
            1, // out_stride, i.e., stride to go to next output row
            -(static_cast<int32_t>(Y_zero_point)), // mat1_zero_bias
            -(static_cast<int32_t>(X_zero_point)), // mat2_zero_bias
            static_cast<int32_t>(out_multiplier), // out_multiplier
            static_cast<int32_t>(out_shift), // out_shift
            static_cast<int32_t>(out_zero_point)); // out_zero_bias
      }
    }
  }
  free((void*)bias_data);
  if (y_data_temp1 != NULL)
    free(y_data_temp1);
  if (y_data_temp2 != NULL)
    free(y_data_temp2);
}

void quantized_matmul_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    const executorch::aten::optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out) {
  if (out.scalar_type() == ScalarType::Byte) {
    _typed_quantized_matmul<uint8_t>(
        X,
        X_zero_point,
        Y,
        Y_zero_point,
        bias,
        out_multiplier,
        out_shift,
        out_zero_point,
        transposed,
        out);
  } else if (out.scalar_type() == ScalarType::Char) {
    _typed_quantized_matmul<int8_t>(
        X,
        X_zero_point,
        Y,
        Y_zero_point,
        bias,
        out_multiplier,
        out_shift,
        out_zero_point,
        transposed,
        out);
  } else {
    ET_CHECK_MSG(
        false,
        "Unhandled input dtype %hhd",
        static_cast<int8_t>(X.scalar_type()));
  }
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence