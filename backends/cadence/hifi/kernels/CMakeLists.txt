# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# lint_cmake: -linelength
add_library(
  cadence_kernels
  kernels.cpp
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/matmul_asym8uxasym8u_asym8u.cpp
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_quantize_custom.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_layer_norm_asym8u.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_conv2d_asym8xasym8.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_conv2d_std_circ_buf.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_matXvec_asym8xasym8_asym8_circ.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_activations_8_8.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_div_mod_f32.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_transpose_32.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_add_f32_broadcast.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_mul_f32_broadcast.c
)

target_include_directories(
  cadence_kernels
  PUBLIC .
         ${NN_LIB_BASE_DIR}/xa_nnlib/algo/common/include/
         ${NN_LIB_BASE_DIR}/xa_nnlib/algo/kernels/cnn/hifi4
         ${NN_LIB_BASE_DIR}/xa_nnlib/include/nnlib
         ${NN_LIB_BASE_DIR}/xa_nnlib/include
         ${NN_LIB_BASE_DIR}/xa_nnlib/algo/ndsp/hifi4/include/
         ${NN_LIB_BASE_DIR}/xa_nnlib/algo/kernels/basic/hifi4/
)

target_link_libraries(cadence_kernels PRIVATE xa_nnlib)