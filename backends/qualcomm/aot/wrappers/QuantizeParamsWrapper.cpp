/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/aot/wrappers/QuantizeParamsWrapper.h>
#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>

namespace executorch {
namespace backends {
namespace qnn {
std::unique_ptr<QuantizeParamsWrapper> CreateQuantizationParamWrapper(
    const Qnn_Tensor_t& tensor) {
  std::unique_ptr<QuantizeParamsWrapper> quantize_param_wrapper;
  auto& quantization = QNN_TENSOR_VER_PTR(tensor)->quantizeParams;

  if (quantization.quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_UNDEFINED) {
    quantize_param_wrapper = std::make_unique<UndefinedQuantizeParamsWrapper>();
  } else if (
      quantization.quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    std::vector<Qnn_ScaleOffset_t> scale_offset(
        quantization.axisScaleOffsetEncoding.scaleOffset,
        quantization.axisScaleOffsetEncoding.scaleOffset +
            quantization.axisScaleOffsetEncoding.numScaleOffsets);
    quantize_param_wrapper =
        std::make_unique<AxisScaleOffsetQuantizeParamsWrapper>(
            quantization.axisScaleOffsetEncoding.axis, scale_offset);
  } else if (
      quantization.quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
    std::vector<float> scales(
        quantization.bwAxisScaleOffsetEncoding.scales,
        quantization.bwAxisScaleOffsetEncoding.scales +
            quantization.bwAxisScaleOffsetEncoding.numElements);
    std::vector<int32_t> offsets(
        quantization.bwAxisScaleOffsetEncoding.offsets,
        quantization.bwAxisScaleOffsetEncoding.offsets +
            quantization.bwAxisScaleOffsetEncoding.numElements);

    quantize_param_wrapper =
        std::make_unique<BwAxisScaleOffsetQuantizeParamsWrapper>(
            quantization.bwAxisScaleOffsetEncoding.bitwidth,
            quantization.bwAxisScaleOffsetEncoding.axis,
            quantization.bwAxisScaleOffsetEncoding.numElements,
            scales,
            offsets);
  } else if (
      quantization.quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET) {
    quantize_param_wrapper =
        std::make_unique<BwScaleOffsetQuantizeParamsWrapper>(
            quantization.bwScaleOffsetEncoding.bitwidth,
            quantization.bwScaleOffsetEncoding.scale,
            quantization.bwScaleOffsetEncoding.offset);
  } else if (
      quantization.quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    quantize_param_wrapper = std::make_unique<ScaleOffsetQuantizeParamsWrapper>(
        quantization.scaleOffsetEncoding.scale,
        quantization.scaleOffsetEncoding.offset);
  } else if (
      quantization.quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION) {
    int ch_axis = quantization.blockwiseExpansion->axis;
    int ele_sz = quantization.blockwiseExpansion->blockScaleStorageType ==
            QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16
        ? 2
        : 1;
    size_t block_scales_sz = quantization.blockwiseExpansion->numBlocksPerAxis *
        QNN_TENSOR_VER_PTR(tensor)->dimensions[ch_axis] * ele_sz;
    std::vector<Qnn_ScaleOffset_t> scale_offsets(
        quantization.blockwiseExpansion->scaleOffsets,
        quantization.blockwiseExpansion->scaleOffsets +
            QNN_TENSOR_VER_PTR(tensor)->dimensions[ch_axis]);
    quantize_param_wrapper =
        std::make_unique<BlockwiseExpansionQuantizeParamsWrapper>(
            quantization.blockwiseExpansion->axis,
            scale_offsets,
            quantization.blockwiseExpansion->numBlocksPerAxis,
            quantization.blockwiseExpansion->blockScaleBitwidth,
            quantization.blockwiseExpansion->blockScaleStorageType,
            quantization.blockwiseExpansion->blocksScale8,
            block_scales_sz);
  } else {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unknown the encoding of quantization: %d",
        quantization.quantizationEncoding);
  }

  return quantize_param_wrapper;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
