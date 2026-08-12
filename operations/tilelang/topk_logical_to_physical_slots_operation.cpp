/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "operations/tilelang/topk_logical_to_physical_slots_operation.h"

#include <acl/acl.h>

#include <cstdint>
#include <limits>

#include "atb_speed/log.h"
#include "xllm/core/kernels/npu/tilelang/tilelang_atb_ops_api.h"

namespace atb_speed {
namespace {

constexpr uint32_t kInputCount = 4;
constexpr uint32_t kOutputCount = 1;

bool get_tensor_numel(const atb::TensorDesc& desc, int64_t* numel) {
  if (numel == nullptr) {
    return false;
  }
  *numel = 1;
  for (uint64_t dim = 0; dim < desc.shape.dimNum; ++dim) {
    const int64_t extent = desc.shape.dims[dim];
    if (extent <= 0 ||
        *numel > std::numeric_limits<int64_t>::max() / extent) {
      return false;
    }
    *numel *= extent;
  }
  return true;
}

bool is_int32_tensor(const atb::Tensor& tensor) {
  return tensor.desc.dtype == ACL_INT32 && tensor.deviceData != nullptr;
}

}  // namespace

TopkLogicalToPhysicalSlotsOperation::TopkLogicalToPhysicalSlotsOperation(
    int32_t block_size)
    : block_size_(block_size) {}

std::string TopkLogicalToPhysicalSlotsOperation::GetName() const {
  return "TopkLogicalToPhysicalSlotsOperation";
}

uint32_t TopkLogicalToPhysicalSlotsOperation::GetInputNum() const {
  return kInputCount;
}

uint32_t TopkLogicalToPhysicalSlotsOperation::GetOutputNum() const {
  return kOutputCount;
}

atb::Status TopkLogicalToPhysicalSlotsOperation::InferShape(
    const atb::SVector<atb::TensorDesc>& in_tensor_descs,
    atb::SVector<atb::TensorDesc>& out_tensor_descs) const {
  if (in_tensor_descs.size() != kInputCount ||
      out_tensor_descs.size() != kOutputCount) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  out_tensor_descs.at(0) = in_tensor_descs.at(2);
  return atb::NO_ERROR;
}

atb::Status TopkLogicalToPhysicalSlotsOperation::Setup(
    const atb::VariantPack& variant_pack,
    uint64_t& workspace_size,
    atb::Context* context) {
  if (context == nullptr || block_size_ <= 0 ||
      variant_pack.inTensors.size() != kInputCount ||
      variant_pack.outTensors.size() != kOutputCount) {
    return atb::ERROR_INVALID_PARAM;
  }
  workspace_size = 0;
  return atb::NO_ERROR;
}

atb::Status TopkLogicalToPhysicalSlotsOperation::Execute(
    const atb::VariantPack& variant_pack,
    uint8_t* workspace,
    uint64_t workspace_size,
    atb::Context* context) {
  if (context == nullptr || block_size_ <= 0 ||
      variant_pack.inTensors.size() != kInputCount ||
      variant_pack.outTensors.size() != kOutputCount) {
    return atb::ERROR_INVALID_PARAM;
  }
  static_cast<void>(workspace);
  static_cast<void>(workspace_size);
  for (const atb::Tensor& tensor : variant_pack.inTensors) {
    if (!is_int32_tensor(tensor)) {
      return atb::ERROR_INVALID_TENSOR_DTYPE;
    }
  }
  if (!is_int32_tensor(variant_pack.outTensors.at(0))) {
    return atb::ERROR_INVALID_TENSOR_DTYPE;
  }

  const atb::Tensor& topk_positions = variant_pack.inTensors.at(0);
  const atb::Tensor& block_tables = variant_pack.inTensors.at(1);
  const atb::Tensor& packed_indices = variant_pack.inTensors.at(2);
  const atb::Tensor& packed_rows = variant_pack.inTensors.at(3);
  const atb::Tensor& physical_slots = variant_pack.outTensors.at(0);
  if (block_tables.desc.shape.dimNum != 2 ||
      packed_indices.desc.shape.dimNum != 1 ||
      packed_rows.desc.shape.dimNum != 1 ||
      physical_slots.desc.shape.dimNum != 1) {
    return atb::ERROR_INVALID_TENSOR_DIM;
  }

  int64_t topk_numel = 0;
  int64_t packed_count = 0;
  int64_t packed_rows_count = 0;
  int64_t physical_slots_count = 0;
  if (!get_tensor_numel(topk_positions.desc, &topk_numel) ||
      !get_tensor_numel(packed_indices.desc, &packed_count) ||
      !get_tensor_numel(packed_rows.desc, &packed_rows_count) ||
      !get_tensor_numel(physical_slots.desc, &physical_slots_count)) {
    return atb::ERROR_INVALID_TENSOR_SIZE;
  }
  const int64_t block_table_rows = block_tables.desc.shape.dims[0];
  const int64_t block_table_cols = block_tables.desc.shape.dims[1];
  if (topk_numel <= 0 || block_table_rows <= 0 || block_table_cols <= 0 ||
      packed_count <= 0 || topk_numel < packed_count ||
      topk_numel > std::numeric_limits<int32_t>::max() ||
      block_table_rows > std::numeric_limits<int32_t>::max() ||
      block_table_cols > std::numeric_limits<int32_t>::max() ||
      packed_count > std::numeric_limits<int32_t>::max() ||
      packed_rows_count != packed_count ||
      physical_slots_count != packed_count) {
    return atb::ERROR_INVALID_TENSOR_SIZE;
  }

  xllm::kernel::npu::tilelang::launch_topk_logical_to_physical_slots(
      topk_positions.deviceData,
      block_tables.deviceData,
      packed_indices.deviceData,
      packed_rows.deviceData,
      physical_slots.deviceData,
      topk_numel,
      block_table_rows,
      block_table_cols,
      packed_count,
      block_size_,
      context->GetExecuteStream());
  return atb::NO_ERROR;
}

}  // namespace atb_speed
