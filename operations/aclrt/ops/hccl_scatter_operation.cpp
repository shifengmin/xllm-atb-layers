/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "operations/aclrt/ops/hccl_scatter_operation.h"

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <vector>

#include <atb_speed/log.h>

namespace atb_speed {
namespace common {
namespace {

HcclDataType GetHcclDataType(aclDataType dtype)
{
    switch (dtype) {
        case ACL_INT8:
            return HCCL_DATA_TYPE_INT8;
        case ACL_UINT8:
            return HCCL_DATA_TYPE_UINT8;
        case ACL_INT16:
            return HCCL_DATA_TYPE_INT16;
        case ACL_UINT16:
            return HCCL_DATA_TYPE_UINT16;
        case ACL_INT32:
            return HCCL_DATA_TYPE_INT32;
        case ACL_UINT32:
            return HCCL_DATA_TYPE_UINT32;
        case ACL_INT64:
            return HCCL_DATA_TYPE_INT64;
        case ACL_UINT64:
            return HCCL_DATA_TYPE_UINT64;
        case ACL_FLOAT16:
            return HCCL_DATA_TYPE_FP16;
        case ACL_FLOAT:
            return HCCL_DATA_TYPE_FP32;
        case ACL_DOUBLE:
            return HCCL_DATA_TYPE_FP64;
        case ACL_BF16:
            return HCCL_DATA_TYPE_BFP16;
        default:
            return HCCL_DATA_TYPE_RESERVED;
    }
}

uint64_t GetElementCount(const atb::Dims &shape)
{
    uint64_t count = 1;
    for (uint32_t index = 0; index < shape.dimNum; ++index) {
        if (shape.dims[index] <= 0) {
            return 0;
        }
        count *= static_cast<uint64_t>(shape.dims[index]);
    }
    return count;
}

}  // namespace

HcclScatterOperation::HcclScatterOperation(const std::string &name, int32_t rank,
                                           int32_t rank_size, int32_t rank_root,
                                           HcclComm hccl_comm)
    : name_(name), rank_(rank), rank_size_(rank_size), rank_root_(rank_root), hccl_comm_(hccl_comm)
{
}

std::string HcclScatterOperation::GetName() const
{
    return name_;
}

atb::Status HcclScatterOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &in_tensor_descs,
    atb::SVector<atb::TensorDesc> &out_tensor_descs) const
{
    if (in_tensor_descs.size() != 1 || out_tensor_descs.size() != 1 || rank_size_ <= 0 ||
        rank_ < 0 || rank_ >= rank_size_ || rank_root_ < 0 || rank_root_ >= rank_size_) {
        return atb::ERROR_INVALID_PARAM;
    }
    const atb::TensorDesc &input_desc = in_tensor_descs.at(0);
    if (input_desc.shape.dimNum <= 1 || input_desc.shape.dims[0] != rank_size_) {
        return atb::ERROR_INVALID_PARAM;
    }
    out_tensor_descs.at(0) = input_desc;
    --out_tensor_descs.at(0).shape.dimNum;
    for (uint32_t index = 0; index < out_tensor_descs.at(0).shape.dimNum; ++index) {
        out_tensor_descs.at(0).shape.dims[index] = input_desc.shape.dims[index + 1];
    }
    return atb::NO_ERROR;
}

uint32_t HcclScatterOperation::GetInputNum() const
{
    return 1;
}

uint32_t HcclScatterOperation::GetOutputNum() const
{
    return 1;
}

atb::Status HcclScatterOperation::Setup(const atb::VariantPack &variant_pack,
                                        uint64_t &workspace_size, atb::Context *context)
{
    if (context == nullptr || variant_pack.inTensors.size() != 1 ||
        variant_pack.outTensors.size() != 1) {
        return atb::ERROR_INVALID_PARAM;
    }
    workspace_size = 0;
    return atb::NO_ERROR;
}

atb::Status HcclScatterOperation::Execute(const atb::VariantPack &variant_pack, uint8_t *,
                                          uint64_t, atb::Context *context)
{
    if (context == nullptr) {
        ATB_SPEED_LOG_ERROR(name_ << " execute context is null");
        return atb::ERROR_INVALID_PARAM;
    }
    if (variant_pack.inTensors.size() != 1 || variant_pack.outTensors.size() != 1) {
        ATB_SPEED_LOG_ERROR(name_ << " invalid tensor count, inputs="
                                  << variant_pack.inTensors.size() << ", outputs="
                                  << variant_pack.outTensors.size());
        return atb::ERROR_INVALID_PARAM;
    }
    if (variant_pack.inTensors.at(0).deviceData == nullptr ||
        variant_pack.outTensors.at(0).deviceData == nullptr) {
        ATB_SPEED_LOG_ERROR(name_ << " input or output device data is null");
        return atb::ERROR_INVALID_PARAM;
    }
    if (hccl_comm_ == nullptr) {
        ATB_SPEED_LOG_ERROR(name_ << " HCCL communicator is null");
        return atb::ERROR_INVALID_PARAM;
    }
    const atb::Tensor &output_tensor = variant_pack.outTensors.at(0);
    const HcclDataType data_type = GetHcclDataType(output_tensor.desc.dtype);
    const uint64_t recv_count = GetElementCount(output_tensor.desc.shape);
    if (data_type == HCCL_DATA_TYPE_RESERVED || recv_count == 0) {
        return atb::ERROR_INVALID_PARAM;
    }
    static std::atomic<uint32_t> debug_count{0};
    const char *debug_layout = std::getenv("XLLM_DEBUG_HCCL_SCATTER_LAYOUT");
    const bool capture_layout = debug_layout != nullptr && debug_layout[0] == '1' &&
                                debug_count.fetch_add(1) == 0;
    const atb::Tensor &input_tensor = variant_pack.inTensors.at(0);
    if (capture_layout) {
        std::ostringstream input_shape;
        for (uint32_t index = 0; index < input_tensor.desc.shape.dimNum; ++index) {
            if (index != 0) {
                input_shape << 'x';
            }
            input_shape << input_tensor.desc.shape.dims[index];
        }
        std::ostringstream output_shape;
        for (uint32_t index = 0; index < output_tensor.desc.shape.dimNum; ++index) {
            if (index != 0) {
                output_shape << 'x';
            }
            output_shape << output_tensor.desc.shape.dims[index];
        }
        ATB_SPEED_LOG_ERROR(name_ << " layout rank=" << rank_ << "/" << rank_size_
                                  << " root=" << rank_root_ << " input_ptr="
                                  << input_tensor.deviceData << " output_ptr="
                                  << output_tensor.deviceData << " input_bytes="
                                  << input_tensor.dataSize << " output_bytes="
                                  << output_tensor.dataSize << " input_shape="
                                  << input_shape.str() << " output_shape="
                                  << output_shape.str() << " input_dtype="
                                  << input_tensor.desc.dtype << " output_dtype="
                                  << output_tensor.desc.dtype << " stream="
                                  << GetExecuteStream(context));
    }
    const HcclResult result = HcclScatter(
        variant_pack.inTensors.at(0).deviceData, output_tensor.deviceData, recv_count, data_type,
        static_cast<uint32_t>(rank_root_), hccl_comm_, GetExecuteStream(context));
    if (result != HCCL_SUCCESS) {
        ATB_SPEED_LOG_ERROR(name_ << " HcclScatter failed, result=" << result);
        return atb::ERROR_INTERNAL_ERROR;
    }
    const char *debug_sync = std::getenv("XLLM_DEBUG_HCCL_SCATTER_SYNC");
    if (debug_sync != nullptr && debug_sync[0] == '1' &&
        aclrtSynchronizeStream(GetExecuteStream(context)) != ACL_SUCCESS) {
        ATB_SPEED_LOG_ERROR(name_ << " debug stream synchronization failed");
        return atb::ERROR_INTERNAL_ERROR;
    }
    if (capture_layout) {
        void *gathered_data = nullptr;
        if (input_tensor.dataSize != output_tensor.dataSize * static_cast<uint64_t>(rank_size_) ||
            aclrtMalloc(&gathered_data, input_tensor.dataSize, ACL_MEM_MALLOC_HUGE_ONLY) != ACL_SUCCESS) {
            ATB_SPEED_LOG_ERROR(name_ << " debug all-gather buffer setup failed");
            return atb::ERROR_INTERNAL_ERROR;
        }
        const HcclResult gather_result = HcclAllGather(
            output_tensor.deviceData, gathered_data, recv_count, data_type, hccl_comm_,
            GetExecuteStream(context));
        if (gather_result != HCCL_SUCCESS ||
            aclrtSynchronizeStream(GetExecuteStream(context)) != ACL_SUCCESS) {
            ATB_SPEED_LOG_ERROR(name_ << " debug all-gather failed, result=" << gather_result);
            aclrtFree(gathered_data);
            return atb::ERROR_INTERNAL_ERROR;
        }
        if (rank_ == rank_root_) {
            std::vector<uint8_t> input_host(input_tensor.dataSize);
            std::vector<uint8_t> gathered_host(input_tensor.dataSize);
            const aclError input_copy_result = aclrtMemcpy(
                input_host.data(), input_host.size(), input_tensor.deviceData,
                input_tensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
            const aclError gathered_copy_result = aclrtMemcpy(
                gathered_host.data(), gathered_host.size(), gathered_data,
                input_tensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
            if (input_copy_result != ACL_SUCCESS || gathered_copy_result != ACL_SUCCESS) {
                ATB_SPEED_LOG_ERROR(name_ << " debug host copy failed, input=" << input_copy_result
                                          << " gathered=" << gathered_copy_result);
                aclrtFree(gathered_data);
                return atb::ERROR_INTERNAL_ERROR;
            }
            const bool matches = std::memcmp(input_host.data(), gathered_host.data(),
                                             input_host.size()) == 0;
            ATB_SPEED_LOG_ERROR(name_ << " debug rank-major round trip "
                                      << (matches ? "PASS" : "FAIL")
                                      << ", bytes=" << input_host.size());
        }
        aclrtFree(gathered_data);
    }
    return atb::NO_ERROR;
}

}  // namespace common
}  // namespace atb_speed
