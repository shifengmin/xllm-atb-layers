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
#pragma once

#include <cstdint>
#include <string>

#include <atb/atb_infer.h>
#include <atb/operation_infra.h>
#include <hccl/hccl.h>

namespace atb_speed {
namespace common {

class HcclScatterOperation final : public atb::OperationInfra {
public:
    HcclScatterOperation(const std::string &name, int32_t rank, int32_t rank_size,
                         int32_t rank_root, HcclComm hccl_comm);
    ~HcclScatterOperation() override = default;

    std::string GetName() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &in_tensor_descs,
                           atb::SVector<atb::TensorDesc> &out_tensor_descs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status Setup(const atb::VariantPack &variant_pack, uint64_t &workspace_size,
                      atb::Context *context) override;
    atb::Status Execute(const atb::VariantPack &variant_pack, uint8_t *workspace,
                        uint64_t workspace_size, atb::Context *context) override;

private:
    std::string name_;
    int32_t rank_;
    int32_t rank_size_;
    int32_t rank_root_;
    HcclComm hccl_comm_;
};

}  // namespace common
}  // namespace atb_speed
