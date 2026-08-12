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

#pragma once

#include <atb/operation_infra.h>

#include <cstdint>
#include <string>

namespace atb_speed {

class TopkLogicalToPhysicalSlotsOperation final
    : public atb::OperationInfra {
 public:
  explicit TopkLogicalToPhysicalSlotsOperation(int32_t block_size);
  ~TopkLogicalToPhysicalSlotsOperation() override = default;

  std::string GetName() const override;
  atb::Status InferShape(
      const atb::SVector<atb::TensorDesc>& in_tensor_descs,
      atb::SVector<atb::TensorDesc>& out_tensor_descs) const override;
  uint32_t GetInputNum() const override;
  uint32_t GetOutputNum() const override;
  atb::Status Setup(const atb::VariantPack& variant_pack,
                    uint64_t& workspace_size,
                    atb::Context* context) override;
  atb::Status Execute(const atb::VariantPack& variant_pack,
                      uint8_t* workspace,
                      uint64_t workspace_size,
                      atb::Context* context) override;

 private:
  int32_t block_size_;
};

}  // namespace atb_speed
