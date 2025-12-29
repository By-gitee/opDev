/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "graph/utils/type_utils.h"
#include "ones_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  const uint32_t NUM_DOUBLE_BUFFER = 2;
  const uint32_t ALIGN_SIZE = 32;
  const uint32_t ALIGN_DOWN_SIZE = 31;
  OnesTilingData tiling;
  const gert::Tensor *hTensor = context->GetInputTensor(0);
  const gert::Tensor *wTensor = context->GetInputTensor(1);
  uint32_t typeLength = 0;
  ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(),
                                   typeLength);
  uint64_t ubSize;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  auto maxOnceLength =
      ubSize / typeLength /
      NUM_DOUBLE_BUFFER; // 开启double Buffer 时每次处理的最大数据量
  maxOnceLength = (maxOnceLength - ALIGN_DOWN_SIZE) / ALIGN_SIZE *
                  ALIGN_SIZE; // 向下取32的整数倍，防止超UB
  const int64_t *hData = hTensor->GetData<int64_t>();
  const int64_t *wData = wTensor->GetData<int64_t>();
  auto totalLength = hData[0] * wData[0];
  auto processTime = (totalLength + maxOnceLength - 1) / maxOnceLength;
  auto lastLength = totalLength % maxOnceLength;
  lastLength = (lastLength == 0) ? maxOnceLength : lastLength;
  tiling.set_processTime(processTime);
  tiling.set_onesLength(maxOnceLength);
  tiling.set_lastLength(lastLength);

  context->SetBlockDim(1);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Tensor *hTensor = context->GetInputTensor(0);
  const gert::Tensor *wTensor = context->GetInputTensor(1);
  const int64_t *hData = hTensor->GetData<int64_t>();
  const int64_t *wData = wTensor->GetData<int64_t>();
  gert::Shape out = gert::Shape({hData[0], wData[0]});

  gert::Shape *y_shape = context->GetOutputShape(0);

  y_shape = &out;
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Ones : public OpDef {
public:
  explicit Ones(const char *name) : OpDef(name) {
    this->Input("h")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .ValueDepend(REQUIRED) // ValueDepend input dtype of op Ones must be
                               // float, int64 or bool
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("w")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .ValueDepend(REQUIRED)
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend310b");
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(Ones);
} // namespace ops
