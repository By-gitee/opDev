/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved. 
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "ball_query_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <string>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    BallQueryTilingData tiling;
    constexpr int P0 = 0;
    constexpr int P1 = 1;
    constexpr int P2 = 2;
    constexpr int D0 = 0;
    constexpr int D1 = 1;
    constexpr int V0 = 0;
    constexpr int BLK_NUM = 3;

    auto dataType = context->GetInputDesc(P0)->GetDataType();
    auto xyzShape = context->GetInputShape(P0)->GetStorageShape();
    auto ctrShape = context->GetInputShape(P1)->GetStorageShape();
    if (dataType == ge::DT_FLOAT16) {
        xyzShape.GetDimNum() == BLK_NUM ? tiling.set_tilingKey(static_cast<int64_t>(BallQueryTilingKey::INPUT_FP16_NORM)) : tiling.set_tilingKey(static_cast<int64_t>(BallQueryTilingKey::INPUT_FP16_STACK));
    } else if (dataType == ge::DT_FLOAT) {
        xyzShape.GetDimNum() == BLK_NUM ? tiling.set_tilingKey(static_cast<int64_t>(BallQueryTilingKey::INPUT_FP32_NORM)) : tiling.set_tilingKey(static_cast<int64_t>(BallQueryTilingKey::INPUT_FP32_STACK));
    }
    if(xyzShape.GetDimNum() == BLK_NUM){          // xyz:[B, N, 3]  ctr:[B, M, 3]
        tiling.set_B(xyzShape.GetDim(D0));
        tiling.set_N(xyzShape.GetDim(D1));
        tiling.set_M(ctrShape.GetDim(D1));
        tiling.set_bn(V0);
    } else{                                   // xyz:[N, 3]  ctr:[M, 3]
         tiling.set_B(V0);
        tiling.set_N(xyzShape.GetDim(D0));
        tiling.set_M(ctrShape.GetDim(D0));
        tiling.set_bn(context->GetInputShape(P2)->GetStorageShape().GetDim(D0));
    }
    auto attrs = context->GetAttrs();
    tiling.set_minr(*(attrs->GetAttrPointer<float>(P0)));
    tiling.set_maxr(*(attrs->GetAttrPointer<float>(P1)));
    tiling.set_sn(*(attrs->GetAttrPointer<int>(P2)));

    uint64_t UBSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    tiling.set_usedCoreNum(static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv()));
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, UBSize);
    tiling.set_UBSize(UBSize);

    context->SetBlockDim(tiling.get_usedCoreNum());
    context->SetTilingKey(tiling.get_tilingKey());

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class BallQuery : public OpDef {
  public:
    explicit BallQuery(const char *name) : OpDef(name) {
        this->Input("xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("center_xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("xyz_batch_cnt")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("center_xyz_batch_cnt")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("min_radius").Float();
        this->Attr("max_radius").Float();
        this->Attr("sample_num").Int();

        this->SetInferShape(ge::InferShape);
        
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(BallQuery);
} // namespace ops
