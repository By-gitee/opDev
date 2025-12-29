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
#include "depth_to_space_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr int zero = 0;
constexpr int one = 1;
constexpr int two = 2;
constexpr int three = 3;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    DepthToSpaceTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    tiling.set_usedCoreNum(static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv()));
    uint64_t UBSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, UBSize);
    tiling.set_UBSize(UBSize);

    auto xShape = context->GetInputShape(zero)->GetStorageShape();
    uint32_t nSize, cSize, hSize, wSize;
    auto attrs = context->GetAttrs();
    const int block_size = *attrs->GetAttrPointer<int>(zero);
    tiling.set_blockSize(static_cast<uint16_t>(block_size));
    const char* data_format = attrs->GetAttrPointer<char>(two) == nullptr ? (char*)"NHWC" : attrs->GetAttrPointer<char>(two);
    if (strcmp(data_format, "NHWC") == 0) {
        nSize = static_cast<uint32_t>(xShape.GetDim(zero));
        cSize = static_cast<uint32_t>(xShape.GetDim(three));
        hSize = static_cast<uint32_t>(xShape.GetDim(one));
        wSize = static_cast<uint32_t>(xShape.GetDim(two));
        if (cSize == static_cast<uint16_t>(block_size) * static_cast<uint16_t>(block_size)) {
            tiling.set_tilingKey(static_cast<int64_t>(RNNTilingKey::NHWC_C1));
        } else if (cSize > static_cast<uint16_t>(block_size) * static_cast<uint16_t>(block_size)) {
            tiling.set_tilingKey(static_cast<int64_t>(RNNTilingKey::NHWC_NO_C1));
        }
    } else {
        tiling.set_tilingKey(static_cast<int64_t>(RNNTilingKey::NCHW_C1));
        nSize = static_cast<uint32_t>(xShape.GetDim(zero));
        cSize = static_cast<uint32_t>(xShape.GetDim(one));
        hSize = static_cast<uint32_t>(xShape.GetDim(two));
        wSize = static_cast<uint32_t>(xShape.GetDim(three));
    }
    tiling.set_nSize(nSize);
    tiling.set_cSize(cSize);
    tiling.set_hSize(hSize);
    tiling.set_wSize(wSize);
    context->SetBlockDim(tiling.get_usedCoreNum());
    context->SetTilingKey(tiling.get_tilingKey());

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class DepthToSpace : public OpDef {
public:
    explicit DepthToSpace(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("block_size").Int();
        this->Attr("mode").String();
        this->Attr("data_format").String();
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(DepthToSpace);
}
