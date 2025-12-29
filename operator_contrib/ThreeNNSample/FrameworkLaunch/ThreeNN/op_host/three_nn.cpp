/**
* @file three_nn.cpp
*
* Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <algorithm>
#include "three_nn_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
const uint32_t BLOCK_SIZE = 32*3*4;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ThreeNNTilingData tiling;

    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t shapeB = context->GetInputTensor(0)->GetOriginShape().GetDim(0);
    uint32_t shapeN = context->GetInputTensor(0)->GetOriginShape().GetDim(1);
    uint32_t shapeM = context->GetInputTensor(1)->GetOriginShape().GetDim(1);

    uint32_t inputNum = context->GetInputShape(1)->GetStorageShape().GetShapeSize()/shapeB;
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(1)->GetDataType());
    uint32_t inputLength = inputBytes * inputNum;

    uint32_t ubDataNumber = 4;
 
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    uint32_t tileDataNum = 1440;

    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE;

    uint32_t TileNum = inputNum / tileDataNum;
    uint32_t finalTileNum = (inputNum % tileDataNum) == 0 ? TileNum : TileNum + 1;
    
    uint32_t TailDataNum = inputNum  - (tileDataNum * TileNum);
    TailDataNum = TailDataNum == 0 ? tileDataNum : TailDataNum;

    tiling.set_CoreDataNum((inputNum*shapeB+31)/32*32); 
    tiling.set_finalTileNum(finalTileNum);
    tiling.set_tileDataNum(tileDataNum); 
    tiling.set_TailDataNum(TailDataNum); 
    tiling.set_shapeB(shapeB);
    tiling.set_shapeN(shapeN); 
    tiling.set_shapeM(shapeM); 
    
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class ThreeNN : public OpDef {
public:
    explicit ThreeNN(const char* name) : OpDef(name)
    {
        this->Input("xyz1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("xyz2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dist")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(ThreeNN);
}
