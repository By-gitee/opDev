/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#include "mse_loss_tiling.h"
#include "register/op_def_registry.h"
#include <string.h>

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MseLossTilingData tiling;
    uint32_t sizeofdatatype;
    uint32_t totalLengthAligned;
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    auto dt = context->GetInputDesc(0)->GetDataType();
    if (dt == 1) {
      sizeofdatatype = 2;
    }

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t ub_block_num = 1024;  
    uint32_t tile_num;

    if (ub_block_num % 2 != 0) {
      ub_block_num = ub_block_num - 1;
    }

    // 获取reduction的值，并设置传入kernel的mode值
    const char* reduction = context->GetAttrs()->GetStr(0);
    const char* mode1 = "mean";
    const char* mode2 = "sum";
    const char* mode3 = "none";
    int str_len = strlen(reduction);
    int mode = 0;
    
    if (str_len == strlen(mode1)) {
        for (int i = 0; i < str_len; i++) {
            if (reduction[i] != mode1[i]) {
                break;
            }
            if (i == str_len-1) {
                mode = 1;
            }
        }
    }
    if (str_len == strlen(mode2)) {
        for (int i = 0; i < str_len; i++) {
            if (reduction[i] != mode2[i]) {
                break;
            }
            if (i == str_len-1) {
                mode = 2;
            }
        }
    }
    if (str_len == strlen(mode3)) {
        for (int i = 0; i < str_len; i++) {
            if (reduction[i] != mode3[i]) {
                break;
            }
            if (i == str_len-1) {
                mode = 3;
            }
        }
    }

    tiling.set_mode(mode);

    if (totalLength % ALIGN_NUM != 0) {  
        totalLengthAligned =
            ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    } else {
        totalLengthAligned = totalLength;
    }

    tiling.set_totalLength(totalLength);

    // 环境为单核环境，故直接设置为1个核
    context->SetBlockDim(1);

    auto block_dim = context->GetBlockDim();
    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
    uint32_t lasttileLength = 0;

    blockLength = totalLengthAligned / block_dim;
    tile_num = blockLength / ALIGN_NUM / ub_block_num;

    // 数据切分策略： 由于为单核环境，则将tileLength设置得尽可能大，最后单独处理剩余数据
    if ((totalLengthAligned / block_dim / ALIGN_NUM) % ub_block_num == 0 || tile_num == 0) {  
        if (tile_num == 0) {
            tile_num = 1;
        } 
        if (blockLength < ub_block_num * ALIGN_NUM) {
            tileLength = ((blockLength / ALIGN_NUM) + 1) / 2 * 2 * ALIGN_NUM;
            lasttileLength = tileLength;
        } 
        else {
            tileLength = ub_block_num * ALIGN_NUM;
            lasttileLength = tileLength;
        }
    } 
    else {  
        tile_num = tile_num + 1;
        tileLength = ub_block_num * ALIGN_NUM;
        lasttileLength = blockLength - (tile_num - 1) * tileLength;
    }

    tiling.set_blockLength(blockLength);
    tiling.set_tileNum(tile_num);
    tiling.set_tileLength(tileLength);
    tiling.set_lasttileLength(lasttileLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
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
class MseLoss : public OpDef {
public:
    explicit MseLoss(const char* name) : OpDef(name)
    {
        this->Input("predict")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("label")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b")
                      .AddConfig("ascend910b");

    }
};

OP_ADD(MseLoss);
}
