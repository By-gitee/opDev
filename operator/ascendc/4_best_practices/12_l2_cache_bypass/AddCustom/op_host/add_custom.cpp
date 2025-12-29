/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "../op_kernel/add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    constexpr uint32_t BLOCK_DIM = 40;
    context->SetBlockDim(BLOCK_DIM);

    // set tiling_key
    auto attrs = context->GetAttrs();
    const int64_t *caseId = attrs->GetInt(0);
    context->SetTilingKey(*caseId);

    AddCustomTilingData *tiling = context->GetTilingData<AddCustomTilingData>();
    // x shape is [5120, 5120], y shape is [5120, 15360], so we set outer loop to 3
    tiling->loopOuter = 3U;

    // set workspace size
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        this->Attr("case_id").Int(1);
    }
};
OP_ADD(AddCustom);
} // namespace ops
