/**
 * @file adds_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "../op_kernel/adds_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    constexpr uint32_t BLOCK_DIM = 16;
    context->SetBlockDim(BLOCK_DIM);

    // set tiling_key
    auto attrs = context->GetAttrs();
    const int64_t *caseId = attrs->GetInt(0);
    context->SetTilingKey(*caseId);

    AddsCustomTilingData *tiling = context->GetTilingData<AddsCustomTilingData>();
    constexpr uint32_t M = 8192;
    constexpr uint32_t N = 128;
    constexpr uint32_t TILE_M = 512;
    constexpr uint32_t TILE_N = 8;
    constexpr uint32_t LOOP_ONE_CORE = M / TILE_M;
    tiling->m = M;
    tiling->n = N;
    tiling->tileM = TILE_M;
    tiling->tileN = TILE_N;
    tiling->loopOneCore = LOOP_ONE_CORE;

    // set workspace size
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class AddsCustom : public OpDef {
public:
    explicit AddsCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        this->Attr("case_id").Int(1);
    }
};
OP_ADD(AddsCustom);
} // namespace ops
