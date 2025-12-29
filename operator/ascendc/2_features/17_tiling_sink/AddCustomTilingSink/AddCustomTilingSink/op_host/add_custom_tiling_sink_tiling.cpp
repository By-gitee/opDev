/**
 * @file add_custom_tiling_sink_tiling.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "add_custom_tiling_sink_tiling.h"
#include "register/device_op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static constexpr uint32_t BLOCK_DIM = 8;
static constexpr uint32_t TILE_NUM = 3;
static constexpr size_t MAX_WORKSPACE_SIZE = 32; // 算子所需用户workspace空间最大值，AddCustomTilingSink算子本身逻辑无需用户workspace空间，此处设置为固定值仅作为示例
static constexpr size_t DEFAULT_WORKSPACE_SIZE = 0;
ge::graphStatus AddCustomSinkTilingFunc(gert::TilingContext *context)
{
    TilingSinkTilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = sysWorkspaceSize + DEFAULT_WORKSPACE_SIZE; // 设置运行时workspace大小，此处为系统workspace空间 + 用户workspace空间
    if (context->GetInputTensor(1) != nullptr && context->GetInputTensor(1)->GetData<float>() == nullptr) {
        // 通过判断值依赖InputTensor的Data是否为空指针来确认当前是否处于编译期。
        // Tiling下沉场景，编译期需要为算子分配内存，包括其所需的workspace。为了保证运行时的高效性，编译期应根据算子的执行需求，合理设置所需的workspace最大值，以避免内存不足或浪费。
        currentWorkspace[0] = sysWorkspaceSize + MAX_WORKSPACE_SIZE; // 设置编译期workspace大小，此处为系统workspace空间 + 用户workspace空间最大值
    }
    return ge::GRAPH_SUCCESS;
}
DEVICE_IMPL_OP_OPTILING(AddCustomTilingSink).Tiling(optiling::AddCustomSinkTilingFunc); // 下沉tiling函数注册
} // namespace optiling