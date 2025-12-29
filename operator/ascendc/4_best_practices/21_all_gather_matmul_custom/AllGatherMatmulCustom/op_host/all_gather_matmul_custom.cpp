/**
 * @file all_gather_matmul_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "../op_kernel/all_gather_matmul_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "tiling/hccl/hccl_tiling.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

// tiling
namespace {
constexpr int32_t TILE_M = 448;
constexpr uint32_t HCCL_CMD_ALLGATHER = 6;
constexpr uint32_t HCCL_REDUCE_SUM = 0;
constexpr int32_t L1_BUFFER_SIZE = 512 * 1024;
}

static ge::graphStatus AllGatherMatmulCustomTilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicCoreNum = ascendcPlatform.GetCoreNumAic();

    // get attrs
    const char *group = context->GetAttrs()->GetAttrPointer<char>(0);
    INFO_LOG("Group %s", group);

    // get shape  [[4096/8,5120], [5120,640]] fp16
    uint64_t rankM = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    uint64_t rankK = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    uint64_t rankN = context->GetInputShape(1)->GetStorageShape().GetDim(1);
    INFO_LOG("RankM %lu, rankK %lu, rankN %lu", rankM, rankK, rankN);

    // get dtype
    auto aType = context->GetInputTensor(0)->GetDataType();
    auto bType = context->GetInputTensor(1)->GetDataType();
    auto cType = aType;
    if (aType != ge::DT_FLOAT16 || bType != ge::DT_FLOAT16) {
        ERROR_LOG("Dtype is unsupported");
        return ge::GRAPH_FAILED;
    }

    // set block dim
    context->SetBlockDim(ascendcPlatform.GetCoreNumAic());

    // set work space size
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *workspaceSizes = context->GetWorkspaceSizes(1); // 获取设置workspace大小的指针。
    workspaceSizes[0] = systemWorkspaceSize;

    uint64_t tileNum = rankM / TILE_M;
    uint64_t tailNum = (rankM % TILE_M == 0) ? 0 : 1;
    uint64_t tailM = rankM % TILE_M;
    INFO_LOG("tileNum %lu, tailNum %lu, tailM %lu", tileNum, tailNum, tailM);

    AllGatherMatmulCustomTilingData *tiling = context->GetTilingData<AllGatherMatmulCustomTilingData>();

    tiling->cfg.tileNum = tileNum;
    tiling->cfg.tailM = tailM;
    tiling->cfg.tailNum = tailNum;
    tiling->cfg.rankM = rankM;
    tiling->cfg.rankN = rankN;
    tiling->cfg.rankK = rankK;
    // matmul tiling func
    auto matmulTilingFunc = [&](int64_t m, int64_t n, int64_t k, TCubeTiling &cubeTiling) -> bool {
        matmul_tiling::MultiCoreMatmulTiling mmTiling;
        mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
        mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
        mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
        mmTiling.SetBias(false);
        mmTiling.SetDim(aicCoreNum);
        mmTiling.SetShape(m, n, k);
        mmTiling.SetOrgShape(m, n, k);
        mmTiling.SetBufferSpace(L1_BUFFER_SIZE);
        if (mmTiling.GetTiling(cubeTiling) != 0) {
            return false;
        }
        return true;
    };
    // matmul local tiling
    if (!matmulTilingFunc(rankM, rankN, rankK, tiling->localTiling)) {
        ERROR_LOG("Get local matmul tiling failed");
        return ge::GRAPH_FAILED;
    }
    // matmul tile tiling
    if (!matmulTilingFunc(TILE_M, rankN, rankK, tiling->tileTiling)) {
        ERROR_LOG("Get tile matmul tiling failed");
        return ge::GRAPH_FAILED;
    }
    // matmul tail tiling
    if (!matmulTilingFunc(rankM % TILE_M, rankN, rankK, tiling->tailTiling)) {
        ERROR_LOG("Get tail matmul tiling failed");
        return ge::GRAPH_FAILED;
    }

    uint32_t opType = HCCL_CMD_ALLGATHER;
    std::string algConfig = "AllGather=level0:doublering";
    uint32_t reduceType = HCCL_REDUCE_SUM;
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig, reduceType);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.SetSkipLocalRankCopy(0);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
    
    return ge::GRAPH_SUCCESS;
}

namespace ops {
class AllGatherMatmulCustom : public OpDef {
public:
    explicit AllGatherMatmulCustom(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .IgnoreContiguous();

        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("gather_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("group").AttrType(REQUIRED).String();

        this->AICore().SetTiling(AllGatherMatmulCustomTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->MC2().HcclGroup("group");
    }
};

OP_ADD(AllGatherMatmulCustom);
}
