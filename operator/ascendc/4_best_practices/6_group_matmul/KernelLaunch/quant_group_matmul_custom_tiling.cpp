/**
 * @file quant_group_matmul_custom_tiling.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "quant_group_matmul_custom_tiling.h"

using matmul_tiling::TPosition;
using matmul_tiling::CubeFormat;
using matmul_tiling::DataType;

constexpr uint32_t BEST_BASE_M = 128;
constexpr uint32_t BEST_BASE_K = 128;
constexpr uint32_t BEST_BASE_N = 256;

bool GenerateTiling(QuantGroupMatmulCustomTilingData &gmmTiling)
{
    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetDim(1);
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::NZ, DataType::DT_INT8, false);
    tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
    tilingApi.SetBias(false);

    tilingApi.SetOrgShape(BEST_BASE_M, gmmTiling.n, gmmTiling.k);
    tilingApi.SetShape(BEST_BASE_M, gmmTiling.n, gmmTiling.k);
    tilingApi.SetFixSplit(BEST_BASE_M, BEST_BASE_N, BEST_BASE_K);

    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
        return false;
    }
    tilingData.set_dbL0C(1);
    tilingData.set_stepKa(4);  // 4: L1中左矩阵单次搬运基于baseK的4倍数据
    tilingData.set_stepKb(4);  // 4: L1中右矩阵单次搬运基于baseK的4倍数据
    tilingData.set_depthA1(8);  // 8: stepKa的两倍，开启double buffer
    tilingData.set_depthB1(8);  // 8: stepKb的两倍，开启double buffer
    tilingData.set_stepM(1);
    tilingData.set_stepN(1);
    
    uint32_t tilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(&gmmTiling.mmTilingData, tilingSize);
    return true;
}
