#ifndef QUANT_MATMUL_CUSTOM_TILING_H
#define QUANT_MATMUL_CUSTOM_TILING_H

#include "register/tilingdata_base.h"
#include "lib/matmul/matmul_tilingdata.h"
#include "lib/matmul/bmm_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CustomTilingInfo)
TILING_DATA_FIELD_DEF(uint32_t, tileN);
TILING_DATA_FIELD_DEF(uint32_t, tileK);
TILING_DATA_FIELD_DEF(uint32_t, blockA);
TILING_DATA_FIELD_DEF(uint32_t, blockM);
TILING_DATA_FIELD_DEF(uint32_t, blockKa);
TILING_DATA_FIELD_DEF(uint32_t, blockKb);
TILING_DATA_FIELD_DEF(uint32_t, blockN);
TILING_DATA_FIELD_DEF(uint32_t, offsetK);
TILING_DATA_FIELD_DEF(uint32_t, offsetN);
TILING_DATA_FIELD_DEF(uint32_t, weightK);
TILING_DATA_FIELD_DEF(uint32_t, weightN);
TILING_DATA_FIELD_DEF(uint32_t, blockWeightK);
TILING_DATA_FIELD_DEF(uint32_t, blockWeightN);
TILING_DATA_FIELD_DEF(uint32_t, blockWeight);
TILING_DATA_FIELD_DEF(uint32_t, blockOffsetK);
TILING_DATA_FIELD_DEF(uint32_t, blockOffsetN);
TILING_DATA_FIELD_DEF(uint32_t, blockOffset);
TILING_DATA_FIELD_DEF(uint32_t, blockWeightAndOffset);
TILING_DATA_FIELD_DEF(uint32_t, sizeBlockAntiqIn);
TILING_DATA_FIELD_DEF(uint32_t, sizeBlockAntiqOut);
TILING_DATA_FIELD_DEF(uint32_t, sizeAExpand);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CustomTilingInfoOp, CustomTilingInfo)

BEGIN_TILING_DATA_DEF(QuantMatmulTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
TILING_DATA_FIELD_DEF_STRUCT(CustomTilingInfo, customTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantMatmulCustom, QuantMatmulTilingData)

struct MatMulArgs {
    ge::DataType aType = ge::DT_FLOAT16;
    ge::DataType bType = ge::DT_FLOAT16;
    ge::DataType cType = ge::DT_FLOAT16;
    uint64_t origBatchA = 0L;
    uint64_t origBatchB = 0L;
    uint64_t origM = 0L;
    uint64_t origN = 0L;
    uint64_t origK = 0L;
};

struct L0TilingData {
    int64_t fracM;
    int64_t fracN;
    int64_t fracK;
};

struct MatMulTilingData {
    int64_t cutM;
    int64_t cutSubM;
    int64_t cutN;
    int64_t cutSubN;
    int64_t cutK;
    int64_t cutSubK;
    int64_t depthA1;
    int64_t depthB1;
    int64_t stepM;
    int64_t stepN;
    int64_t stepKa;
    int64_t stepKb;
    int64_t stepSubN;
    int64_t iterateOrder;
};

struct MatMulCalParam {
    int64_t blockDim;
    int64_t origM;
    int64_t origK;
    int64_t origN;
    int64_t singleCoreM;
    int64_t singleCoreK;
    int64_t singleCoreN;
    uint64_t l1Size = 0;
    uint64_t l0CSize = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
};

class QuantMatMulTiling {
public:
    explicit QuantMatMulTiling(
            const platform_ascendc::PlatformAscendC &ascendcPlatform, const MatMulArgs &commonArgs);
    ~QuantMatMulTiling()
    {}

    uint32_t GetTiling(optiling::TCubeTiling &tilingOut);

private:
    uint32_t GetL0Tiling(L0TilingData &l0TilingData);
    uint32_t GetL1Tiling(const L0TilingData &l0TilingData, MatMulTilingData &tilingData);
    void SetOutTilingRes(MatMulTilingData tilingData, optiling::TCubeTiling &tilingOut);
    void PrintTilingData(optiling::TCubeTiling &tilingOut);

private:
    MatMulCalParam matmulCalPara_;
};
} //namespace optiling

#endif