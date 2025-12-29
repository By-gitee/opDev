#include "quant_matmul_custom_tiling.h"
#include <iostream>
#include <cmath>

namespace optiling {
#define CHECK_TRUE_CONTINUE(expr) \
    {                             \
        if (expr) {               \
            continue;             \
        }                         \
    }

#define CEIL_ALIGN(N, n) (((N) + (n)-1) / (n) * (n))
#define FLOOR_ALIGN(N, n) ((N) / (n) * (n))
#define CEIL(N, n) (((N) + (n)-1) / (n))

constexpr uint32_t PINGPONG = 2;
constexpr uint32_t MIN_MNK_BASE_SIZE = 16;
constexpr uint32_t S16_4_C0 = 64;
constexpr uint32_t ELE_Size = sizeof(uint16_t);
constexpr uint32_t BLOCK_SIZE = 32;
constexpr int64_t WEIGHT_BW_FULL_BYTES = 16 * 1024;
QuantMatMulTiling::QuantMatMulTiling(
        const platform_ascendc::PlatformAscendC &ascendcPlatform, const MatMulArgs &commonArgs)
{
    uint32_t calEqualM = commonArgs.origM;
    if (commonArgs.origBatchB == 1) {
        calEqualM = calEqualM * commonArgs.origBatchA;
    }
    matmulCalPara_.origM = calEqualM;
    matmulCalPara_.origN = commonArgs.origN;
    matmulCalPara_.origK = commonArgs.origK;

    matmulCalPara_.blockDim = 1;
    matmulCalPara_.singleCoreM = calEqualM;
    matmulCalPara_.singleCoreN = commonArgs.origN;
    matmulCalPara_.singleCoreK = commonArgs.origK;

    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, matmulCalPara_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, matmulCalPara_.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, matmulCalPara_.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, matmulCalPara_.l0BSize);

    printf("QuantMatMulTiling soc[%u], l1Size[%lu], l0CSize[%lu], l0ASize[%lu], l0BSize[%lu]\n",
        (uint32_t)socVersion,
        matmulCalPara_.l1Size,
        matmulCalPara_.l0CSize,
        matmulCalPara_.l0ASize,
        matmulCalPara_.l0BSize);
}

uint32_t QuantMatMulTiling::GetL0Tiling(L0TilingData &l0TilingData)
{
    int64_t minBase = MIN_MNK_BASE_SIZE;
    int64_t cutM = minBase;
    int64_t cutN = minBase;
    int64_t cutK = minBase;
    int64_t localCutNLimit = 128;
    uint64_t scoreMin = UINT64_MAX;
    for (int64_t cutMTemp = minBase; cutMTemp <= CEIL_ALIGN(matmulCalPara_.singleCoreM, minBase); cutMTemp += minBase) {
        // cal cutN
        int64_t cutNTemp = matmulCalPara_.l0CSize / ELE_Size / PINGPONG / cutMTemp;
        cutNTemp = std::min(cutNTemp, CEIL_ALIGN(matmulCalPara_.singleCoreN, minBase));
        cutNTemp = cutNTemp > localCutNLimit ? localCutNLimit : cutNTemp; // fix L0 N=128
        cutNTemp = FLOOR_ALIGN(cutNTemp, minBase);
        CHECK_TRUE_CONTINUE(cutNTemp == 0);

        // cal cutK
        int64_t cutAKTemp = matmulCalPara_.l0ASize / ELE_Size / PINGPONG / cutMTemp;
        cutAKTemp = std::min(cutAKTemp, CEIL_ALIGN(matmulCalPara_.singleCoreK, minBase));
        int64_t cutBKTemp = matmulCalPara_.l0BSize / ELE_Size / PINGPONG / cutNTemp;
        cutBKTemp = std::min(cutBKTemp, CEIL_ALIGN(matmulCalPara_.singleCoreK, minBase));
        int64_t cutKTemp = std::min(cutAKTemp, cutBKTemp);
        cutKTemp = FLOOR_ALIGN(cutKTemp, minBase * 2);
        CHECK_TRUE_CONTINUE(cutKTemp < S16_4_C0 && S16_4_C0 <= matmulCalPara_.singleCoreK);
        CHECK_TRUE_CONTINUE(cutMTemp == 0 || cutNTemp == 0 || cutKTemp == 0);

        uint64_t loopM = static_cast<uint64_t>(ceil(static_cast<double>(1.0 * matmulCalPara_.singleCoreM / cutMTemp)));
        uint64_t loopN = static_cast<uint64_t>(ceil(static_cast<double>(1.0 * matmulCalPara_.singleCoreN / cutNTemp)));
        uint64_t loopK = static_cast<uint64_t>(ceil(static_cast<double>(1.0 * matmulCalPara_.singleCoreK / cutKTemp)));
        uint64_t loopAll = loopM * loopN * loopK;
        uint64_t loopMN = loopM * loopN;
        uint64_t score = (loopMN << 32) | loopAll;

        if (score < scoreMin) {
            cutM = cutMTemp;
            cutN = cutNTemp;
            cutK = cutKTemp;
            scoreMin = score;
        }
    }
    l0TilingData.fracM = cutM;
    l0TilingData.fracN = cutN;
    l0TilingData.fracK = cutK;

    return ge::GRAPH_SUCCESS;
}
uint32_t QuantMatMulTiling::GetL1Tiling(const L0TilingData &l0TilingData, MatMulTilingData &tilingData)
{
    int64_t m = matmulCalPara_.singleCoreM;
    int64_t k = CEIL_ALIGN(matmulCalPara_.singleCoreK, MIN_MNK_BASE_SIZE);
    int64_t aSize = CEIL_ALIGN(m * k * ELE_Size, BLOCK_SIZE);

    int64_t l0FracK = l0TilingData.fracK;
    int64_t l0FracN = l0TilingData.fracN;
    int64_t l0FracM = l0TilingData.fracM;

    int64_t cutK = 0;
    int64_t bEleSize = ELE_Size;
    for (int64_t cutKTemp = l0FracK; cutKTemp <= CEIL_ALIGN(matmulCalPara_.singleCoreK, l0FracK); cutKTemp += l0FracK) {
        int64_t bL1PingUseSize = CEIL_ALIGN(cutKTemp * l0FracN * bEleSize, BLOCK_SIZE);
        bL1PingUseSize = CEIL_ALIGN(bL1PingUseSize, BLOCK_SIZE);

        int64_t useL1Size = aSize + bL1PingUseSize * PINGPONG;
        if ((uint64_t)useL1Size <= matmulCalPara_.l1Size) {
            if (bL1PingUseSize >= WEIGHT_BW_FULL_BYTES) {
                cutK = cutKTemp;
                break;
            }
            cutK = cutKTemp;
        }
    }

    if (cutK == 0) {
        printf("AFull tiling search failed.\n");
        return ge::GRAPH_FAILED;
    }

    tilingData.cutM = matmulCalPara_.singleCoreM;
    tilingData.cutSubM = l0FracM;
    tilingData.cutN = l0FracN;
    tilingData.cutSubN = l0FracN;
    tilingData.cutK = cutK;
    tilingData.cutSubK = l0FracK;

    tilingData.stepM = CEIL(tilingData.cutM, tilingData.cutSubM);
    tilingData.stepN = CEIL(tilingData.cutN, tilingData.cutSubN);
    tilingData.stepKa = CEIL(matmulCalPara_.singleCoreK, tilingData.cutSubK);
    tilingData.stepKb = CEIL(tilingData.cutK, tilingData.cutSubK);
    tilingData.depthA1 = tilingData.stepM * tilingData.stepKa;
    tilingData.depthB1 = tilingData.stepN * tilingData.stepKb * PINGPONG;
    tilingData.stepSubN = 1;
    tilingData.iterateOrder = 1;

    return ge::GRAPH_SUCCESS;
}
void QuantMatMulTiling::SetOutTilingRes(MatMulTilingData tilingData, optiling::TCubeTiling &tilingOut)
{
    tilingOut.set_usedCoreNum(matmulCalPara_.blockDim);
    tilingOut.set_M(matmulCalPara_.origM);
    tilingOut.set_N(matmulCalPara_.origN);
    tilingOut.set_Ka(matmulCalPara_.origK);
    tilingOut.set_Kb(matmulCalPara_.origK);

    tilingOut.set_singleCoreM(matmulCalPara_.singleCoreM);
    tilingOut.set_singleCoreN(matmulCalPara_.singleCoreN);
    tilingOut.set_singleCoreK(matmulCalPara_.singleCoreK);

    tilingOut.set_baseM(tilingData.cutSubM);
    tilingOut.set_baseN(tilingData.cutSubN);
    tilingOut.set_baseK(tilingData.cutSubK);

    tilingOut.set_depthA1(tilingData.depthA1);
    tilingOut.set_depthB1(tilingData.depthB1);

    tilingOut.set_stepM(tilingData.stepM);
    tilingOut.set_stepN(tilingData.stepN);
    tilingOut.set_isBias(false);
    tilingOut.set_iterateOrder(tilingData.iterateOrder);
    tilingOut.set_stepKa(tilingData.stepKa);
    tilingOut.set_stepKb(tilingData.stepKb);

    constexpr int32_t DB_ON = 2;
    constexpr int32_t DB_OFF = 1;
    int32_t dbSwitch = (tilingData.stepKa == 1 && tilingData.stepKb == 1) ? DB_OFF : DB_ON;

    tilingOut.set_dbL0A(dbSwitch);
    tilingOut.set_dbL0B(dbSwitch);
    tilingOut.set_dbL0C(DB_ON);

    tilingOut.set_ALayoutInfoB(tilingData.stepSubN);

    return;
}
void QuantMatMulTiling::PrintTilingData(optiling::TCubeTiling &tilingOut)
{
    std::cout << "TCubeTiling CubeTilingData: " << std::endl;
    std::cout << "tiling.usedCoreNum = " << tilingOut.get_usedCoreNum() << std::endl;
}

uint32_t QuantMatMulTiling::GetTiling(optiling::TCubeTiling &tilingOut)
{
    L0TilingData l0TilingOut;
    if (GetL0Tiling(l0TilingOut) != ge::GRAPH_SUCCESS) {
        printf("L0 tiling get failed.\n");
        return ge::GRAPH_FAILED;
    }

    MatMulTilingData tilingData;
    if (GetL1Tiling(l0TilingOut, tilingData) != ge::GRAPH_SUCCESS) {
        printf("L1 tiling get failed.\n");
        return ge::GRAPH_FAILED;
    }

    SetOutTilingRes(tilingData, tilingOut);
    PrintTilingData(tilingOut);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
