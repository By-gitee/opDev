#include "graph/tensor.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

void GenerateTiling(const std::vector<int64_t> shape, const char *socVersion, uint8_t *tilingBuf)
{
    platform_ascendc::PlatformAscendC *ascendcPlatform;
    if (socVersion != nullptr) {
        ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    }
    else{
        ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    }

    ge::Shape srcShape(shape);
    uint32_t tmpMinSize, tmpMaxSize;
    std::vector<int64_t> dstShape = srcShape.GetDims();

    AscendC::GetUnPadMaxMinTmpSize(*ascendcPlatform, srcShape, sizeof(int16_t), tmpMaxSize, tmpMinSize);
    optiling::UnPadTiling tilingData;
    AscendC::UnPadTilingFunc(srcShape, tmpMaxSize, sizeof(int16_t), tilingData);
    uint32_t tilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tilingSize);
    return;
}
