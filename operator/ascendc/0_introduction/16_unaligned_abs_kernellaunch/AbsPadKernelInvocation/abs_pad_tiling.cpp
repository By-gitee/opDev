#include "graph/tensor.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

void GenerateTiling(const std::vector<int64_t> shapePad, const std::vector<int64_t> shapeUsed, uint8_t *tilingBuf)
{
    ge::Shape srcShape(shapePad);
    ge::Shape oriSrcShape(shapeUsed);
    uint32_t tmpMinSize, tmpMaxSize;
    AscendC::GetPadMaxMinTmpSize(srcShape, sizeof(int16_t), tmpMaxSize, tmpMinSize);
    optiling::PadTiling tilingData;
    AscendC::PadTilingFunc(srcShape, oriSrcShape, tmpMaxSize, sizeof(int16_t), tilingData);
    uint32_t tilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tilingSize);
    return;
}
