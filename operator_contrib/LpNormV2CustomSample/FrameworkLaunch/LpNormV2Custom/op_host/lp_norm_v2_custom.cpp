
#include "lp_norm_v2_custom_tiling.h"
#include <cmath>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 64;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  LpNormV2CustomTilingData tiling;
  uint32_t sizeofdatatype;
  uint32_t totalLengthAligned;
  auto ascendcPlatform = 
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto socVersion = ascendcPlatform.GetSocVersion();
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  auto aivNum = ascendcPlatform.GetCoreNumAiv();

  float p = *(context->GetAttrs()->GetFloat(0));
  float epsilon = std::numeric_limits<float>::epsilon();
  uint32_t pType = 0;
  tiling.set_pValue(p);
  if (std::fabs(p - 2.0f) < epsilon)
  {
    tiling.set_pType(0);
  }
  else if (std::isinf(p))
  {
    if (p > 0)
    {
      tiling.set_pType(1);
      pType = 1;
    }
    else
    {
      tiling.set_pType(2);
      pType = 2;
    }
  }
  else if (std::fabs(p) < epsilon)
  {
    tiling.set_pType(3);
    pType = 3;
  }
  else if (std::fabs(p - 1.0f) < epsilon)
  {
    tiling.set_pType(4);
    pType = 4;
  }
  else
  {
    tiling.set_pType(5);
    pType = 5;
  }
  const gert::TypedContinuousVector<int64_t> *axes = 
                        context->GetAttrs()->GetListInt(1);
  uint32_t totalLength = 1;
  uint32_t stepSize = 1;
  uint32_t unitCount = 1;
  auto axesDimNum = axes->GetSize();
  if (axesDimNum != 0)
  {
    // 处理多轴情况，axes为空时，不需要计算
    // 仅处理了axes列表中为连续维度的情况，例如[0], [1, 2]之类
    // 因为是连续维度，所以定位通过axes列表的值，将shape中的维度分为三部分
    // 前部分（unitCount），中间部分(stepSize)，后部分(totalLength)
    auto shape = context->GetInputShape(0)->GetOriginShape();
    uint32_t dimNum = shape.GetDimNum();
    uint32_t maxdim = 0;
    uint32_t mindim = dimNum;
    for (int i = 0; i < axesDimNum; i++)
    {
      auto index = *(axes->GetData() + i);
      if (index < mindim)
        mindim = index;
      if (index > maxdim)
        maxdim = index;
    }
    if (maxdim > dimNum)
    {
      totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
      stepSize = 1;
      unitCount = 1;
    }
    else
    {
      for (int i = 0; i < dimNum; i++)
      {
        if (i < mindim)
        {
          unitCount *= shape.GetDim(i);
        }
        else if (i > maxdim)
        {
          stepSize *= shape.GetDim(i);
        }
        else
        {
          totalLength *= shape.GetDim(i);
        }
      }
    }
  }
  else
  {
    stepSize = 1;
    unitCount = 1;
    totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  }
  tiling.set_totalLength(totalLength);
  tiling.set_stepSize(stepSize);
  tiling.set_unitCount(unitCount);
  auto dt = context->GetInputDesc(0)->GetDataType();
  uint32_t typeKey = 0;
  uint32_t dataNum = 0;
  // 不同类型下一次tile处理中，ub中最多存储的tileLength的倍数
  // 即一次tile处理中长度为tileLength的数据块个数
  if (dt == 1)
  {
    sizeofdatatype = 2;
    dataNum = 6;
    tiling.set_typeKey(1);
  }
  else if (dt == 0)
  {
    sizeofdatatype = 4;
    dataNum = 3;
    tiling.set_typeKey(0);
  }

  uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
  uint32_t ub_block_num =
      ((ub_size) / BLOCK_SIZE / dataNum) * 13 / 20;
  uint32_t tile_num;
  if (pType == 3)
  {
    ub_block_num = 8;
  }
  if (ub_block_num % 2 != 0)
  {
    ub_block_num = ub_block_num - 1;
  }

  if (totalLength % ALIGN_NUM != 0)
  {
    totalLengthAligned =
        ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
  }
  else
  {
    totalLengthAligned = totalLength;
  }
  context->SetBlockDim(1);
  uint32_t tileLength = 0;
  uint32_t lasttileLength = 0;
  tile_num = totalLengthAligned / ALIGN_NUM / ub_block_num;
  if ((totalLengthAligned / ALIGN_NUM) % ub_block_num == 0 ||
      tile_num == 0)
  {
    if (tile_num == 0)
    {
      tile_num = 1;
    }
    if (totalLengthAligned < ub_block_num * ALIGN_NUM)
    {
      tileLength = totalLengthAligned;
      lasttileLength = tileLength;
    }
    else
    {
      tileLength = ub_block_num * ALIGN_NUM;
      lasttileLength = tileLength;
    }
  }
  else
  {
    tile_num = tile_num + 1;
    tileLength = ub_block_num * ALIGN_NUM;
    lasttileLength = totalLengthAligned - (tile_num - 1) * tileLength;
  }
  if (totalLength % tileLength)
  {
    lasttileLength = totalLength % tileLength;
  }
  tiling.set_blockLength(totalLengthAligned);
  tiling.set_tileNum(tile_num);
  tiling.set_tileLength(tileLength);
  tiling.set_lasttileLength(lasttileLength);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
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
class LpNormV2Custom : public OpDef {
public:
  explicit LpNormV2Custom(const char* name) : OpDef(name)
  {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("p").AttrType(OPTIONAL).Float(2);
    this->Attr("axes").AttrType(OPTIONAL).ListInt({});
    this->Attr("keepdim").AttrType(OPTIONAL).Bool(false);
    this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-12);

    this->SetInferShape(ge::InferShape);

    this->AICore()
        .SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend310b")
                  .AddConfig("ascend910b");
  }
};

OP_ADD(LpNormV2Custom);
}
