
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeSoftMaxTopkTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, lastDim);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, k);
  TILING_DATA_FIELD_DEF(int32_t, scoreSum);
  TILING_DATA_FIELD_DEF(int32_t, indicesSum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeSoftMaxTopk, MoeSoftMaxTopkTilingData)
}
