#include "kernel_operator.h"
#include "flash_attention_score_s1s2_bn2gs1.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void flash_attention_score(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR attention_out, GM_ADDR workspace, GM_ADDR tiling) {
    
    TPipe tPipe;
    set_mask_norm();    
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreTilingData, tilingDataIn, tiling);
    const FlashAttentionScoreTilingData *__restrict tilingData = &tilingDataIn;
    const TCubeTiling *__restrict bmm1tiling = &(tilingData->bmm1TilingData);
    const TCubeTiling *__restrict bmm2tiling = &(tilingData->bmm2TilingData);
    FlashAttentionScoreS1s2Bn2gs1 op;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);
    op.Init(query, key, value, softmax_max, softmax_sum, attention_out, user, tilingData, &tPipe);
    op.Process();
    // TODO: user kernel impl
}