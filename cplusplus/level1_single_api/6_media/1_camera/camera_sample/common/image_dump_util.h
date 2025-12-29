/**
* @File image_dump_util.h
* @Description Provide util function for reading RAW & YUV data from the VI PIPE & CHN
*
* Copyright (c) Huawei Technologies Co., Ltd. 2016-2021. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef IMAGE_DUMP_UTIL_H
#define IMAGE_DUMP_UTIL_H

#include "sample_comm.h"

hi_s32 DumpYuv(hi_vi_pipe Pipe, hi_vi_chn Chn, hi_u32 u32FrameCnt,
    hi_u32 u32ByteAlign, hi_video_frame_info stFrame);
hi_s32 DumpLinearBayer(hi_vi_pipe ViPipe, hi_u32 u32Nbit, hi_compress_mode enCompressMode,
    hi_u32 u32Cnt, hi_u32 u32ByteAlign, hi_u32 u32RatioShow);
#endif // IMAGE_DUMP_UTIL_H