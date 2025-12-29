/**
* @File image_dump_util.c
* @Description Provide util function for reading RAW & YUV data from the VI PIPE & CHN
*
* Copyright (c) Huawei Technologies Co., Ltd. 2016-2021. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <errno.h>
#include <stdio.h>
#include "image_dump_util.h"

static void sample_yuv_8bit_dump(hi_video_frame* pVBuf, FILE* pfd, hi_video_frame_info stFrame);
static hi_s32 ConvertBitPixel(hi_u8 *pu8Data, hi_u32 u32DataNum, hi_u32 u32BitWidth, hi_u16 *pu16OutData);
static inline hi_s32 BitWidth2PixelFormat(hi_u32 u32Nbit, hi_pixel_format *penPixelFormat);

#define MAX_FRM_CNT        25
#define MAX_FRM_WIDTH      16384

static inline hi_s32 BitWidth2PixelFormat(hi_u32 u32Nbit, hi_pixel_format *penPixelFormat)
{
    hi_pixel_format enPixelFormat;
    if (8 == u32Nbit) {
        enPixelFormat = HI_PIXEL_FORMAT_RGB_BAYER_8BPP;
    } else if (10 == u32Nbit) {
        enPixelFormat = HI_PIXEL_FORMAT_RGB_BAYER_10BPP;
    } else if (12 == u32Nbit) {
        enPixelFormat = HI_PIXEL_FORMAT_RGB_BAYER_12BPP;
    } else if (14 == u32Nbit) {
        enPixelFormat = HI_PIXEL_FORMAT_RGB_BAYER_14BPP;
    } else if (16 == u32Nbit) {
        enPixelFormat = HI_PIXEL_FORMAT_RGB_BAYER_16BPP;
    } else {
        enPixelFormat = HI_PIXEL_FORMAT_RGB_BAYER_16BPP;
    }
    *penPixelFormat = enPixelFormat;
    return HI_SUCCESS;
}

hi_s32 DumpYuv(hi_vi_pipe Pipe, hi_vi_chn Chn, hi_u32 u32FrameCnt, hi_u32 u32ByteAlign, hi_video_frame_info stFrame)
{
    hi_char szYuvName[128];
    hi_char szPixFrm[10];
    hi_char szDynamicRange[10];
    hi_char szVideoFrm[10];
    hi_u32 u32Cnt = u32FrameCnt;
    hi_bool bSendToVgs = HI_FALSE;
    hi_video_frame_info stFrmInfo;
    hi_s32 s32Ret;
    FILE* pfd = HI_NULL;

    switch (stFrame.v_frame.pixel_format) {
        case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420:
            snprintf(szPixFrm, 10, "P420");
            break;
        case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_422:
            snprintf(szPixFrm, 10, "P422");
            break;
        case HI_PIXEL_FORMAT_YUV_400:
            snprintf(szPixFrm, 10, "P400");
            break;
        default:
            snprintf(szPixFrm, 10, "--");
            break;
    }

    switch (stFrame.v_frame.video_format) {
        case HI_VIDEO_FORMAT_LINEAR:
            snprintf(szVideoFrm, 10, "linear");
            break;
        case HI_VIDEO_FORMAT_TILE_64x16:
            snprintf(szVideoFrm, 10, "tile_64X16");
            break;
        case 3:
            snprintf(szVideoFrm, 10, "tile_16X8");
            break;
        default:
            snprintf(szVideoFrm, 10, "--");
            break;
    }

    switch (stFrame.v_frame.dynamic_range) {
        case HI_DYNAMIC_RANGE_SDR8:
            snprintf(szDynamicRange, 10, "SDR8");
            break;
        case HI_DYNAMIC_RANGE_SDR10:
            snprintf(szDynamicRange, 10, "SDR10");
            break;
        case HI_DYNAMIC_RANGE_HDR10:
            snprintf(szDynamicRange, 10, "HDR10");
            break;
        case HI_DYNAMIC_RANGE_XDR:
            snprintf(szDynamicRange, 10, "XDR");
            break;
        case HI_DYNAMIC_RANGE_HLG:
            snprintf(szDynamicRange, 10, "HLG");
            break;
        case HI_DYNAMIC_RANGE_SLF:
            snprintf(szDynamicRange, 10, "SLF");
            break;
        default:
            snprintf(szDynamicRange, 10, "--");
            break;
    }
    /* make file name */
    snprintf(szYuvName, 128, "./vi_pipe%d_chn%d_w%d_h%d_%s_%s_%s_%d.yuv",
        Pipe, Chn, stFrame.v_frame.width,
        stFrame.v_frame.height, szPixFrm, szVideoFrm, szDynamicRange, u32ByteAlign);
    printf("Dump YUV frame of vi chn %d  to file: \"%s\"\n", Chn, szYuvName);
    fflush(stdout);

    /* open file */
    pfd = fopen(szYuvName, "ab");

    if (HI_NULL == pfd) {
        printf("open file failed:%s!\n", strerror(errno));
        return HI_FAILURE;
    }
    /* get frame */
    while (u32Cnt--) {
        if (HI_DYNAMIC_RANGE_SDR8 == stFrame.v_frame.dynamic_range) {
            sample_yuv_8bit_dump(&stFrame.v_frame, pfd, stFrame);
        } else {
            printf("Get ViPipe %d frame %d failed! only support dump 8bit yuv\n", Pipe, u32Cnt);
        }
        printf("Get ViPipe %d frame %d!!\n", Pipe, u32Cnt);
    }
    fclose(pfd);
    return HI_SUCCESS;
}

/*When saving a file,sp420 will be denoted by p420 and sp422
    will be denoted by p422 in the name of the file */
static void sample_yuv_8bit_dump(hi_video_frame* pVBuf, FILE* pfd, hi_video_frame_info stFrame)
{
    unsigned int w, h;
    char* pVBufVirt_Y;
    char* pVBufVirt_C;
    char* pMemContent;
    // If this value is too small and the image is big, this memory may not be enough
    unsigned char TmpBuff[MAX_FRM_WIDTH];
    hi_u8* pUserPageAddr[2];
    hi_u32 u32Size = 0;

    hi_u64 phy_addr;
    hi_pixel_format  enPixelFormat = pVBuf->pixel_format;
    hi_video_format  enVideoFormat = stFrame.v_frame.video_format;
    /*When the storage format is a planar format,
        this variable is used to keep the height of the UV component */
    hi_u32 u32UvHeight = 0; 

    if (HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420 == enPixelFormat) {
        if (3 != enVideoFormat) {
            u32Size = (pVBuf->width_stride[0]) * (pVBuf->height) * 3 / 2;
            u32UvHeight = pVBuf->height / 2;
        } else {
            u32Size = (pVBuf->width_stride[0]) * (pVBuf->height);
            u32UvHeight = 0;
        }
    } else if(HI_PIXEL_FORMAT_YVU_SEMIPLANAR_422 == enPixelFormat) {
        u32Size = (pVBuf->width_stride[0]) * (pVBuf->height) * 2;
        u32UvHeight = pVBuf->height;
    } else if(HI_PIXEL_FORMAT_YUV_400 == enPixelFormat) {
        u32Size = (pVBuf->width_stride[0]) * (pVBuf->height);
        u32UvHeight = pVBuf->height;
    }
    phy_addr = pVBuf->phys_addr[0];
    printf("phy_addr:%#x, u32Size:%d\n", phy_addr, u32Size);
    pUserPageAddr[0] = (hi_char*)phy_addr;
    if (HI_NULL == pUserPageAddr[0]) {
        return;
    }
    printf("stride: %d,%d\n",pVBuf->width_stride[0], pVBuf->width_stride[1] );
    pVBufVirt_Y = pUserPageAddr[0];
    pVBufVirt_C = pVBufVirt_Y + (pVBuf->width_stride[0]) * (pVBuf->height);
    /* save Y ----------------------------------------------------------------*/
    fprintf(stderr, "saving......Y......");
    fflush(stderr);
    if(3 == enVideoFormat) {
        for (h = 0; h < pVBuf->height; h++) {
            pMemContent = pVBufVirt_Y + h * pVBuf->width_stride[0];
            fwrite(pMemContent, pVBuf->width_stride[0], 1, pfd);
        }
    } else {
        for (h = 0; h < pVBuf->height; h++) {
            pMemContent = pVBufVirt_Y + h * pVBuf->width_stride[0];
            fwrite(pMemContent, pVBuf->width, 1, pfd);
        }
    }

    if(HI_PIXEL_FORMAT_YUV_400 != enPixelFormat && 3 != enVideoFormat) {
        fflush(pfd);
        /* save U ----------------------------------------------------------------*/
        fprintf(stderr, "U......");
        fflush(stderr);

        for (h = 0; h < u32UvHeight; h++) {
            pMemContent = pVBufVirt_C + h * pVBuf->width_stride[1];
            pMemContent += 1;
            for (w = 0; w < pVBuf->width / 2; w++) {
                TmpBuff[w] = *pMemContent;
                pMemContent += 2;
            }
            fwrite(TmpBuff, pVBuf->width / 2, 1, pfd);
        }
        fflush(pfd);
        /* save V ----------------------------------------------------------------*/
        fprintf(stderr, "V......");
        fflush(stderr);
        for (h = 0; h < u32UvHeight; h++) {
            pMemContent = pVBufVirt_C + h * pVBuf->width_stride[1];
            for (w = 0; w < pVBuf->width / 2; w++) {
                TmpBuff[w] = *pMemContent;
                pMemContent += 2;
            }
            fwrite(TmpBuff, pVBuf->width / 2, 1, pfd);
        }
    }
    fflush(pfd);
    fprintf(stderr, "done %d!\n", pVBuf->time_ref);
    fflush(stderr);
    pUserPageAddr[0] = HI_NULL;
}

static int SampleSaveUncompressRaw(hi_video_frame* pVBuf, hi_u32 u32Nbit,
    FILE* pfd, hi_u32 u32ByteAlign)
{
    hi_u32 u32H;
    hi_u16 *pu16Data = NULL;
    hi_u64 phy_addr, size;
    hi_u8* pUserPageAddr[2];
    hi_u8  *pu8Data;
    hi_pixel_format enPixelFormat = HI_PIXEL_FORMAT_BUTT;

    BitWidth2PixelFormat(u32Nbit, &enPixelFormat);
    if (enPixelFormat != pVBuf->pixel_format) {
        fprintf(stderr, "NoCmp: invalid pixel format:%d, u32Nbit: %d\n",
            pVBuf->pixel_format, u32Nbit);
        return HI_FAILURE;
    }

    size = (pVBuf->width_stride[0]) * (pVBuf->height);
    phy_addr = pVBuf->phys_addr[0];
    pUserPageAddr[0] = (hi_u8*)phy_addr;
    if (NULL == pUserPageAddr[0]) {
        return HI_FAILURE;
    }
    pu8Data = pUserPageAddr[0];
    if ((8 != u32Nbit) && (16 != u32Nbit)) {
        pu16Data = (hi_u16*)malloc(pVBuf->width * 2U);
        if (NULL == pu16Data) {
            fprintf(stderr, "alloc memory failed\n");
            pUserPageAddr[0] = NULL;
            return HI_FAILURE;
        }
    }
    /* save Y ----------------------------------------------------------------*/
    fprintf(stderr, "saving......dump data......u32Stride[0]: %d, width: %d\n",
        pVBuf->width_stride[0], pVBuf->width);
    fflush(stderr);

    for (u32H = 0; u32H < pVBuf->height; u32H++) {
        if (8 == u32Nbit) {
            fwrite(pu8Data, pVBuf->width, 1, pfd);
        } else if (16 == u32Nbit) {
            fwrite(pu8Data, pVBuf->width, 2, pfd);
            fflush(pfd);
        } else {
            if (1 == u32ByteAlign) {
                ConvertBitPixel(pu8Data, pVBuf->width, u32Nbit, pu16Data);
                fwrite(pu16Data, pVBuf->width, 2, pfd);
            } else {
                if (0 == ((pVBuf->width * u32Nbit) % 8)) {
                    //-- pVBuf->width * u32Nbit / 8
                    fwrite(pu8Data, pVBuf->width * u32Nbit / 8, 1, pfd);
                } else {
                    fwrite(pu8Data, ((pVBuf->width * u32Nbit) / 8 + 8), 1, pfd);
                }
            }
        }
        pu8Data += pVBuf->width_stride[0];
    }
    fflush(pfd);
    fprintf(stderr, "u32Nbit_%d done u32TimeRef: %d, u32ByteAlign: %d!\n", u32Nbit,
        pVBuf->time_ref, u32ByteAlign);
    fflush(stderr);
    if (NULL != pu16Data) {
        free(pu16Data);
    }
    pUserPageAddr[0] = NULL;
    return HI_SUCCESS;
}

static int SampleSaveCompressedRaw(hi_video_frame* pVBuf, hi_u32 u32Nbit, FILE* pfd)
{
    hi_u32 u32H;
    hi_u32 u32DataSize;
    hi_u16 u16HeadData = 0x0;
    hi_u64 phy_addr, size;
    hi_u8* pUserPageAddr[2];
    hi_u8  *pu8Data;
    hi_pixel_format enPixelFormat = HI_PIXEL_FORMAT_BUTT;

    BitWidth2PixelFormat(u32Nbit, &enPixelFormat);
    if (enPixelFormat != pVBuf->pixel_format) {
        fprintf(stderr, "Cmp: invalid pixel format:%d, u32Nbit: %d\n", pVBuf->pixel_format, u32Nbit);
        return HI_FAILURE;
    }

    size = (pVBuf->width_stride[0]) * (pVBuf->height);
    phy_addr = pVBuf->phys_addr[0];
    pUserPageAddr[0] = (hi_u8*)phy_addr;
    if (NULL == pUserPageAddr[0]) {
        return HI_FAILURE;
    }

    pu8Data = pUserPageAddr[0];
    /* save Y ----------------------------------------------------------------*/
    fprintf(stderr, "saving......dump data......u32Stride[0]: %d, width: %d\n",
        pVBuf->width_stride[0], pVBuf->width);
    fflush(stderr);
    for (u32H = 0; u32H < pVBuf->height; u32H++) {
        u16HeadData = *(hi_u16*)pu8Data;
        u32DataSize =  (u16HeadData + 1) * 16;
        fwrite(pu8Data, u32DataSize, 1, pfd);
        pu8Data += pVBuf->width_stride[0];
    }
    fflush(pfd);

    fprintf(stderr, "done u32TimeRef: %d!\n", pVBuf->time_ref);
    fflush(stderr);
    pUserPageAddr[0] = NULL;
    return HI_SUCCESS;
}

static int SampleBayerDump(hi_video_frame* pVBuf, hi_u32 u32Nbit, FILE* pfd, hi_u32 u32ByteAlign)
{
    if(HI_COMPRESS_MODE_NONE == pVBuf->compress_mode) {
        return SampleSaveUncompressRaw(pVBuf, u32Nbit, pfd, u32ByteAlign);
    } else {
        return SampleSaveCompressedRaw(pVBuf, u32Nbit, pfd);
    }
}

char* CompressModeToString(hi_compress_mode enCompressMode)
{
    if(HI_COMPRESS_MODE_NONE == enCompressMode) {
        return "CMP_NONE";
    } else if(HI_COMPRESS_MODE_LINE == enCompressMode) {
        return "CMP_LINE";
    } else if(HI_COMPRESS_MODE_SEG == enCompressMode) {
        return "CMP_SEG";
    } else {
        return "CMP_XXX";
    }
}

hi_s32 DumpLinearBayer(hi_vi_pipe ViPipe, hi_u32 u32Nbit, hi_compress_mode enCompressMode, hi_u32 u32Cnt,
    hi_u32 u32ByteAlign, hi_u32 u32RatioShow)
{
    hi_s32                 s32Ret = HI_SUCCESS;
    hi_s32                 i, j;
    hi_char                szYuvName[256] = {0};
    FILE*                  pfd = NULL;
    hi_s32                 s32MilliSec = 5000; // 5s
    hi_u32                 u32CapCnt = 0;
    hi_u64                 u64IspInfoPhyAddr = 0;
    hi_video_frame_info     astFrame[MAX_FRM_CNT];
    hi_pixel_format         enPixelFormat;

    BitWidth2PixelFormat(u32Nbit, &enPixelFormat);
    while (1) {
        printf("Start get vi Pipe %d frame\n", ViPipe);
        if (HI_SUCCESS != hi_mpi_vi_get_pipe_frame(ViPipe, &astFrame[0], s32MilliSec)) {
            printf("Linear: get vi Pipe %d frame err\n", ViPipe);
            break;
        }
        if ((astFrame[0].v_frame.compress_mode == enCompressMode)
            && (astFrame[0].v_frame.pixel_format == enPixelFormat)) {
            hi_mpi_vi_release_pipe_frame(ViPipe, &astFrame[0]);
            break;
        }
        hi_mpi_vi_release_pipe_frame(ViPipe, &astFrame[0]);
    }
    /* get VI frame  */
    for (i = 0; i < u32Cnt; i++) {
        if (HI_SUCCESS != hi_mpi_vi_get_pipe_frame(ViPipe, &astFrame[i], s32MilliSec)) {
            printf("Linear: get vi Pipe %d frame err\n", ViPipe);
            printf("only get %d frame\n", i);
            break;
        }
        printf("Linear: get vi Pipe %d frame num %d ok\n",ViPipe,  i);
    }
    u32CapCnt = i;
    if (0 == u32CapCnt) {
        return -1;
    }
    /* make file name */
    if (0 == u32RatioShow) {
        snprintf(szYuvName, 256, "./vi_Pipe_%d_%d_%d_%d_%dbits_%s_%d_%d_%d.raw",
            ViPipe, astFrame[0].v_frame.width,
            astFrame[0].v_frame.height, u32CapCnt, u32Nbit,
            CompressModeToString(astFrame[0].v_frame.compress_mode),
            u32ByteAlign, u32RatioShow, astFrame[0].v_frame.time_ref);
    }
    /* open file */
    pfd = fopen(szYuvName, "wb");
    if (NULL == pfd) {
        printf("open file failed:%s!\n", strerror(errno));
        goto end;
    }

    for (j = 0; j < u32CapCnt; j++) {
        /* save VI frame to file */
        printf("SampleBayerDump u32Nbit = %d!\n", u32Nbit);
        SampleBayerDump(&astFrame[j].v_frame, u32Nbit, pfd, u32ByteAlign);
        /* release frame after using */
        hi_mpi_vi_release_pipe_frame(ViPipe, &astFrame[j]);
    }
    fclose(pfd);
    return HI_SUCCESS;
end:
    for (j = 0; j < u32CapCnt; j++) {
        hi_mpi_vi_release_pipe_frame(ViPipe, &astFrame[j]);
    }
    return HI_FAILURE;
}

static hi_s32 ConvertBitPixel(hi_u8 *pu8Data, hi_u32 u32DataNum, hi_u32 u32BitWidth, hi_u16 *pu16OutData)
{
    hi_s32  i, u32Tmp, s32OutCnt;
    hi_u64  u64Val;
    hi_u32  u32Val;
    hi_u8  *pu8Tmp = pu8Data;
    s32OutCnt = 0;
    switch(u32BitWidth) {
    case 10:
        {
            /* 4 pixels consist of 5 bytes  */
            u32Tmp = u32DataNum / 4;
            for (i = 0; i < u32Tmp; i++) {
                /* byte4 byte3 byte2 byte1 byte0 */
                pu8Tmp = pu8Data + 5 * i;
                u64Val = pu8Tmp[0] + ((hi_u32)pu8Tmp[1] << 8) + ((hi_u32)pu8Tmp[2] << 16) +
                         ((hi_u32)pu8Tmp[3] << 24) + ((hi_u64)pu8Tmp[4] << 32);
                pu16OutData[s32OutCnt++] = u64Val & 0x3ff;
                pu16OutData[s32OutCnt++] = (u64Val >> 10) & 0x3ff;
                pu16OutData[s32OutCnt++] = (u64Val >> 20) & 0x3ff;
                pu16OutData[s32OutCnt++] = (u64Val >> 30) & 0x3ff;
            }
        }
        break;
    case 12:
        {
            /* 2 pixels consist of 3 bytes  */
            u32Tmp = u32DataNum / 2;
            for (i = 0; i < u32Tmp; i++) {
                /* byte2 byte1 byte0 */
                pu8Tmp = pu8Data + 3 * i;
                u32Val = pu8Tmp[0] + (pu8Tmp[1] << 8) + (pu8Tmp[2] << 16);
                pu16OutData[s32OutCnt++] = u32Val & 0xfff;
                pu16OutData[s32OutCnt++] = (u32Val >> 12) & 0xfff;
            }
        }
        break;
    case 14:
        {
            /* 4 pixels consist of 7 bytes  */
            u32Tmp = u32DataNum / 4;
            for (i = 0; i < u32Tmp; i++) {
                pu8Tmp = pu8Data + 7 * i;
                u64Val = pu8Tmp[0] + ((hi_u32)pu8Tmp[1] << 8) + ((hi_u32)pu8Tmp[2] << 16) +
                         ((hi_u32)pu8Tmp[3] << 24) + ((hi_u64)pu8Tmp[4] << 32) +
                         ((hi_u64)pu8Tmp[5] << 40) + ((hi_u64)pu8Tmp[6] << 48);

                pu16OutData[s32OutCnt++] = u64Val & 0x3fff;
                pu16OutData[s32OutCnt++] = (u64Val >> 14) & 0x3fff;
                pu16OutData[s32OutCnt++] = (u64Val >> 28) & 0x3fff;
                pu16OutData[s32OutCnt++] = (u64Val >> 42) & 0x3fff;
            }
        }
        break;
    default:
        fprintf(stderr, "unsuport bitWidth: %d\n", u32BitWidth);
        return -1;
        break;
    }
    return s32OutCnt;
}