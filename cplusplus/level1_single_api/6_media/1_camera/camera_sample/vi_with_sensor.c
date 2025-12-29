#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <errno.h>
#include "vi_with_sensor.h"
#include "sample_comm.h"
#include "sample_comm_vi.h"
#include "sample_comm_isp.h"
#include "image_dump_util.h"
#include "acl/acl_rt.h"
#include "acl/acl.h"

#define HI_MAX_PIPE_NUM  4
#define HI_MAX_NAME_SIZE 128
SAMPLE_VI_CONFIG_S g_stViConfig;
hi_s32 pipe_mult_sensor[HI_MAX_PIPE_NUM] = {0};

hi_u32 hi_pixel_format_bit_width(hi_pixel_format enPixelFormat)
{
    switch (enPixelFormat) {
        case HI_PIXEL_FORMAT_RGB_BAYER_8BPP:
            return 8;
        case HI_PIXEL_FORMAT_RGB_BAYER_10BPP:
            return 10;
        case HI_PIXEL_FORMAT_RGB_BAYER_12BPP:
            return 12;
        case HI_PIXEL_FORMAT_RGB_BAYER_14BPP:
            return 14;
        case HI_PIXEL_FORMAT_RGB_BAYER_16BPP:
            return 16;
        default:
            return 0;
    }
}

static hi_s32 sample_sys_init()
{
    hi_s32 ret = HI_FAILURE;
    hi_mpi_sys_exit();

    SAMPLE_PRT("begin hi_mpi_sys_init !\n");
    ret = hi_mpi_sys_init();
    if (HI_SUCCESS != ret) {
        SAMPLE_PRT("hi_mpi_sys_init failed!\n");
        return HI_FAILURE;
    }
    SAMPLE_PRT("begin aclInit !\n");
    ret = aclInit(NULL);
    if (ret != HI_SUCCESS) {
        SAMPLE_PRT("aclInit failed!\n");
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

hi_void close_camera(void)
{
    hi_s32 s32Ret;
    (void)Sample_Comm_Vi_StopVi(&g_stViConfig);
    s32Ret = hi_mpi_sys_exit();
    if (s32Ret != HI_SUCCESS) {
        SAMPLE_PRT("hi_mpi_sys_exit failed!\n");
    } else {
        SAMPLE_PRT("hi_mpi_sys_exit succ \n");
    }
    s32Ret = aclFinalize();
    if (s32Ret != ACL_SUCCESS) {
        SAMPLE_PRT("aclFinalize failed!\n");
    } else {
        SAMPLE_PRT("aclFinalize succ \n");
    }
    return;
}

hi_s32 dump_raw(hi_vi_pipe ViPipe)
{
    hi_s32 s32Ret       = 0;
    hi_u32 u32Nbit      = 12;
    hi_u32 u32FrmCnt    = 1;
    hi_u32 u32RawDepth  = 2;
    hi_u32 u32ByteAlign = 1;
    hi_u32 u32RatioShow = 0;
    hi_compress_mode enCompressMode = HI_COMPRESS_MODE_NONE;
    hi_vi_frame_dump_attr astBackUpDumpAttr;
    hi_vi_frame_dump_attr stDumpAttr;
    hi_vi_pipe_attr stPipeAttr;

    if (1 > u32FrmCnt || MAX_FRM_CNT < u32FrmCnt) {
        printf("invalid FrmCnt %d, FrmCnt range from 1 to %d\n", u32FrmCnt, MAX_FRM_CNT);
        exit(HI_FAILURE);
    }

    s32Ret = hi_mpi_vi_get_pipe_attr(ViPipe, &stPipeAttr);
    if (HI_SUCCESS != s32Ret) {
        printf("Get Pipe %d attr failed!\n", ViPipe);
        return s32Ret;
    }
    u32Nbit = hi_pixel_format_bit_width(stPipeAttr.pixel_format);
    if (u32Nbit == 0) {
        printf("the enPixFmt is %d, cannot dump raw!\n", stPipeAttr.pixel_format);
        return -1;
    }  
    s32Ret = hi_mpi_vi_get_pipe_frame_dump_attr(ViPipe, &astBackUpDumpAttr);
    if (HI_SUCCESS != s32Ret) {
        printf("Get Pipe %d dump attr failed!\n", ViPipe);
        return s32Ret;
    }
    memcpy(&stDumpAttr, &astBackUpDumpAttr, sizeof(hi_vi_frame_dump_attr));
    stDumpAttr.enable = HI_TRUE;
    stDumpAttr.depth = u32RawDepth;
    s32Ret = hi_mpi_vi_set_pipe_frame_dump_attr(ViPipe, &stDumpAttr);
    if (HI_SUCCESS != s32Ret) {
        printf("Set Pipe %d dump attr failed!\n", ViPipe);
        return s32Ret;
    }
    // dump raw
    DumpLinearBayer(ViPipe, u32Nbit, enCompressMode, u32FrmCnt, u32ByteAlign, u32RatioShow);
    stDumpAttr.enable = HI_FALSE;
    s32Ret = hi_mpi_vi_set_pipe_frame_dump_attr(ViPipe, &stDumpAttr);
    if (HI_SUCCESS != s32Ret) {
        printf("Set Pipe %d dump attr failed!\n", ViPipe);
        return s32Ret;
    }
    return s32Ret;
}

hi_s32 dump(hi_s32 Pipe)
{
    hi_char szYuvName[HI_MAX_NAME_SIZE];
    hi_u32 depth = 3;
    hi_s32 s32MilliSec = 120000;
    hi_video_frame_info stFrame;
    hi_s32 s32Ret;
    hi_vi_chn_attr stChnAttr;
    hi_s32 Chn = 0;
    FILE* pfd = HI_NULL;

    s32Ret = hi_mpi_vi_get_chn_attr(Pipe, Chn, &stChnAttr);
    if (s32Ret != HI_SUCCESS)  {
        SAMPLE_PRT("get chn attr error!!!\n");
        return HI_FAILURE;
    }

    stChnAttr.depth = depth;
    s32Ret = hi_mpi_vi_set_chn_attr(Pipe, Chn, &stChnAttr);
    if (s32Ret != HI_SUCCESS) {
        SAMPLE_PRT("set chn attr error!!!\n");
        return HI_FAILURE;
    }

    memset(&stFrame, 0, sizeof(stFrame));

    s32Ret = hi_mpi_vi_get_chn_frame(Pipe, Chn, &stFrame, s32MilliSec);
    if (s32Ret != HI_SUCCESS) {
        SAMPLE_PRT("Get frame fails s32Ret = 0x%x!!\n", s32Ret);
    }
    SAMPLE_PRT("hi_mpi_vi_get_chn_frame success\n!");

    if (s32Ret != HI_SUCCESS) {
        SAMPLE_PRT("get frame error for 4 times, now exit !!!\n");
        return HI_FAILURE;
    }
    /* get frame */
    snprintf(szYuvName, HI_MAX_NAME_SIZE, "./vi_pipe%d_chn%d_w%d_h%d.yuv",
        Pipe, Chn, stFrame.v_frame.width, stFrame.v_frame.height);
    /* open file */
    pfd = fopen(szYuvName, "wb");
    if (HI_NULL == pfd) {
        SAMPLE_PRT("open file failed:%s!\n", strerror(errno));
    }
    hi_u32 u32Size = (stFrame.v_frame.width) * (stFrame.v_frame.height) * 3 / 2;
    fwrite(stFrame.v_frame.phys_addr[0], u32Size, 1, pfd);
    fflush(pfd);
    fclose(pfd);
    s32Ret = hi_mpi_vi_release_chn_frame(Pipe, Chn, &stFrame);
    if (HI_SUCCESS != s32Ret) {
        SAMPLE_PRT("Release frame error ,now exit s32Ret is %#x!!!\n", s32Ret);
    }
    return s32Ret;
}

hi_s32 start_camera(hi_u32 index, case_info sensor_info)
{
    hi_s32 ret;
    hi_size stSize;
    hi_s32 sensorId;
    hi_s32 num;
    lane_divide_mode_t divide_mode;
    SAMPLE_SNS_TYPE_E sensor_type;
    SAMPLE_VI_INFO_S sampleViInfo[HI_MAX_PIPE_NUM];

    ret = sample_sys_init();
    if (ret != HI_SUCCESS) {
        goto sys_init_failed;
    }
    // Obtaining Sensor and MIPI Information
    if (sensor_info.sensor_num == 1) {
        divide_mode = 1;
        g_stViConfig.s32WorkingViNum = 1; // simple pipe
    } else {
        divide_mode = 3;
        g_stViConfig.s32WorkingViNum = HI_MAX_PIPE_NUM;
    }
    // 1: imx219
    if (sensor_info.sensor_type == 1) {
        sensor_type = SONY_IMX219_MIPI_1920x1080_10BIT_30FPS_NORMAL;
    } else if (sensor_info.sensor_type == 2) {
        sensor_type = SONY_IMX477_MIPI_1920x1080_10BIT_DOL2_15FPS;
    } else {
        goto sys_init_failed;
    }
    SAMPLE_PRT("The input sensor type is %#x\n", sensor_type);

    g_stViConfig.mipi_lane_divide_mode = divide_mode;
    for (num = 0; num < g_stViConfig.s32WorkingViNum; num++) {
        ret = get_sample_vi_info_by_sensorType(sensor_type, &sampleViInfo[num], num);
        if (ret != HI_SUCCESS) {
            return -1;
        }
        hi_vi_dev_attr stViDevAttr;
        Sample_Comm_Vi_GetDevAttrBySns(sensor_type, &stViDevAttr);
        g_stViConfig.as32WorkingViId[num] = num;
        memcpy(&(g_stViConfig.astViInfo[num]), &sampleViInfo[num], sizeof(SAMPLE_VI_INFO_S));
    }

    ret = Sample_Comm_Vi_StartVi(&g_stViConfig);
    if (ret != HI_SUCCESS) {
        goto start_vi_failed;
    }

    return 0;
start_vi_failed:
    Sample_Comm_Vi_StopVi(&g_stViConfig);
sys_init_failed:
    return -1;
}

hi_s32 get_sample_vi_info_by_sensorType(const SAMPLE_SNS_TYPE_E sensor_type,
    SAMPLE_VI_INFO_S *sample_vi_info, hi_s32 dev_id)
{
    static SAMPLE_VI_INFO_S sample_vi_info_linear = {
        .stSnsInfo = {
            .s32SnsId = 4,
            .s32BusId = 2,
            .MipiDev = 0,
        },
        .stDevInfo = {
            .enWDRMode = HI_WDR_MODE_NONE,
        },
        .stPipeInfo = {
            .aPipe = {0, -1, -1, -1},
        },
        .stChnInfo = {
            .ViChn = 0,
            .enPixFormat = HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420,
            .enDynamicRange = HI_DYNAMIC_RANGE_SDR8,
            .enVideoFormat = HI_VIDEO_FORMAT_LINEAR,
            .enCompressMode = HI_COMPRESS_MODE_NONE
        }
    };
    switch (sensor_type) {
        case SONY_IMX219_MIPI_3K_10BIT_NORMAL:
        case SONY_IMX219_MIPI_1920x1080_10BIT_NORMAL:
        case SONY_IMX219_MIPI_1920x1080_10BIT_30FPS_NORMAL:
            memcpy(sample_vi_info, &sample_vi_info_linear, sizeof(SAMPLE_VI_INFO_S));
            sample_vi_info->stSnsInfo.enSnsType = sensor_type;
            sample_vi_info->stSnsInfo.s32BusId = dev_id;
            sample_vi_info->stSnsInfo.s32SnsId = 0;
            sample_vi_info->stSnsInfo.MipiDev = dev_id;
            sample_vi_info->stPipeInfo.aPipe[0] = dev_id;
            sample_vi_info->stDevInfo.ViDev = dev_id;
            pipe_mult_sensor[dev_id] = dev_id;
            break;
        case SONY_IMX477_MIPI_1920x1080_10BIT_DOL2_15FPS:
            memcpy(sample_vi_info, &sample_vi_info_linear, sizeof(SAMPLE_VI_INFO_S));
            sample_vi_info->stSnsInfo.enSnsType = sensor_type;
            sample_vi_info->stSnsInfo.s32BusId = 0 + dev_id * 2;
            sample_vi_info->stSnsInfo.s32SnsId = 1;
            sample_vi_info->stSnsInfo.MipiDev = 0 + dev_id * 2;
            sample_vi_info->stPipeInfo.aPipe[0] = 0 + dev_id * 2;
            sample_vi_info->stPipeInfo.aPipe[1] = 1 + dev_id * 2;
            sample_vi_info->stDevInfo.ViDev = 0 + dev_id * 2;
            sample_vi_info->stDevInfo.enWDRMode = HI_WDR_MODE_2To1_LINE;
            pipe_mult_sensor[dev_id] = 0 + 2 * dev_id;
            break;
        default:
            SAMPLE_PRT("Function %s() Error. Undefined sensor type: %#x\n", __FUNCTION__, sensor_type);
            return HI_FAILURE;
    }

    return HI_SUCCESS;
}

hi_s32 vi_with_sensor(hi_u32 index, case_info info) {
    hi_s32 ret;
    ret = start_camera(index, info);
    if (ret != 0) {
        goto exit;
    }
    if (index == 0) { // dump yuv
        sleep(1); // 休眠1s，保证出图效果正常
        dump(0);
        if (info.sensor_num == 2) { // dump 2路图像       
          dump(2);
        }
    } else { // dump raw
        sleep(1); // 休眠1s，保证出图效果正常
        dump_raw(0);
        if (info.sensor_num == 2) { // dump 2路图像        
          dump_raw(2);
        }  
    }
exit:
    close_camera();
    return 0;
}