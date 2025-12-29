#ifndef __SAMPLE_COMM_H__
#define __SAMPLE_COMM_H__

#include <pthread.h>

#include "hi_mpi_isp.h"
#include "hi_mpi_vi.h"
#include "hi_mpi_ae.h"
#include "hi_mpi_awb.h"
#include "hi_mpi_vpss.h"
#include "hi_common_3a.h"
#include "hi_sns_ctrl.h"
#include "sensor_management.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

/*******************************************************
    macro define
*******************************************************/
#define FILE_NAME_LEN               128
#define WDR_MAX_PIPE_NUM            4
#define MAX_FRM_CNT                 25

#define CHECK_RET(express,name) \
    do{ \
        hi_s32 Ret; \
        Ret = express; \
        if (HI_SUCCESS != Ret) \
        { \
            printf("\033[0;31m%s failed at %s: LINE: %d with %#x!\033[0;39m\n", \
                name, __FUNCTION__, __LINE__, Ret); \
            return Ret; \
        } \
    }while(0)

#define SAMPLE_PRT(fmt...)   \
    do {\
        printf("[%s]-%d: ", __FUNCTION__, __LINE__);\
        printf(fmt);\
    }while(0)

/*******************************************************
    enum define
*******************************************************/
typedef enum hiPIC_SIZE_E
{
    PIC_CIF,
    PIC_360P,      /* 640 * 360 */
    PIC_720P,	   /* 1280 * 720  */
    PIC_1080P,	   /* 1920 * 1080 */
    PIC_2592x1520,
    PIC_2592x1944,
    PIC_3264x2448,
    PIC_3840x2160,
    PIC_4096x2160,
    PIC_BUTT
} PIC_SIZE_E;

typedef struct HISAMPLE_SENSOR_INFO_S
{
    SAMPLE_SNS_TYPE_E   enSnsType;
    hi_s32              s32SnsId;
    hi_s32              s32BusId;
    combo_dev_t         MipiDev;
} SAMPLE_SENSOR_INFO_S;

typedef struct HISAMPLE_DEV_INFO_S
{
    hi_vi_dev      ViDev;
    hi_wdr_mode    enWDRMode;
} SAMPLE_DEV_INFO_S;

typedef struct HISAMPLE_PIPE_INFO_S
{
    hi_vi_pipe           aPipe[WDR_MAX_PIPE_NUM];
    hi_u32               u32MaxW;                /* RW;Range[VI_PIPE_MIN_WIDTH, VI_PIPE_MAX_WIDTH];Maximum width */
    hi_u32               u32MaxH;                /* RW;Range[VI_PIPE_MIN_HEIGHT, VI_PIPE_MAX_HEIGHT];Maximum height */
    hi_data_bit_width    enBitWidth;             /* RW;Range:[0, 4];Bit width */
    hi_compress_mode     enCompressMode; // VI压缩模式
} SAMPLE_PIPE_INFO_S;

typedef struct HISAMPLE_CHN_INFO_S
{
    hi_vi_chn            ViChn;
    hi_pixel_format      enPixFormat;
    hi_dynamic_range     enDynamicRange;
    hi_video_format      enVideoFormat;
    hi_compress_mode     enCompressMode;
} SAMPLE_CHN_INFO_S;

typedef struct HISAMPLE_VI_INFO_S
{
    SAMPLE_SENSOR_INFO_S    stSnsInfo;
    SAMPLE_DEV_INFO_S       stDevInfo;
    SAMPLE_PIPE_INFO_S      stPipeInfo;
    SAMPLE_CHN_INFO_S       stChnInfo;
} SAMPLE_VI_INFO_S;

typedef struct HISAMPLE_VI_CONFIG_S
{
    SAMPLE_VI_INFO_S    astViInfo[HI_VI_MAX_DEV_NUM];
    hi_s32              as32WorkingViId[HI_VI_MAX_DEV_NUM];
    hi_s32              s32WorkingViNum;
    lane_divide_mode_t  mipi_lane_divide_mode;
} SAMPLE_VI_CONFIG_S;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif /* End of #ifndef __SAMPLE_COMMON_H__ */
