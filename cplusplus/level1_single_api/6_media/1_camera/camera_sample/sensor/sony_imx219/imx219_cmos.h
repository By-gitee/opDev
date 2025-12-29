/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Header file of imx219_coms
 * Author: Hisilicon multimedia software group
 * Create: 2023-03-06
 */

#ifndef IMX219_CMOS_H
#define IMX219_CMOS_H

#include "hi_common_isp.h"
#include "hi_sns_ctrl.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#define IMX219_I2C_ADDR    0x20
#define IMX219_ADDR_BYTE   2U
#define IMX219_DATA_BYTE   1U
#define IMX219_SENSOR_GET_CTX(dev, pstCtx)   ((pstCtx) = imx219_get_ctx(dev))

#define IMX219_FULL_LINES_MAX 0xFFFF

#ifndef MIN
#define MIN(a, b) (((a) > (b)) ?  (b) : (a))
#endif

#define     FULL_LINES_MAX                  (0xFFFF)
// registers to control exposure
#define     IMX219_COARSE_INTEG_TIME_L      (0x0203)
#define     IMX219_COARSE_INTEG_TIME_H      (0x0202)
#define     IMX219_ANA_GAIN_GLOBAL_L        (0x0205)
#define     IMX219_ANA_GAIN_GLOBAL_H        (0x0204)
#define     IMX219_DPGA_USE_GLOBAL_GAIN     (0x3FF9)
#define     IMX219_DIG_GAIN_GR_L            (0x020F)
#define     IMX219_DIG_GAIN_GR_H            (0x020E)
#define     IMX219_LINE_LENGTH_PCK_L        (0x341)
#define     IMX219_LINE_LENGTH_PCK_H        (0x340)
#define     IMX219_FRM_LENGTH_CTL           (0x350)
#define     IMX219_PRSH_LENGTH_LINE_L       (0x3F3B)
#define     IMX219_PRSH_LENGTH_LINE_H       (0x3F3A)

typedef enum {
    IMX219_24M_RAW10_MODE = 0,
    IMX219_24M_RAW8_MODE,
    IMX219_1920x1080_RAW10_30FPS_MODE,
    IMX219_MODE_BUTT
} IMX219_RES_MODE_E;

typedef struct hiIMX219_VIDEO_MODE_TBL_S {
    hi_u32      u32VertiLines;
    hi_u32      u32MaxVertiLines;
    hi_float    f32MaxFps;
    hi_float    f32MinFps;
    hi_u32      u32Width;
    hi_u32      u32Height;
    hi_u8       u8SnsMode;
    const char *pszModeName;
} IMX219_VIDEO_MODE_TBL_S;

hi_isp_sns_state *imx219_get_ctx(hi_vi_pipe vi_pipe);

extern hi_isp_sns_state      *g_pastImx219[HI_ISP_MAX_PIPE_NUM];
extern hi_isp_sns_commbus     g_aunImx219BusInfo[HI_ISP_MAX_PIPE_NUM];
extern const IMX219_VIDEO_MODE_TBL_S g_astImx219ModeTbl[IMX219_MODE_BUTT];
extern hi_isp_slave_sns_sync gstImx219Sync[HI_ISP_MAX_PIPE_NUM];
extern hi_u8 g_u8Sensor219ImageMode;
extern hi_u8 g_u8Imx219LensMode;

void imx219_init(hi_vi_pipe viPipe);
void imx219_exit(hi_vi_pipe viPipe);
void imx219_standby(hi_vi_pipe viPipe);
void imx219_restart(hi_vi_pipe viPipe);
int imx219_write_register(hi_vi_pipe viPipe, hi_u32 addr, hi_u32 data);
int imx219_read_register(hi_vi_pipe viPipe, hi_u32 addr);
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
#endif /* IMX219_CMOS_H */
