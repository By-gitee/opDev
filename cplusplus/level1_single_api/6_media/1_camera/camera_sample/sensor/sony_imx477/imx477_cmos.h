/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Header file of imx477_coms
 * Author: Hisilicon multimedia software group
 * Create: 2023-03-06
 */

#ifndef IMX477_CMOS_H
#define IMX477_CMOS_H

#include "hi_common_isp.h"
#include "hi_sns_ctrl.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#define IMX477_I2C_ADDR    0x34
#define IMX477_ADDR_BYTE   2U
#define IMX477_DATA_BYTE   1U
#define IMX477_SENSOR_GET_CTX(dev, pstCtx)   ((pstCtx) = imx477_get_ctx(dev))

#ifndef MIN
#define MIN(a, b) (((a) > (b)) ?  (b) : (a))
#endif

#define     IMX477_FULL_LINES_MAX                       (0xFFFF)
// registers to control exposure
#define     IMX477_COARSE_INTEG_TIME_L                  (0x0203)
#define     IMX477_COARSE_INTEG_TIME_H                  (0x0202)
#define     IMX477_ANA_GAIN_GLOBAL_L                    (0x0205)
#define     IMX477_ANA_GAIN_GLOBAL_H                    (0x0204)
#define     IMX477_DPGA_USE_GLOBAL_GAIN                 (0x3FF9)
#define     IMX477_DIG_GAIN_GR_L                        (0x020F)
#define     IMX477_DIG_GAIN_GR_H                        (0x020E)
#define     IMX477_FRM_LENGTH_LINES_L                   (0x0341)
#define     IMX477_FRM_LENGTH_LINES_H                   (0x0340)
#define     IMX477_LINE_LENGTH_PCK_L                    (0x0343)
#define     IMX477_LINE_LENGTH_PCK_H                    (0x0342)

#define     IMX477_DOL1_COARSE_INTEG_TIME_L             (0x00EB)
#define     IMX477_DOL1_COARSE_INTEG_TIME_H             (0x00EA)
#define     IMX477_DOL2_COARSE_INTEG_TIME_L             (0x00ED)
#define     IMX477_DOL2_COARSE_INTEG_TIME_H             (0x00EC)
#define     IMX477_DOL1_ANA_GAIN_GLOBAL_L               (0x00F1)
#define     IMX477_DOL1_ANA_GAIN_GLOBAL_H               (0x00F0)
#define     IMX477_DOL2_ANA_GAIN_GLOBAL_L               (0x00F3)
#define     IMX477_DOL2_ANA_GAIN_GLOBAL_H               (0x00F2)
#define     IMX477_DOL1_DIG_GAIN_GR_L                   (0x00F7)
#define     IMX477_DOL1_DIG_GAIN_GR_H                   (0x00F6)
#define     IMX477_DOL2_DIG_GAIN_GR_L                   (0x00F9)
#define     IMX477_DOL2_DIG_GAIN_GR_H                   (0x00F8)

typedef enum {
    IMX477_4056x3040_RAW12_25FPS = 0,
    IMX477_4056x3040_DOL2_RAW10_15fps,
    IMX477_4K_45FPS,
    IMX477_1920x1080_RAW12,
    IMX477_1920x1080_DOL2_RAW10,
    IMX477_MODE_BUTT
} IMX477_RES_MODE_E;

typedef struct hiIMX477_VIDEO_MODE_TBL_S {
    hi_u32      u32VertiLines;
    hi_u32      u32MaxVertiLines;
    hi_float    f32MaxFps;
    hi_float    f32MinFps;
    hi_u32      u32Width;
    hi_u32      u32Height;
    hi_u8       u8SnsMode;
    const char *pszModeName;
} IMX477_VIDEO_MODE_TBL_S;

hi_isp_sns_state *imx477_get_ctx(hi_vi_pipe viPipe);
hi_isp_sns_commbus *imx477_get_bus_Info(hi_vi_pipe viPipe);

extern hi_isp_sns_state      *g_pastImx477[HI_ISP_MAX_PIPE_NUM];
extern hi_isp_sns_commbus     g_aunImx477BusInfo[HI_ISP_MAX_PIPE_NUM];
extern const IMX477_VIDEO_MODE_TBL_S g_astImx477ModeTbl[IMX477_MODE_BUTT];
extern hi_u8 g_u8Sensor477ImageMode;

hi_void imx477_init(hi_vi_pipe viPipe);
hi_void imx477_exit(hi_vi_pipe viPipe);
hi_void imx477_standby(hi_vi_pipe viPipe);
hi_void imx477_restart(hi_vi_pipe viPipe);
hi_s32 imx477_write_register(hi_vi_pipe viPipe, hi_u32 addr, hi_u32 data);
hi_s32 imx477_read_register(hi_vi_pipe viPipe, hi_u32 addr);
hi_s32 imx477_identify_module(hi_vi_pipe viPipe);
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
#endif /* IMX477_CMOS_H */
