/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Function of imx219 adapt isp firmware
 * Author: Hisilicon multimedia software group
 * Create: 2023-03-06
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hi_mpi_isp.h"
#include "hi_mpi_ae.h"
#include "hi_mpi_awb.h"
#include "imx219_cmos_ex.h"
#include "imx219_cmos.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#define IMX219_ID              219U
#define IMX219_AGAIN_TBL_RANGE 233U

#define IMX219_SENSOR_SET_CTX(dev, pstCtx)   ((g_pastImx219[dev]) = (pstCtx))
#define IMX219_SENSOR_RESET_CTX(dev)         (g_pastImx219[dev] = HI_NULL)

#define ISP_SNS_SAVE_INFO_CUR_FRAME 0U
#define ISP_SNS_SAVE_INFO_PRE_FRAME 1U

static hi_u32 g_au32InitExposure[HI_ISP_MAX_PIPE_NUM]  = {0};
static hi_u32 g_au32LinesPer500ms[HI_ISP_MAX_PIPE_NUM] = {0};

static hi_u16 g_au16InitWBGain[HI_ISP_MAX_PIPE_NUM][HI_ISP_RGB_CHN_NUM] = {{0}};
static hi_u16 g_au16SampleRgain[HI_ISP_MAX_PIPE_NUM] = {0};
static hi_u16 g_au16SampleBgain[HI_ISP_MAX_PIPE_NUM] = {0};

typedef struct hiIMX219_STATE_S {
    hi_u8       u8Hcg;
    hi_u32      u32BRL;
    hi_u32      u32RHS1_MAX;
    hi_u32      u32RHS2_MAX;
} IMX219_STATE_S;

IMX219_STATE_S g_astimx219State[HI_ISP_MAX_PIPE_NUM] = {{0}};
// FPGA IMX219 - I2C 2
hi_isp_sns_commbus g_aunImx219BusInfo[HI_ISP_MAX_PIPE_NUM] = {
    [0] = { .i2c_dev = 0},
    [1 ... HI_ISP_MAX_PIPE_NUM - 1] = { .i2c_dev = -1}
};

hi_isp_sns_state *g_pastImx219[HI_ISP_MAX_PIPE_NUM] = {HI_NULL};
hi_isp_sns_state *imx219_get_ctx(hi_vi_pipe vi_pipe)
{
    return g_pastImx219[vi_pipe];
}

const IMX219_VIDEO_MODE_TBL_S g_astImx219ModeTbl[IMX219_MODE_BUTT] = {
    {2645, 2645, 20, 1, 3280, 2464, 0, "3280_2464_raw10_20FPS"  }, // MODE0
    {2645, 2645, 20, 1, 3280, 2464, 1, "3280_2464_raw8_20FPS"  }, // MODE1
    {2645, 1322, 40, 1, 1920, 1080, 2, "IMX219_1920x1080_RAW10_30FPS_MODE"  }, // MODE2
};

static hi_s32 cmos_get_ae_default(hi_vi_pipe viPipe,
    hi_isp_ae_sensor_default *pstAeSnsDft)
{
    hi_isp_sns_state *pstSnsState = HI_NULL;
    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstAeSnsDft);
    IMX219_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER(pstSnsState);
    memset(&pstAeSnsDft->ae_route_attr, 0, sizeof(hi_isp_ae_route));
    pstAeSnsDft->full_lines_std = pstSnsState->fl_std;
    pstAeSnsDft->flicker_freq = 0U; // 0:default flicker_freq
    pstAeSnsDft->full_lines_max = 0xffff; // 0xffff:default full_lines_max
    pstAeSnsDft->full_lines = pstSnsState->fl_std;
    pstAeSnsDft->int_time_accu.accu_type = HI_ISP_AE_ACCURACY_LINEAR;
    pstAeSnsDft->int_time_accu.accuracy = 1.0f; // 1:default int_time_accu.accuracy
    pstAeSnsDft->int_time_accu.offset = 0; // 0:default int_time_accu.offset
    pstAeSnsDft->again_accu.accu_type = HI_ISP_AE_ACCURACY_TABLE;
    pstAeSnsDft->again_accu.accuracy = 1.0f; // 1:default again_accu.accuracy
    pstAeSnsDft->dgain_accu.accu_type = HI_ISP_AE_ACCURACY_LINEAR;
    pstAeSnsDft->dgain_accu.accuracy = 0.00390625f; // 0.00390625:1/256;
    pstAeSnsDft->isp_dgain_shift = 8U; // 8: default isp_dgain_shift
    pstAeSnsDft->min_isp_dgain_target = 1UL << pstAeSnsDft->isp_dgain_shift;
    pstAeSnsDft->max_isp_dgain_target = 2UL << pstAeSnsDft->isp_dgain_shift;
    memcpy(&pstAeSnsDft->piris_attr, &ST_PIRIS_ATTR, sizeof(hi_isp_piris_attr));
    pstAeSnsDft->max_iris_fno = HI_ISP_IRIS_F_NO_1_4;
    pstAeSnsDft->min_iris_fno = HI_ISP_IRIS_F_NO_5_6;
    pstAeSnsDft->ae_route_ex_valid = HI_FALSE;
    pstAeSnsDft->ae_route_attr.total_num = 0; // 0:default ae_route_attr.total_num
    pstAeSnsDft->ae_route_attr_ex.total_num = 0; // 0:default ae_route_attr_ex.total_num
    if (g_au32InitExposure[viPipe] == 0) {
        pstAeSnsDft->init_exposure = 1000000U;  // 1000000:default init_exposure
    }
    if (g_au32LinesPer500ms[viPipe] == 0) {
        /* pstSnsState->fl_std:0xa55 f32MaxFps max:30 */
        pstAeSnsDft->lines_per500ms = (hi_u32)(pstSnsState->fl_std *
            g_astImx219ModeTbl[IMX219_24M_RAW10_MODE].f32MaxFps / 2); /* div 2 */
    } else {
        pstAeSnsDft->lines_per500ms = g_au32LinesPer500ms[viPipe];
    }
    pstAeSnsDft->hist_thresh[0] = 0xd; /*  array index 0, 0xd */
    pstAeSnsDft->hist_thresh[1] = 0x28; /* array index 1, 0x28 */
    pstAeSnsDft->hist_thresh[2] = 0x60; /* array index 2, 0x60 */
    pstAeSnsDft->hist_thresh[3] = 0x80; /* array index 3, 0x80 */
    pstAeSnsDft->ae_compensation = 0x44; /* 0x38 ae_compensation */
    pstAeSnsDft->ae_exp_mode = HI_ISP_AE_EXP_HIGHLIGHT_PRIOR;
    /* pstSnsState->fl_std:0xa55 */
    pstAeSnsDft->max_int_time = pstSnsState->fl_std > 4 ? // 4:fl_std-4
        pstSnsState->fl_std - 4 : pstSnsState->fl_std; // 4:fl_std-4
    pstAeSnsDft->min_int_time = 1; // 1:min_int_time
    pstAeSnsDft->max_int_time_target = 65535U; // 65535:max_int_time_target
    pstAeSnsDft->min_int_time_target = pstAeSnsDft->min_int_time;
    pstAeSnsDft->max_again = 10922U; // 10922:max_again
    pstAeSnsDft->min_again = 1024U; // 1024:min_again
    pstAeSnsDft->max_again_target = pstAeSnsDft->max_again;
    pstAeSnsDft->min_again_target = pstAeSnsDft->min_again;
    pstAeSnsDft->max_dgain = 4095U; // 4095:max_dgain
    pstAeSnsDft->min_dgain = 256U; // 256:min_dgain
    pstAeSnsDft->max_dgain_target = pstAeSnsDft->max_dgain;
    pstAeSnsDft->min_dgain_target = pstAeSnsDft->min_dgain;
    return HI_SUCCESS;
}

static hi_void cmos_config_vmax(hi_isp_sns_state *pstSnsState, hi_u32 u32VMAX)
{
    /* array index 5 */
    pstSnsState->regs_info[0].i2c_data[5].data = HIGH_8BITS(u32VMAX);
    /* array index 6 */
    pstSnsState->regs_info[0].i2c_data[6].data = LOW_8BITS(u32VMAX);
    return;
}


/* the function of sensor set fps */
static hi_void cmos_fps_set(hi_vi_pipe viPipe,
    hi_float f32Fps, hi_isp_ae_sensor_default *pstAeSnsDft)
{
    hi_u32 u32Lines, u32LinesMax;
    hi_float f32MaxFps, f32MinFps;
    hi_isp_sns_state *pstSnsState = HI_NULL;
    SNS_CHECK_PIPE_VOID(viPipe);
    CMOS_CHECK_POINTER_VOID(pstAeSnsDft);
    IMX219_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);
    switch (pstSnsState->img_mode) {
        case IMX219_24M_RAW10_MODE:
        case IMX219_24M_RAW8_MODE:
        case IMX219_1920x1080_RAW10_30FPS_MODE:
            u32Lines = g_astImx219ModeTbl[pstSnsState->img_mode].u32VertiLines;
            u32LinesMax = g_astImx219ModeTbl[pstSnsState->img_mode].u32MaxVertiLines;
            f32MaxFps = g_astImx219ModeTbl[pstSnsState->img_mode].f32MaxFps;
            f32MinFps = g_astImx219ModeTbl[pstSnsState->img_mode].f32MinFps;
            if ((f32Fps <= f32MaxFps) && (f32Fps >= f32MinFps)) {
                /* u32MaxFps max:30 */
                u32Lines = (hi_u32)(u32LinesMax * f32MaxFps / SNS_DIV_0_TO_1_FLOAT(f32Fps));
            } else {
                SNS_ERR_TRACE("imx219 Not support fps: %f, should be in [%f,%f]\n",
                    f32Fps, f32MinFps, f32MaxFps);
                return;
            }
            break;
        default:
            SNS_ERR_TRACE("imx219 Not support ImgMode: %d, should less than %d\n",
                pstSnsState->img_mode, IMX219_MODE_BUTT);
            return;
    }
    cmos_config_vmax(pstSnsState, u32Lines);
    pstSnsState->fl_std = u32Lines;
    pstAeSnsDft->fps = f32Fps;
    pstAeSnsDft->lines_per500ms = (hi_u32)(pstSnsState->fl_std * f32Fps / 2); /* div2 */
    pstAeSnsDft->full_lines_std = pstSnsState->fl_std;
    pstAeSnsDft->max_int_time =
        /* MaxIntTime: fl_std - 5 */
        pstSnsState->fl_std > 5U ? pstSnsState->fl_std - 5U : pstSnsState->fl_std;
    pstSnsState->fl[0] = pstSnsState->fl_std;
    pstAeSnsDft->full_lines = pstSnsState->fl[0];
    return;
}

static hi_void cmos_slow_framerate_set(hi_vi_pipe viPipe,
    hi_u32 u32FullLines, hi_isp_ae_sensor_default *pstAeSnsDft)
{
    hi_isp_sns_state *pstSnsState = HI_NULL;
    SNS_CHECK_PIPE_VOID(viPipe);
    CMOS_CHECK_POINTER_VOID(pstAeSnsDft);
    IMX219_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);
    pstSnsState->fl[0] = u32FullLines; /* array index 0 */
    if (pstSnsState->wdr_mode == HI_WDR_MODE_NONE) {
        /* array index 5 */
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[5].data =
            HIGH_8BITS(u32FullLines);
        /* array index 6 */
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[6].data =
            LOW_8BITS(u32FullLines);
    }
    pstAeSnsDft->full_lines = pstSnsState->fl[0];
    pstAeSnsDft->max_int_time =
        /* MaxIntTime: Flstd - 4 */
        pstSnsState->fl[0] > 4U ? pstSnsState->fl[0] - 4U : pstSnsState->fl[0];
    return;
}

/* while isp notify ae to update sensor regs, ae call these funcs. */
static hi_void cmos_inttime_update(hi_vi_pipe viPipe, hi_u32 u32IntTime)
{
    hi_isp_sns_state *pstSnsState = HI_NULL;
    SNS_CHECK_PIPE_VOID(viPipe);
    IMX219_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);
    // SET CORASE_INTEG_TIME
    /* array index 3 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[3].data
        = HIGH_8BITS(u32IntTime);
    /* array index 4 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[4].data
        = LOW_8BITS(u32IntTime);
    return;
}

static const hi_u32 ad_gain_table[IMX219_AGAIN_TBL_RANGE] = {
    1024, 1028, 1032, 1036, 1041, 1044, 1049, 1053, 1058, 1061, 1066, 1070, 1075, 1078, 1083, 1087,
    1092, 1097, 1101, 1106, 1111, 1115, 1120, 1125, 1131, 1134, 1140, 1145, 1150, 1154, 1160, 1165,
    1170, 1176, 1181, 1187, 1192, 1198, 1203, 1209, 1214, 1220, 1225, 1231, 1237, 1243, 1248, 1254,
    1260, 1267, 1273, 1279, 1285, 1291, 1298, 1304, 1310, 1318, 1324, 1331, 1338, 1344, 1351, 1358,
    1366, 1372, 1380, 1388, 1394, 1402, 1409, 1417, 1425, 1433, 1440, 1448, 1456, 1465, 1473, 1482,
    1489, 1497, 1506, 1515, 1523, 1532, 1543, 1552, 1561, 1570, 1579, 1590, 1599, 1608, 1617, 1629,
    1638, 1649, 1659, 1670, 1680, 1692, 1701, 1713, 1725, 1737, 1747, 1759, 1771, 1784, 1796, 1808,
    1821, 1834, 1846, 1859, 1872, 1885, 1900, 1913, 1927, 1942, 1956, 1972, 1985, 2001, 2017, 2031,
    2048, 2064, 2081, 2098, 2115, 2132, 2149, 2167, 2184, 2202, 2222, 2240, 2261, 2279, 2300, 2319,
    2340, 2362, 2384, 2406, 2428, 2451, 2473, 2496, 2519, 2546, 2569, 2596, 2620, 2647, 2675, 2703,
    2731, 2759, 2788, 2820, 2850, 2879, 2913, 2946, 2981, 3012, 3046, 3085, 3121, 3157, 3197, 3238,
    3276, 3317, 3360, 3403, 3450, 3494, 3542, 3592, 3642, 3692, 3744, 3800, 3853, 3911, 3970, 4035,
    4095, 4162, 4230, 4298, 4368, 4444, 4522, 4600, 4681, 4768, 4856, 4947, 5038, 5138, 5246, 5349,
    5461, 5576, 5699, 5825, 5961, 6100, 6242, 6394, 6551, 6719, 6899, 7084, 7283, 7487, 7714, 7940,
    8190, 8459, 8736, 9043, 9361, 9712, 10088, 10491, 10922
};

static hi_void cmos_again_calc_table(hi_vi_pipe viPipe, hi_u32 *pu32AgainLin, hi_u32 *pu32AgainDb)
{
    hi_u32 i;
    SNS_CHECK_PIPE_VOID(viPipe);
    CMOS_CHECK_POINTER_VOID(pu32AgainLin);
    CMOS_CHECK_POINTER_VOID(pu32AgainDb);
    if (*pu32AgainLin >= ad_gain_table[IMX219_AGAIN_TBL_RANGE - 1]) {
        *pu32AgainLin = ad_gain_table[IMX219_AGAIN_TBL_RANGE - 1];
        *pu32AgainDb = IMX219_AGAIN_TBL_RANGE - 1;
        goto calc_table_end;
    }
    for (i = 1; i < IMX219_AGAIN_TBL_RANGE; i++) {
        if (*pu32AgainLin < ad_gain_table[i]) {
            *pu32AgainLin = ad_gain_table[i - 1];
            *pu32AgainDb = i - 1;
            goto calc_table_end;
        }
    }
calc_table_end:
    return;
}

static hi_void cmos_gains_update(hi_vi_pipe viPipe, hi_u32 u32Again, hi_u32 u32Dgain)
{
    hi_isp_sns_state *pstSnsState = HI_NULL;
    SNS_CHECK_PIPE_VOID(viPipe);
    IMX219_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);

    // Again
    /* array index 0 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[0].data = LOW_8BITS(u32Again);
    // Dgain
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[1].data = HIGH_8BITS(u32Dgain);
    /* array index 2 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[2].data = LOW_8BITS(u32Dgain);
    return;
}

static hi_s32 cmos_init_ae_exp_function(hi_isp_ae_sensor_exp_func *pstExpFuncs)
{
    hi_s32 ret;
    CMOS_CHECK_POINTER(pstExpFuncs);
    memset(pstExpFuncs, 0, sizeof(hi_isp_ae_sensor_exp_func));
    pstExpFuncs->pfn_cmos_get_ae_default    = cmos_get_ae_default;
    pstExpFuncs->pfn_cmos_fps_set           = cmos_fps_set;
    pstExpFuncs->pfn_cmos_slow_framerate_set = cmos_slow_framerate_set;
    pstExpFuncs->pfn_cmos_inttime_update    = cmos_inttime_update;
    pstExpFuncs->pfn_cmos_gains_update      = cmos_gains_update;
    pstExpFuncs->pfn_cmos_again_calc_table  = cmos_again_calc_table;
    pstExpFuncs->pfn_cmos_dgain_calc_table  = NULL;
    return HI_SUCCESS;
}

/* awb static param for Fuji Lens New IR_Cut */
#define CALIBRATE_STATIC_TEMP                         4950U
#define CALIBRATE_STATIC_WB_R_GAIN                    377U
#define CALIBRATE_STATIC_WB_GR_GAIN                   0X100
#define CALIBRATE_STATIC_WB_GB_GAIN                   0x100
#define CALIBRATE_STATIC_WB_B_GAIN                    336U

/* Calibration results for Auto WB Planck */
#define CALIBRATE_AWB_P1                              (167)
#define CALIBRATE_AWB_P2                              (-131)
#define CALIBRATE_AWB_Q1                              (-220)
#define CALIBRATE_AWB_A1                              (322690)
#define CALIBRATE_AWB_B1                              (128)
#define CALIBRATE_AWB_C1                              (-278581)

/* Rgain and Bgain of the golden sample */
#define GOLDEN_RGAIN                                  0U
#define GOLDEN_BGAIN                                  0U
static hi_s32 cmos_get_awb_default(hi_vi_pipe viPipe,
    hi_isp_awb_sensor_default *pstAwbSnsDft)
{
    hi_s32 ret;
    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstAwbSnsDft);
    memset(pstAwbSnsDft, 0, sizeof(hi_isp_awb_sensor_default));
    pstAwbSnsDft->wb_ref_temp = 5120; /* 5120 */
    pstAwbSnsDft->gain_offset[0] = CALIBRATE_STATIC_WB_R_GAIN; /* array index 0 */
    pstAwbSnsDft->gain_offset[1] = CALIBRATE_STATIC_WB_GR_GAIN; /* array index 1 */
    pstAwbSnsDft->gain_offset[2] = CALIBRATE_STATIC_WB_GB_GAIN; /* array index 2 */
    pstAwbSnsDft->gain_offset[3] = CALIBRATE_STATIC_WB_B_GAIN; /* array index 3 */
    pstAwbSnsDft->wb_para[0] = CALIBRATE_AWB_P1; /* array index 0 */
    pstAwbSnsDft->wb_para[1] = CALIBRATE_AWB_P2; /* array index 1 */
    pstAwbSnsDft->wb_para[2] = CALIBRATE_AWB_Q1; /* array index 2 */
    pstAwbSnsDft->wb_para[3] = CALIBRATE_AWB_A1; /* array index 3 */
    pstAwbSnsDft->wb_para[4] = CALIBRATE_AWB_B1; /* array index 4 */
    pstAwbSnsDft->wb_para[5] = CALIBRATE_AWB_C1; /* array index 5 */
    pstAwbSnsDft->golden_rgain = GOLDEN_RGAIN; // GOLDEN_RGAIN;
    pstAwbSnsDft->golden_bgain = GOLDEN_BGAIN; // GOLDEN_BGAIN;
    memcpy(&pstAwbSnsDft->ccm, &g_stAwbCcm_NormalLens, sizeof(hi_isp_awb_ccm));
    memcpy(&pstAwbSnsDft->agc_tbl, &ST_AWB_AGC_TABLE, sizeof(hi_isp_awb_agc_table));
    pstAwbSnsDft->init_rgain = g_au16InitWBGain[viPipe][0];
    pstAwbSnsDft->init_ggain = g_au16InitWBGain[viPipe][1];
    pstAwbSnsDft->init_bgain = g_au16InitWBGain[viPipe][2]; /* array index 2 */
    pstAwbSnsDft->sample_rgain = g_au16SampleRgain[viPipe];
    pstAwbSnsDft->sample_bgain = g_au16SampleBgain[viPipe];

    return HI_SUCCESS;
}

static hi_s32 cmos_init_awb_exp_function(hi_isp_awb_sensor_exp_func *pstExpFuncs)
{
    hi_s32 ret;
    CMOS_CHECK_POINTER(pstExpFuncs);
    memset(pstExpFuncs, 0, sizeof(hi_isp_awb_sensor_exp_func));
    pstExpFuncs->pfn_cmos_get_awb_default = cmos_get_awb_default;
    return HI_SUCCESS;
}

static const hi_isp_cmos_dng_color_param g_stDngColorParam = {
    {378, 256, 430},
    {439, 256, 439}
};

#define SENSOR_MAX_WIDTH    3264U
#define SNESOR_MAX_HIEGHT   2448U

static hi_void cmos_get_isp_dng_default(hi_isp_sns_state *pstSnsState, hi_isp_cmos_default *pstDef)
{
    hi_s32 ret;
    sns_unused(pstSnsState);
    memcpy(&pstDef->dng_color_param, &g_stDngColorParam, sizeof(hi_isp_cmos_dng_color_param));
    pstDef->sensor_mode.dng_raw_format.bits_per_sample = 10UL; /* 10bit */
    pstDef->sensor_mode.dng_raw_format.white_level = 1023U; /* max 1023 */
    pstDef->sensor_mode.dng_raw_format.default_scale.default_scale_h.denominator = 1;
    pstDef->sensor_mode.dng_raw_format.default_scale.default_scale_h.numerator = 1;
    pstDef->sensor_mode.dng_raw_format.default_scale.default_scale_v.denominator = 1;
    pstDef->sensor_mode.dng_raw_format.default_scale.default_scale_v.numerator = 1;
    /* pattern 2 */
    pstDef->sensor_mode.dng_raw_format.cfa_repeat_pattern_dim.repeat_pattern_dim_rows = 2U;
    /* pattern 2 */
    pstDef->sensor_mode.dng_raw_format.cfa_repeat_pattern_dim.repeat_pattern_dim_cols = 2U;
    pstDef->sensor_mode.dng_raw_format.blc_repeat_dim.blc_repeat_rows = 2U; /* pattern 2 */
    pstDef->sensor_mode.dng_raw_format.blc_repeat_dim.blc_repeat_cols = 2U; /* pattern 2 */
    pstDef->sensor_mode.dng_raw_format.cfa_layout = CFALAYOUT_TYPE_RECTANGULAR;
    pstDef->sensor_mode.dng_raw_format.cfa_plane_color[0] = 0U;
    pstDef->sensor_mode.dng_raw_format.cfa_plane_color[1] = 1U;
    pstDef->sensor_mode.dng_raw_format.cfa_plane_color[2] = 2U; /* index 2, CfaPlaneColor 2 */
    pstDef->sensor_mode.dng_raw_format.cfa_pattern[0] = 0U;
    pstDef->sensor_mode.dng_raw_format.cfa_pattern[1] = 1U;
    pstDef->sensor_mode.dng_raw_format.cfa_pattern[2] = 1U; /* index 2, CfaPattern 1 */
    pstDef->sensor_mode.dng_raw_format.cfa_pattern[3] = 2U; /* index 3, CfaPattern 2 */
    pstDef->sensor_mode.valid_dng_raw_format = HI_TRUE;
    return;
}

static hi_s32 cmos_get_isp_linear_default(hi_isp_cmos_default *pstDef)
{
    hi_s32 ret;
    pstDef->key.bit1_demosaic       = 1U;
    pstDef->demosaic              = &g_stIspDemosaic;
    pstDef->key.bit1_drc            = 1U;
    pstDef->drc                   = &g_stIspDRC;
    pstDef->key.bit1_gamma          = 1U;
    pstDef->gamma                 = &g_stIspGamma;
    pstDef->key.bit1_bayer_nr        = 1U;
    pstDef->bayer_nr               = &g_stIspBayerNr;
    pstDef->key.bit1_sharpen        = 1U;
    pstDef->sharpen               = &g_stIspYuvSharpen;
    pstDef->key.bit1_edge_mark       = 0U;
    pstDef->edge_mark              = &g_stIspEdgeMark;
    pstDef->key.bit1_ge             = 1U;
    pstDef->ge                    = &g_stIspGe;
    pstDef->key.bit1_anti_false_color = 1U;
    pstDef->anti_false_color        = &g_stIspAntiFalseColor;
    pstDef->key.bit1_ldci           = 1U;
    pstDef->ldci                  = &g_stIspLdci;
    memcpy(&pstDef->noise_calibration, &ST_ISP_NOISE_CALIB_RATIO, sizeof(hi_isp_cmos_noise_calibration));
    return HI_SUCCESS;
}

static hi_s32 cmos_get_isp_default(hi_vi_pipe viPipe, hi_isp_cmos_default *pstDef)
{
    hi_s32 ret;
    hi_isp_sns_state *pstSnsState = HI_NULL;
    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstDef);
    IMX219_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER(pstSnsState);
    memset(pstDef, 0, sizeof(hi_isp_cmos_default));
    pstDef->key.bit1_ca       = 1U;
    pstDef->ca              = &g_stIspCA;
    pstDef->key.bit1_clut     = 1U;
    pstDef->clut            = &g_stIspCLUT;
    pstDef->key.bit1_wdr      = 1U;
    pstDef->wdr             = &g_stIspWDR;
    pstDef->key.bit1_dpc      = 1U;
    pstDef->dpc             = &g_stCmosDpc;
    pstDef->key.bit1_lsc      = 1U;
    pstDef->lsc             = &g_stCmosLsc;
    switch (pstSnsState->wdr_mode) {
        case HI_WDR_MODE_NONE:
                ret = cmos_get_isp_linear_default(pstDef);
            break;
        default:
            break;
    }
    pstDef->sensor_max_resolution.max_width  = SENSOR_MAX_WIDTH;
    pstDef->sensor_max_resolution.max_height = SNESOR_MAX_HIEGHT;
    pstDef->sensor_mode.sensor_id = IMX219_ID;
    pstDef->sensor_mode.sensor_mode = pstSnsState->img_mode;
    cmos_get_isp_dng_default(pstSnsState, pstDef);
    return HI_SUCCESS;
}


static hi_s32 cmos_get_isp_black_level(hi_vi_pipe viPipe,
    hi_isp_cmos_black_level *pstBlackLevel)
{
    hi_s32 i;
    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstBlackLevel);
    /* Don't need to update black level when iso change */
    pstBlackLevel->update = HI_FALSE;
    if (pstBlackLevel->update == HI_TRUE) {
    } else {
        for (i = 0; i < HI_ISP_BAYER_CHN_NUM; i++) {
            pstBlackLevel->black_level[i] = 255U; /* 255:black_level */
        }
    }
    return HI_SUCCESS;
}

static hi_void cmos_set_pixel_detect(hi_vi_pipe viPipe, hi_bool bEnable)
{
    hi_isp_sns_state *pstSnsState = HI_NULL;
    SNS_CHECK_PIPE_VOID(viPipe);
    IMX219_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);

    if (bEnable) {
        /* Detect set 5fps */
        imx219_write_register(viPipe, IMX219_ANA_GAIN_GLOBAL_L, 0x0);
        imx219_write_register(viPipe, IMX219_ANA_GAIN_GLOBAL_H, 0x0);
        imx219_write_register(viPipe, IMX219_DIG_GAIN_GR_L, 0x0);
        imx219_write_register(viPipe, IMX219_DIG_GAIN_GR_H, 0xff);
    } else { /* setup for ISP 'normal mode' */
        pstSnsState->sync_init = HI_FALSE;
    }
    return;
}

static hi_void cmos_comm_sns_reg_info_init(hi_vi_pipe viPipe, hi_isp_sns_state *pstSnsState)
{
    hi_u32 i;
    SNS_CHECK_PIPE_VOID(viPipe);
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].sns_type = HI_ISP_SNS_I2C_TYPE;
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].com_bus.i2c_dev
        = g_aunImx219BusInfo[viPipe].i2c_dev;
    // DelayMax 3
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].cfg2_valid_delay_max = 3U;
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].reg_num = 7U; // RegNum 7
    for (i = 0; i < pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].reg_num; i++) {
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[i].update = HI_TRUE;
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[i].dev_addr
            = IMX219_I2C_ADDR;
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[i].addr_byte_num
            = IMX219_ADDR_BYTE;
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[i].data_byte_num
            = IMX219_DATA_BYTE;
    }

    // again
    /* index 0, reg_addr 0x157 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[0].reg_addr = 0x157;
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[0].delay_frm_num = 1U;
    // dgain
    /* index 1, reg_addr 0x158 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[1].reg_addr = 0x158;
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[1].delay_frm_num = 1U;
    /* index 2, reg_addr 0x159 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[2].reg_addr = 0x159;
    /* index 2 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[2].delay_frm_num = 1U;

    // time
    /* index 3, reg_addr 0x15A */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[3].reg_addr = 0x15A;
    /* index 3 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[3].delay_frm_num = 0U;
    /* index 4, reg_addr 0x15B */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[4].reg_addr = 0x15B;
    /* index 4 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[4].delay_frm_num = 0U;

    // frm length
    /* index 5, reg_addr 0x160 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[5].reg_addr = 0x160;
    /* index 5 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[5].delay_frm_num = 0U;
    /* index 6, reg_addr 0x161 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[6].reg_addr = 0x161;
    /* index 6 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[6].delay_frm_num = 0U;

    pstSnsState->sync_init = HI_TRUE;
}

static hi_s32 cmos_get_sns_regs_info(hi_vi_pipe viPipe, hi_isp_sns_regs_info *pstSnsRegsInfo)
{
    hi_s32 ret;
    hi_u32 i;
    hi_isp_sns_state *pstSnsState = HI_NULL;

    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstSnsRegsInfo);
    IMX219_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER(pstSnsState);
    if ((pstSnsState->sync_init == HI_FALSE) || (pstSnsRegsInfo->config == HI_FALSE)) {
        cmos_comm_sns_reg_info_init(viPipe, pstSnsState);
    } else {
        for (i = 0; i < pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].reg_num; i++) {
            if (pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[i].data ==
                pstSnsState->regs_info[ISP_SNS_SAVE_INFO_PRE_FRAME].i2c_data[i].data) {
                pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[i].update = HI_FALSE;
            } else {
                pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[i].update = HI_TRUE;
            }
        }
    }
    memcpy(pstSnsRegsInfo, &pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME],
        sizeof(hi_isp_sns_regs_info));
    memcpy(&pstSnsState->regs_info[ISP_SNS_SAVE_INFO_PRE_FRAME],
        &pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME], sizeof(hi_isp_sns_regs_info));
    pstSnsState->fl[1] = pstSnsState->fl[0];
    return HI_SUCCESS;
}

static hi_s32 cmos_set_image_mode(hi_vi_pipe viPipe,
    hi_isp_cmos_sensor_image_mode *pstSensorImageMode)
{
    hi_u8 u8SensorImageMode;
    hi_isp_sns_state *pstSnsState = HI_NULL;
    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstSensorImageMode);
    IMX219_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER(pstSnsState);

    u8SensorImageMode = pstSnsState->img_mode;
    pstSnsState->sync_init = HI_FALSE;
    if (pstSensorImageMode->sns_mode < IMX219_MODE_BUTT) {
        // The App directly specifies the image mode
        u8SensorImageMode = pstSensorImageMode->sns_mode;
        SNS_INFO_TRACE("IMX219 ISP Firmware specifies the image mode: %d\n",
            u8SensorImageMode);
    } else {
        return HI_FAILURE;
    }
    if ((pstSnsState->init == HI_TRUE) && (u8SensorImageMode == pstSnsState->img_mode)) {
        return HI_ISP_DO_NOT_NEED_SWITCH_IMAGEMODE; /* Don't need to switch SensorImageMode */
    }
    pstSnsState->img_mode = u8SensorImageMode;
    return HI_SUCCESS;
}

static hi_void sensor_global_init(hi_vi_pipe viPipe)
{
    hi_s32 ret;
    hi_isp_sns_state *pstSnsState = HI_NULL;
    SNS_CHECK_PIPE_VOID(viPipe);
    IMX219_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);
    pstSnsState->wdr_mode = HI_WDR_MODE_NONE;
    pstSnsState->init = HI_FALSE;
    pstSnsState->sync_init = HI_FALSE;
    pstSnsState->fl_std = 0xa55; /* 0xa55 fl_std */
    pstSnsState->img_mode = IMX219_24M_RAW10_MODE;
    pstSnsState->fl[0] = pstSnsState->fl_std;
    pstSnsState->fl[1] = pstSnsState->fl_std;
    memset(&pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME],
        0, sizeof(hi_isp_sns_regs_info));
    memset(&pstSnsState->regs_info[ISP_SNS_SAVE_INFO_PRE_FRAME],
        0, sizeof(hi_isp_sns_regs_info));
}

static hi_s32 cmos_set_wdr_mode(hi_vi_pipe viPipe, hi_u8 u8Mode)
{
    SNS_CHECK_PIPE_RETURN(viPipe);
    sns_unused(u8Mode);
    return HI_SUCCESS;
}

static hi_s32 cmos_init_sensor_exp_function(hi_isp_sensor_exp_func *pstSensorExpFunc)
{
    hi_s32 ret;
    CMOS_CHECK_POINTER(pstSensorExpFunc);
    memset(pstSensorExpFunc, 0, sizeof(hi_isp_sensor_exp_func));
    pstSensorExpFunc->pfn_cmos_sensor_init                  = imx219_init;
    pstSensorExpFunc->pfn_cmos_sensor_exit                  = imx219_exit;
    pstSensorExpFunc->pfn_cmos_sensor_global_init           = sensor_global_init;
    pstSensorExpFunc->pfn_cmos_set_image_mode               = cmos_set_image_mode;
    pstSensorExpFunc->pfn_cmos_set_wdr_mode                 = cmos_set_wdr_mode;
    pstSensorExpFunc->pfn_cmos_get_isp_default              = cmos_get_isp_default;
    pstSensorExpFunc->pfn_cmos_get_isp_black_level          = cmos_get_isp_black_level;
    pstSensorExpFunc->pfn_cmos_set_pixel_detect             = cmos_set_pixel_detect;
    pstSensorExpFunc->pfn_cmos_get_sns_reg_info             = cmos_get_sns_regs_info;
    return HI_SUCCESS;
}

static hi_s32 imx219_set_bus_info(hi_vi_pipe viPipe, hi_isp_sns_commbus unSNSBusInfo)
{
    SNS_CHECK_PIPE_RETURN(viPipe);
    g_aunImx219BusInfo[viPipe].i2c_dev = unSNSBusInfo.i2c_dev;
    SNS_INFO_TRACE("Config IMX219 sensor on VI PIPE %d: I2C bus %d\n",
        viPipe, g_aunImx219BusInfo[viPipe].i2c_dev);
    return HI_SUCCESS;
}

static hi_s32 sensor_ctx_init(hi_vi_pipe viPipe)
{
    hi_s32 ret;
    hi_isp_sns_state *pastSnsStateCtx = HI_NULL;
    SNS_CHECK_PIPE_RETURN(viPipe);
    IMX219_SENSOR_GET_CTX(viPipe, pastSnsStateCtx);
    if (pastSnsStateCtx == HI_NULL) {
        pastSnsStateCtx = (hi_isp_sns_state *)malloc(sizeof(hi_isp_sns_state));
        if (pastSnsStateCtx == HI_NULL) {
            SNS_ERR_TRACE("Isp[%d] SnsCtx malloc memory failed!\n", viPipe);
            return HI_ERR_ISP_NO_MEM;
        }
    }
    memset(pastSnsStateCtx, 0, sizeof(hi_isp_sns_state));
    IMX219_SENSOR_SET_CTX(viPipe, pastSnsStateCtx);
    return HI_SUCCESS;
}

static hi_void sensor_ctx_exit(hi_vi_pipe viPipe)
{
    hi_isp_sns_state *pastSnsStateCtx = HI_NULL;
    SNS_CHECK_PIPE_VOID(viPipe);
    IMX219_SENSOR_GET_CTX(viPipe, pastSnsStateCtx);
    SENSOR_FREE(pastSnsStateCtx);
    IMX219_SENSOR_RESET_CTX(viPipe);
}

static hi_s32 sensor_register_callback(hi_vi_pipe viPipe, hi_isp_3a_alg_lib *pstAeLib,
    hi_isp_3a_alg_lib *pstAwbLib, hi_isp_3a_alg_lib *pstAfLib)
{
    hi_s32 s32Ret;
    hi_isp_sensor_register stIspRegister;
    hi_isp_ae_sensor_register  stAeRegister;
    hi_isp_awb_sensor_register stAwbRegister;
    hi_isp_sns_attr_info   stSnsAttrInfo;
    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstAeLib);
    CMOS_CHECK_POINTER(pstAwbLib);
    s32Ret = sensor_ctx_init(viPipe);
    if (s32Ret != HI_SUCCESS) {
        return HI_FAILURE;
    }

    stSnsAttrInfo.sensor_id = IMX219_ID;
    s32Ret  = cmos_init_sensor_exp_function(&stIspRegister.sns_exp);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx219 cmos_init_sensor_exp_function err\n");
        goto fail0;
    }

    s32Ret = hi_mpi_isp_sensor_reg_callback(viPipe, &stSnsAttrInfo, &stIspRegister);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx219 sensor register callback function failed with 0x%x!\n", (hi_u32)s32Ret);
        goto fail0;
    }

    s32Ret  = cmos_init_ae_exp_function(&stAeRegister.sns_exp);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx219 cmos_init_ae_exp_function err\n");
        goto fail1;
    }

    s32Ret = hi_mpi_ae_sensor_reg_callback(viPipe, pstAeLib, &stSnsAttrInfo, &stAeRegister);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx219 sensor register callback function to ae lib failed with 0x%x!\n", (hi_u32)s32Ret);
        goto fail1;
    }

    s32Ret  = cmos_init_awb_exp_function(&stAwbRegister.sns_exp);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx219 cmos_init_awb_exp_function err\n");
        goto fail2;
    }

    s32Ret = hi_mpi_awb_sensor_reg_callback(viPipe, pstAwbLib, &stSnsAttrInfo, &stAwbRegister);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx219 sensor register callback function to awb lib failed with 0x%x!\n", (hi_u32)s32Ret);
        goto fail2;
    }
    return HI_SUCCESS;

fail2:
    (hi_void)hi_mpi_ae_sensor_unreg_callback(viPipe, pstAeLib, IMX219_ID);
fail1:
    (hi_void)hi_mpi_isp_sensor_unreg_callback(viPipe, IMX219_ID);
fail0:
    sensor_ctx_exit(viPipe);
    return HI_FAILURE;
}

static hi_s32 sensor_unregister_callback(hi_vi_pipe viPipe, hi_isp_3a_alg_lib *pstAeLib,
    hi_isp_3a_alg_lib *pstAwbLib, hi_isp_3a_alg_lib *pstAfLib)
{
    hi_s32 s32Ret = HI_SUCCESS;
    hi_s32 result = HI_SUCCESS;
    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstAeLib);
    CMOS_CHECK_POINTER(pstAwbLib);
    s32Ret = hi_mpi_isp_sensor_unreg_callback(viPipe, IMX219_ID);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx219 sensor unregister callback function failed with 0x%x!\n", (hi_u32)s32Ret);
        result = HI_FAILURE;
    }
    s32Ret = hi_mpi_ae_sensor_unreg_callback(viPipe, pstAeLib, IMX219_ID);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx219 sensor unregister callback function to ae lib failed with 0x%x!\n", (hi_u32)s32Ret);
        result = HI_FAILURE;
    }
    s32Ret = hi_mpi_awb_sensor_unreg_callback(viPipe, pstAwbLib, IMX219_ID);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx219 sensor unregister callback function to awb lib failed with 0x%x!\n", (hi_u32)s32Ret);
        result = HI_FAILURE;
    }
    sensor_ctx_exit(viPipe);
    return result;
}

static hi_s32 sensor_set_init(hi_vi_pipe viPipe, hi_isp_init_attr *pstInitAttr)
{
    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstInitAttr);
    g_au32InitExposure[viPipe] = pstInitAttr->exposure;
    g_au32LinesPer500ms[viPipe] = pstInitAttr->lines_per500ms;
    g_au16InitWBGain[viPipe][0] = pstInitAttr->wb_r_gain;
    g_au16InitWBGain[viPipe][1] = pstInitAttr->wb_g_gain;
    g_au16InitWBGain[viPipe][2] = pstInitAttr->wb_b_gain; /* index 2 */
    g_au16SampleRgain[viPipe] = pstInitAttr->sample_r_gain;
    g_au16SampleBgain[viPipe] = pstInitAttr->sample_b_gain;
    return HI_SUCCESS;
}

hi_isp_sns_obj g_sns_imx219_obj = {
    .pfn_register_callback     = sensor_register_callback,
    .pfn_un_register_callback  = sensor_unregister_callback,
    .pfn_standby               = imx219_standby,
    .pfn_restart               = imx219_restart,
    .pfn_write_reg             = imx219_write_register,
    .pfn_read_reg              = imx219_read_register,
    .pfn_set_bus_info          = imx219_set_bus_info,
    .pfn_set_init              = sensor_set_init
};
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /*  End of #ifdef __cplusplus  */
