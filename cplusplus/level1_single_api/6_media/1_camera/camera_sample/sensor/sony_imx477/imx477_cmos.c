/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Function of imx477 adapt isp firmware
 * Author: Hisilicon multimedia software group
 * Create: 2023-03-06
 */

#include <stdio.h>
#include <string.h>
#include "hi_mpi_isp.h"
#include "hi_mpi_ae.h"
#include "hi_mpi_awb.h"
#include "imx477_cmos_ex.h"
#include "imx477_cmos.h"

#define IMX477_ID                    477U
#define IMX477_AGAIN_TBL_RANGE       979U

#define IMX477_SENSOR_SET_CTX(dev, pstCtx)   ((g_pastImx477[dev]) = (pstCtx))
#define IMX477_SENSOR_RESET_CTX(dev)         (g_pastImx477[dev] = HI_NULL)

#define ISP_SNS_SAVE_INFO_CUR_FRAME 0U
#define ISP_SNS_SAVE_INFO_PRE_FRAME 1U

static hi_isp_fswdr_mode g_genFSWDRMode[HI_ISP_MAX_PIPE_NUM] = {
    [0 ... HI_ISP_MAX_PIPE_NUM - 1] = HI_ISP_FSWDR_NORMAL_MODE
};

static hi_u32 g_au32InitExposure[HI_ISP_MAX_PIPE_NUM]  = {0};
static hi_u32 g_au32LinesPer500ms[HI_ISP_MAX_PIPE_NUM] = {0};

static hi_u16 g_au16InitWBGain[HI_ISP_MAX_PIPE_NUM][HI_ISP_RGB_CHN_NUM] = {{0}};
static hi_u16 g_au16SampleRgain[HI_ISP_MAX_PIPE_NUM] = {0};
static hi_u16 g_au16SampleBgain[HI_ISP_MAX_PIPE_NUM] = {0};

static hi_bool g_abAERouteExValid[HI_ISP_MAX_PIPE_NUM] = {0};
static hi_isp_ae_route g_astInitAERoute[HI_ISP_MAX_PIPE_NUM] = {{0}};
static hi_isp_ae_route_ex g_astInitAERouteEx[HI_ISP_MAX_PIPE_NUM] = {{0}};
static hi_isp_ae_route g_astInitAERouteSF[HI_ISP_MAX_PIPE_NUM] = {{0}};
static hi_isp_ae_route_ex g_astInitAERouteSFEx[HI_ISP_MAX_PIPE_NUM] = {{0}};

hi_isp_sns_commbus g_aunImx477BusInfo[HI_ISP_MAX_PIPE_NUM] = {
    [0] = { .i2c_dev = 0},
    [1 ... HI_ISP_MAX_PIPE_NUM - 1] = { .i2c_dev = -1}
};

/* Depart different sensor mode to get CCM/AWB param */
hi_u8 g_u8Sensor477ImageMode = IMX477_4056x3040_RAW12_25FPS;

hi_isp_sns_state *g_pastImx477[HI_ISP_MAX_PIPE_NUM] = {HI_NULL};

hi_isp_sns_commbus *imx477_get_bus_Info(hi_vi_pipe viPipe)
{
    SNS_CHECK_PIPE_RETURN(viPipe);
    return &g_aunImx477BusInfo[viPipe];
}

hi_isp_sns_state *imx477_get_ctx(hi_vi_pipe viPipe)
{
    SNS_CHECK_PIPE_RETURN(viPipe);
    return g_pastImx477[viPipe];
}

const IMX477_VIDEO_MODE_TBL_S g_astImx477ModeTbl[IMX477_MODE_BUTT] = {
    {3100,  3100,     25,    1,  4056,   3040,   0,  "IMX477_4056x3040_RAW12_25FPS"  },
    {3100,  3100,     15,    1,  4056,   3040,   1,  "IMX477_4056x3040_DOL2_RAW10_15fps"  },
    {3100,  2516,     45,    1,  3840,   2160,   2,  "IMX477_4K_45FPS"  },
    {3100,  1292,     60,    1,  1920,   1080,   3,  "IMX477_1920x1080_RAW12"  },
    {3100,  1230,     40,    1,  1920,   1080,   4,  "IMX477_1920x1080_DOL2_RAW10"  },
};

static hi_void cmos_get_ae_comm_default(hi_vi_pipe viPipe,
    hi_isp_ae_sensor_default *pstAeSnsDft, hi_isp_sns_state *pstSnsState)
{
    hi_u32 u32Fll;
    hi_float F32MaxFps;
    u32Fll    = g_astImx477ModeTbl[pstSnsState->img_mode].u32VertiLines;
    F32MaxFps = g_astImx477ModeTbl[pstSnsState->img_mode].f32MaxFps;
    pstSnsState->fl_std = u32Fll;
    pstAeSnsDft->full_lines_std = pstSnsState->fl_std;
    pstAeSnsDft->flicker_freq  = 50U * 256U;  /* light flicker freq: 50Hz, accuracy: 256  */
    /* 1000000 / (fll * maxfps) */
    pstAeSnsDft->hmax_times = (hi_u32)((1000000U) / (u32Fll * F32MaxFps));

    pstAeSnsDft->int_time_accu.accu_type = HI_ISP_AE_ACCURACY_LINEAR;
    pstAeSnsDft->int_time_accu.accuracy = 1.0f;

    pstAeSnsDft->init_exposure =
        /* if equal to 0 init 1000000 */
        (g_au32InitExposure[viPipe] == 0) ? 1000000U : g_au32InitExposure[viPipe];
    pstAeSnsDft->lines_per500ms = (g_au32LinesPer500ms[viPipe] == 0) ?
        (hi_u32)(((hi_u64)(u32Fll * F32MaxFps)) >> 1) : g_au32LinesPer500ms[viPipe];

    pstAeSnsDft->max_int_time = pstSnsState->fl_std - 20U; /*  sub 20 */
    pstAeSnsDft->int_time_accu.offset = 0U;

    pstAeSnsDft->again_accu.accu_type  = HI_ISP_AE_ACCURACY_TABLE;
    pstAeSnsDft->dgain_accu.accu_type = HI_ISP_AE_ACCURACY_LINEAR;
    pstAeSnsDft->dgain_accu.accuracy = 0.00390625; // 0.00390625:1/256;
    pstAeSnsDft->isp_dgain_shift         = 8U;  /* accuracy: 8 */
    pstAeSnsDft->min_isp_dgain_target = 1U << pstAeSnsDft->isp_dgain_shift;
    pstAeSnsDft->max_isp_dgain_target = 6U << pstAeSnsDft->isp_dgain_shift; /* max 6 */
    (hi_void)memcpy(&pstAeSnsDft->piris_attr,
                      &ST_PIRIS_ATTR, sizeof(hi_isp_piris_attr));
    pstAeSnsDft->max_iris_fno = HI_ISP_IRIS_F_NO_1_4;
    pstAeSnsDft->min_iris_fno = HI_ISP_IRIS_F_NO_5_6;
    pstAeSnsDft->ae_route_ex_valid = HI_FALSE;
    pstAeSnsDft->ae_route_attr.total_num    = 0U;
    pstAeSnsDft->ae_route_attr_ex.total_num  = 0U;
    return;
}

static hi_void cmos_get_ae_linear_default(hi_vi_pipe viPipe,
    hi_isp_ae_sensor_default *pstAeSnsDft, hi_isp_sns_state *pstSnsState)
{
    sns_unused(pstSnsState);
    pstAeSnsDft->hist_thresh[0] = 0xd;
    pstAeSnsDft->hist_thresh[1] = 0x28;
    pstAeSnsDft->hist_thresh[2] = 0x60; /* index 2 */
    pstAeSnsDft->hist_thresh[3] = 0x80; /* index 3 */

    pstAeSnsDft->ae_compensation = 0x38;
    pstAeSnsDft->ae_exp_mode = HI_ISP_AE_EXP_HIGHLIGHT_PRIOR;

    pstAeSnsDft->min_int_time = 8U; /* min 8 */
    pstAeSnsDft->max_int_time_target = 65515U; /* max 65515 */
    pstAeSnsDft->min_int_time_target = 8U; /* min 8 */

    pstAeSnsDft->max_again = 22795U; /* max 22795 */
    pstAeSnsDft->min_again = 1024U; /* min 1024 */
    pstAeSnsDft->max_again_target = pstAeSnsDft->max_again;
    pstAeSnsDft->min_again_target = pstAeSnsDft->min_again;

    pstAeSnsDft->max_dgain = 4095U; // 4095:max_dgain
    pstAeSnsDft->min_dgain = 256U; // 256:min_dgain
    pstAeSnsDft->max_dgain_target = pstAeSnsDft->max_dgain;
    pstAeSnsDft->min_dgain_target = pstAeSnsDft->min_dgain;

    pstAeSnsDft->ae_route_ex_valid = g_abAERouteExValid[viPipe];
    (hi_void)memcpy(&pstAeSnsDft->ae_route_attr,
                      &g_astInitAERoute[viPipe],  sizeof(hi_isp_ae_route));
    (hi_void)memcpy(&pstAeSnsDft->ae_route_attr_ex,
                      &g_astInitAERouteEx[viPipe], sizeof(hi_isp_ae_route_ex));
    return;
}

static hi_void cmos_get_ae_2to1_line_wdr_default(hi_vi_pipe viPipe,
    hi_isp_ae_sensor_default *pstAeSnsDft, hi_isp_sns_state *pstSnsState)
{
    sns_unused(pstSnsState);
    pstAeSnsDft->hist_thresh[0] = 0xC;
    pstAeSnsDft->hist_thresh[1] = 0x18;
    pstAeSnsDft->hist_thresh[2] = 0x60; /* index 2 */
    pstAeSnsDft->hist_thresh[3] = 0x80; /* index 3 */

    pstAeSnsDft->min_int_time = 8U; /* min 8 */
    pstAeSnsDft->max_int_time_target = 65515U; /* max 65515 */
    pstAeSnsDft->min_int_time_target = 8U; /* min 8 */

    pstAeSnsDft->max_again = 22795U; /* max 22795 */
    pstAeSnsDft->min_again = 1024U; /* min 1024 */
    pstAeSnsDft->max_again_target = pstAeSnsDft->max_again;
    pstAeSnsDft->min_again_target = pstAeSnsDft->min_again;

    pstAeSnsDft->max_dgain = 4095U; // 4095:max_dgain
    pstAeSnsDft->min_dgain = 256U; // 256:min_dgain
    pstAeSnsDft->max_dgain_target = pstAeSnsDft->max_dgain;
    pstAeSnsDft->min_dgain_target = pstAeSnsDft->min_dgain;

    pstAeSnsDft->init_exposure = (g_au32InitExposure[viPipe] != 0) ?
        g_au32InitExposure[viPipe] : 52000U; /* init 52000 */

    if (g_genFSWDRMode[viPipe] == HI_ISP_FSWDR_LONG_FRAME_MODE) {
        pstAeSnsDft->ae_compensation = 64U; /* AeCompensation 64 */
        pstAeSnsDft->ae_exp_mode = HI_ISP_AE_EXP_HIGHLIGHT_PRIOR;
    } else {
        pstAeSnsDft->ae_compensation = 30U; /* AeCompensation 30 */
        pstAeSnsDft->ae_exp_mode = HI_ISP_AE_EXP_HIGHLIGHT_PRIOR;
        pstAeSnsDft->prior_frame = HI_ISP_SHORT_FRAME;
        pstAeSnsDft->man_ratio_enable = HI_TRUE;
        pstAeSnsDft->arr_ratio[0] = 0x800;
        pstAeSnsDft->arr_ratio[1] = 0x40;
        pstAeSnsDft->arr_ratio[2] = 0x40; /* array index 2 */
    }
    pstAeSnsDft->ae_route_ex_valid = g_abAERouteExValid[viPipe];
    (hi_void)memcpy(&pstAeSnsDft->ae_route_attr,
                      &g_astInitAERoute[viPipe],  sizeof(hi_isp_ae_route));
    (hi_void)memcpy(&pstAeSnsDft->ae_route_attr_ex,
                      &g_astInitAERouteEx[viPipe],  sizeof(hi_isp_ae_route_ex));
    (hi_void)memcpy(&pstAeSnsDft->ae_route_sf_attr,
                      &g_astInitAERouteSF[viPipe], sizeof(hi_isp_ae_route));
    (hi_void)memcpy(&pstAeSnsDft->ae_route_sf_attr_ex,
                      &g_astInitAERouteSFEx[viPipe],  sizeof(hi_isp_ae_route_ex));
    return;
}

static hi_s32 cmos_get_ae_default(hi_vi_pipe viPipe, hi_isp_ae_sensor_default *pstAeSnsDft)
{
    hi_isp_sns_state *pstSnsState = HI_NULL;

    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstAeSnsDft);
    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER(pstSnsState);

    (hi_void)memset(&pstAeSnsDft->ae_route_attr, 0, sizeof(hi_isp_ae_route));

    cmos_get_ae_comm_default(viPipe, pstAeSnsDft, pstSnsState);
    switch (pstSnsState->wdr_mode) {
        case HI_WDR_MODE_NONE:
            cmos_get_ae_linear_default(viPipe, pstAeSnsDft, pstSnsState);
            break;
        case HI_WDR_MODE_2To1_LINE:
            cmos_get_ae_2to1_line_wdr_default(viPipe, pstAeSnsDft, pstSnsState);
            break;
        default:
            break;
    }

    return HI_SUCCESS;
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
    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);

    switch (pstSnsState->img_mode) {
        case IMX477_4056x3040_RAW12_25FPS:
        case IMX477_4056x3040_DOL2_RAW10_15fps:
        case IMX477_4K_45FPS:
        case IMX477_1920x1080_RAW12:
        case IMX477_1920x1080_DOL2_RAW10:
            u32Lines = g_astImx477ModeTbl[pstSnsState->img_mode].u32MaxVertiLines;
            u32LinesMax = g_astImx477ModeTbl[pstSnsState->img_mode].u32MaxVertiLines;
            f32MaxFps = g_astImx477ModeTbl[pstSnsState->img_mode].f32MaxFps;
            f32MinFps = g_astImx477ModeTbl[pstSnsState->img_mode].f32MinFps;
            if ((f32Fps <= f32MaxFps) && (f32Fps >= f32MinFps)) {
                u32Lines = (hi_u32)(u32LinesMax * f32MaxFps / SNS_DIV_0_TO_1_FLOAT(f32Fps));
            } else {
                SNS_ERR_TRACE("imx477 Not support fps: %f, should be in [%f,%f]\n",
                    f32Fps, f32MinFps, f32MaxFps);
                return;
            }
            break;
        default:
            SNS_ERR_TRACE("imx477 Not support ImgMode: %d, should less than %d\n",
                pstSnsState->img_mode, IMX477_MODE_BUTT);
            return;
    }
    SNS_INFO_TRACE("imx477 Fps: %f\n", f32Fps);
        /* array index 5 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[1].data
        = HIGH_8BITS(u32Lines);
        /* array index 6 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[0].data
        = LOW_8BITS(u32Lines);

    pstSnsState->fl_std = u32Lines;

    pstAeSnsDft->fps = f32Fps;
    pstAeSnsDft->lines_per500ms = (hi_u32)(pstSnsState->fl_std * f32Fps / 2U); /* div2 */
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
    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);

    pstSnsState->fl[0] = u32FullLines;

    pstAeSnsDft->full_lines= pstSnsState->fl[0];
    /* array index 5 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[1].data =
        HIGH_8BITS(u32FullLines);
    /* array index 6 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[0].data =
        LOW_8BITS(u32FullLines);

    switch (pstSnsState->img_mode) {
        case IMX477_4056x3040_RAW12_25FPS:
        case IMX477_4056x3040_DOL2_RAW10_15fps:
        case IMX477_4K_45FPS:
        case IMX477_1920x1080_RAW12:
        case IMX477_1920x1080_DOL2_RAW10:
            pstAeSnsDft->max_int_time = pstSnsState->fl[0] - 20U; /* sub 20  */
            break;
        default:
            SNS_ERR_TRACE("imx477 Not support ImgMode: %d, should less than %d\n",
                pstSnsState->img_mode, IMX477_MODE_BUTT);
            return;
    }
    return;
}

static hi_void cmos_inttime_update_linear(hi_vi_pipe viPipe, hi_u32 u32IntTime)
{
    hi_isp_sns_state *pstSnsState = HI_NULL;

    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);

    // SET CORASE_INTEG_TIME
    /* array index 8 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[8].data
        = HIGH_8BITS(u32IntTime);
    /* array index 7 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[7].data
        = LOW_8BITS(u32IntTime);

    return;
}

static hi_void cmos_inttime_update_2to1_line(hi_vi_pipe viPipe, hi_u32 u32IntTime)
{
    hi_isp_sns_state *pstSnsState = HI_NULL;
    static hi_bool bFirst[HI_ISP_MAX_PIPE_NUM] = {[0 ...(HI_ISP_MAX_PIPE_NUM - 1)] = 1};

    static hi_u32 u32ShortIntTime[HI_ISP_MAX_PIPE_NUM] = {0};
    static hi_u32 u32LongIntTime[HI_ISP_MAX_PIPE_NUM] = {0};

    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);

    if (bFirst[viPipe]) { /* short exposure */
        pstSnsState->wdr_int_time[0] = u32IntTime;
        u32ShortIntTime[viPipe] = u32IntTime;
        // SET CORASE_INTEG_TIME
        /* array index 8 */
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[8].data
            = HIGH_8BITS(u32ShortIntTime[viPipe]);
        /* array index 7 */
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[7].data
            = LOW_8BITS(u32ShortIntTime[viPipe]);
        bFirst[viPipe] = HI_FALSE;
    } else { /* long exposure */
        pstSnsState->wdr_int_time[1] = u32IntTime;
        u32LongIntTime[viPipe] = u32IntTime;
        /* array index 14 */
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[14].data
            = HIGH_8BITS(u32LongIntTime[viPipe]);
        /* array index 13 */
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[13].data
            = LOW_8BITS(u32LongIntTime[viPipe]);
        bFirst[viPipe] = HI_TRUE;
    }
    return;
}

/* while isp notify ae to update sensor regs, ae call these funcs. */
static hi_void cmos_inttime_update(hi_vi_pipe viPipe, hi_u32 u32IntTime)
{
    hi_isp_sns_state *pstSnsState = HI_NULL;

    SNS_CHECK_PIPE_VOID(viPipe);

    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);

    if (pstSnsState->wdr_mode == HI_WDR_MODE_2To1_LINE) {
        cmos_inttime_update_2to1_line(viPipe, u32IntTime);
    } else {
        cmos_inttime_update_linear(viPipe, u32IntTime);
    }
    return;
}

static const hi_u32 ad_gain_table[IMX477_AGAIN_TBL_RANGE] = {
    1024,  1025,  1026,  1027,  1028,  1029,  1030,  1031,  1032,  1033,
    1034,  1035,  1036,  1037,  1038,  1039,  1040,  1041,  1042,  1043,
    1044,  1045,  1046,  1047,  1048,  1049,  1050,  1051,  1052,  1053,
    1054,  1055,  1057,  1058,  1059,  1060,  1061,  1062,  1063,  1064,
    1065,  1066,  1067,  1068,  1069,  1071,  1072,  1073,  1074,  1075,
    1076,  1077,  1078,  1079,  1081,  1082,  1083,  1084,  1085,  1086,
    1087,  1088,  1089,  1091,  1092,  1093,  1094,  1095,  1096,  1097,
    1099,  1100,  1101,  1102,  1103,  1104,  1106,  1107,  1108,  1109,
    1110,  1111,  1113,  1114,  1115,  1116,  1117,  1119,  1120,  1121,
    1122,  1123,  1125,  1126,  1127,  1128,  1129,  1131,  1132,  1133,
    1134,  1136,  1137,  1138,  1139,  1140,  1142,  1143,  1144,  1145,
    1147,  1148,  1149,  1151,  1152,  1153,  1154,  1156,  1157,  1158,
    1159,  1161,  1162,  1163,  1165,  1166,  1167,  1168,  1170,  1171,
    1172,  1174,  1175,  1176,  1178,  1179,  1180,  1182,  1183,  1184,
    1186,  1187,  1188,  1190,  1191,  1192,  1194,  1195,  1197,  1198,
    1199,  1201,  1202,  1203,  1205,  1206,  1208,  1209,  1210,  1212,
    1213,  1215,  1216,  1217,  1219,  1220,  1222,  1223,  1224,  1226,
    1227,  1229,  1230,  1232,  1233,  1235,  1236,  1237,  1239,  1240,
    1242,  1243,  1245,  1246,  1248,  1249,  1251,  1252,  1254,  1255,
    1257,  1258,  1260,  1261,  1263,  1264,  1266,  1267,  1269,  1271,
    1272,  1274,  1275,  1277,  1278,  1280,  1281,  1283,  1285,  1286,
    1288,  1289,  1291,  1292,  1294,  1296,  1297,  1299,  1300,  1302,
    1304,  1305,  1307,  1309,  1310,  1312,  1314,  1315,  1317,  1318,
    1320,  1322,  1323,  1325,  1327,  1328,  1330,  1332,  1334,  1335,
    1337,  1339,  1340,  1342,  1344,  1346,  1347,  1349,  1351,  1353,
    1354,  1356,  1358,  1360,  1361,  1363,  1365,  1367,  1368,  1370,
    1372,  1374,  1376,  1377,  1379,  1381,  1383,  1385,  1387,  1388,
    1390,  1392,  1394,  1396,  1398,  1399,  1401,  1403,  1405,  1407,
    1409,  1411,  1413,  1415,  1416,  1418,  1420,  1422,  1424,  1426,
    1428,  1430,  1432,  1434,  1436,  1438,  1440,  1442,  1444,  1446,
    1448,  1450,  1452,  1454,  1456,  1458,  1460,  1462,  1464,  1466,
    1468,  1470,  1472,  1474,  1476,  1478,  1481,  1483,  1485,  1487,
    1489,  1491,  1493,  1495,  1497,  1500,  1502,  1504,  1506,  1508,
    1510,  1513,  1515,  1517,  1519,  1521,  1524,  1526,  1528,  1530,
    1533,  1535,  1537,  1539,  1542,  1544,  1546,  1548,  1551,  1553,
    1555,  1558,  1560,  1562,  1565,  1567,  1569,  1572,  1574,  1576,
    1579,  1581,  1583,  1586,  1588,  1591,  1593,  1596,  1598,  1600,
    1603,  1605,  1608,  1610,  1613,  1615,  1618,  1620,  1623,  1625,
    1628,  1630,  1633,  1635,  1638,  1640,  1643,  1646,  1648,  1651,
    1653,  1656,  1659,  1661,  1664,  1667,  1669,  1672,  1675,  1677,
    1680,  1683,  1685,  1688,  1691,  1693,  1696,  1699,  1702,  1705,
    1707,  1710,  1713,  1716,  1718,  1721,  1724,  1727,  1730,  1733,
    1736,  1738,  1741,  1744,  1747,  1750,  1753,  1756,  1759,  1762,
    1765,  1768,  1771,  1774,  1777,  1780,  1783,  1786,  1789,  1792,
    1795,  1798,  1801,  1804,  1807,  1811,  1814,  1817,  1820,  1823,
    1826,  1829,  1833,  1836,  1839,  1842,  1846,  1849,  1852,  1855,
    1859,  1862,  1865,  1869,  1872,  1875,  1879,  1882,  1885,  1889,
    1892,  1896,  1899,  1903,  1906,  1909,  1913,  1916,  1920,  1923,
    1927,  1931,  1934,  1938,  1941,  1945,  1949,  1952,  1956,  1959,
    1963,  1967,  1971,  1974,  1978,  1982,  1985,  1989,  1993,  1997,
    2001,  2004,  2008,  2012,  2016,  2020,  2024,  2028,  2032,  2036,
    2040,  2044,  2048,  2052,  2056,  2060,  2064,  2068,  2072,  2076,
    2080,  2084,  2088,  2092,  2097,  2101,  2105,  2109,  2114,  2118,
    2122,  2126,  2131,  2135,  2139,  2144,  2148,  2153,  2157,  2162,
    2166,  2170,  2175,  2179,  2184,  2189,  2193,  2198,  2202,  2207,
    2212,  2216,  2221,  2226,  2231,  2235,  2240,  2245,  2250,  2255,
    2259,  2264,  2269,  2274,  2279,  2284,  2289,  2294,  2299,  2304,
    2309,  2314,  2319,  2325,  2330,  2335,  2340,  2345,  2351,  2356,
    2361,  2366,  2372,  2377,  2383,  2388,  2394,  2399,  2404,  2410,
    2416,  2421,  2427,  2432,  2438,  2444,  2449,  2455,  2461,  2467,
    2473,  2478,  2484,  2490,  2496,  2502,  2508,  2514,  2520,  2526,
    2532,  2538,  2545,  2551,  2557,  2563,  2570,  2576,  2582,  2589,
    2595,  2601,  2608,  2614,  2621,  2628,  2634,  2641,  2647,  2654,
    2661,  2668,  2674,  2681,  2688,  2695,  2702,  2709,  2716,  2723,
    2730,  2737,  2744,  2752,  2759,  2766,  2774,  2781,  2788,  2796,
    2803,  2811,  2818,  2826,  2833,  2841,  2849,  2857,  2864,  2872,
    2880,  2888,  2896,  2904,  2912,  2920,  2928,  2937,  2945,  2953,
    2962,  2970,  2978,  2987,  2995,  3004,  3013,  3021,  3030,  3039,
    3048,  3057,  3066,  3075,  3084,  3093,  3102,  3111,  3120,  3130,
    3139,  3148,  3158,  3167,  3177,  3187,  3196,  3206,  3216,  3226,
    3236,  3246,  3256,  3266,  3276,  3287,  3297,  3307,  3318,  3328,
    3339,  3350,  3360,  3371,  3382,  3393,  3404,  3415,  3426,  3437,
    3449,  3460,  3472,  3483,  3495,  3506,  3518,  3530,  3542,  3554,
    3566,  3578,  3591,  3603,  3615,  3628,  3640,  3653,  3666,  3679,
    3692,  3705,  3718,  3731,  3744,  3758,  3771,  3785,  3799,  3813,
    3826,  3840,  3855,  3869,  3883,  3898,  3912,  3927,  3942,  3956,
    3971,  3986,  4002,  4017,  4032,  4048,  4064,  4080,  4096,  4112,
    4128,  4144,  4161,  4177,  4194,  4211,  4228,  4245,  4262,  4279,
    4297,  4315,  4332,  4350,  4369,  4387,  4405,  4424,  4443,  4462,
    4481,  4500,  4519,  4539,  4559,  4578,  4599,  4619,  4639,  4660,
    4681,  4702,  4723,  4744,  4766,  4788,  4809,  4832,  4854,  4877,
    4899,  4922,  4946,  4969,  4993,  5017,  5041,  5065,  5090,  5115,
    5140,  5165,  5190,  5216,  5242,  5269,  5295,  5322,  5349,  5377,
    5405,  5433,  5461,  5489,  5518,  5548,  5577,  5607,  5637,  5667,
    5698,  5729,  5761,  5793,  5825,  5857,  5890,  5924,  5957,  5991,
    6026,  6061,  6096,  6132,  6168,  6204,  6241,  6278,  6316,  6355,
    6393,  6432,  6472,  6512,  6553,  6594,  6636,  6678,  6721,  6765,
    6808,  6853,  6898,  6944,  6990,  7037,  7084,  7133,  7182,  7231,
    7281,  7332,  7384,  7436,  7489,  7543,  7598,  7653,  7710,  7767,
    7825,  7884,  7943,  8004,  8065,  8128,  8192,  8256,  8322,  8388,
    8456,  8525,  8594,  8665,  8738,  8811,  8886,  8962,  9039,  9118,
    9198,  9279,  9362,  9446,  9532,  9619,  9709,  9799,  9892,  9986,
    10082, 10180, 10280, 10381, 10485, 10591, 10699, 10810, 10922, 11037,
    11155, 11275, 11397, 11522, 11650, 11781, 11915, 12052, 12192, 12336,
    12483, 12633, 12787, 12945, 13107, 13273, 13443, 13617, 13797, 13981,
    14169, 14364, 14563, 14768, 14979, 15196, 15420, 15650, 15887, 16131,
    16384, 16644, 16912, 17189, 17476, 17772, 18078, 18396, 18724, 19065,
    19418, 19784, 20164, 20560, 20971, 21399, 21845, 22310, 22795
};

static hi_void cmos_again_calc_table(hi_vi_pipe viPipe,
    hi_u32 *pu32AgainLin, hi_u32 *pu32AgainDb)
{
    hi_u32 i;

    SNS_CHECK_PIPE_VOID(viPipe);
    CMOS_CHECK_POINTER_VOID(pu32AgainLin);
    CMOS_CHECK_POINTER_VOID(pu32AgainDb);

    if (*pu32AgainLin >= ad_gain_table[IMX477_AGAIN_TBL_RANGE - 1]) {
        *pu32AgainLin = ad_gain_table[IMX477_AGAIN_TBL_RANGE - 1];
        *pu32AgainDb = IMX477_AGAIN_TBL_RANGE - 1U;
        goto calc_table_end;
    }

    for (i = 1; i < IMX477_AGAIN_TBL_RANGE; i++) {
        if (*pu32AgainLin < ad_gain_table[i]) {
            *pu32AgainLin = ad_gain_table[i - 1];
            *pu32AgainDb = i - 1U;
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

    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);
    /* index 6   Dgain */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[6].data = 0U;
    // Again
    // I2C写的时刻
    /* array index 2 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[2].data
        = LOW_8BITS(u32Again);
    /* array index 3 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[3].data
        = HIGH_8BITS(u32Again);
    // Dgain
    /* Shift Right 4 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[4].data
        = LOW_8BITS(u32Dgain);
    /* array index 5 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[5].data
        = HIGH_8BITS(u32Dgain);

    if (pstSnsState->wdr_mode == HI_WDR_MODE_2To1_LINE) {
        // Again
        // I2C写的时刻
        /* array index 9 */
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[9].data
            = LOW_8BITS(u32Again);
        /* array index 10 */
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[10].data
            = HIGH_8BITS(u32Again);
        // Dgain
        /* Shift Right 11 */
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[11].data
            = LOW_8BITS(u32Dgain);
        /* array index 12 */
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[12].data
            = HIGH_8BITS(u32Dgain);
    }

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
#define CALIBRATE_STATIC_WB_R_GAIN                    800U
#define CALIBRATE_STATIC_WB_GR_GAIN                   0X100
#define CALIBRATE_STATIC_WB_GB_GAIN                   0x100
#define CALIBRATE_STATIC_WB_B_GAIN                    450U

/* Calibration results for Auto WB Planck */
#define CALIBRATE_AWB_P1                              (48)
#define CALIBRATE_AWB_P2                              (48)
#define CALIBRATE_AWB_Q1                              (-160)
#define CALIBRATE_AWB_A1                              (473053)
#define CALIBRATE_AWB_B1                              (128)
#define CALIBRATE_AWB_C1                              (-421420)

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

    pstAwbSnsDft->wb_ref_temp = 5120U; /* 5120 */

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

#define SENSOR_MAX_WIDTH     4056U
#define SNESOR_MAX_HIEGHT    3040U

static hi_void cmos_get_isp_dng_default(hi_isp_sns_state *pstSnsState, hi_isp_cmos_default *pstDef)
{
    hi_s32 ret;
    memcpy(&pstDef->dng_color_param, &g_stDngColorParam, sizeof(hi_isp_cmos_dng_color_param));

    switch (pstSnsState->img_mode) {
        case IMX477_4056x3040_RAW12_25FPS:
        case IMX477_4056x3040_DOL2_RAW10_15fps:
        case IMX477_4K_45FPS:
        case IMX477_1920x1080_RAW12:
        case IMX477_1920x1080_DOL2_RAW10:
            pstDef->sensor_mode.dng_raw_format.bits_per_sample = 12U; /* 12bit */
            pstDef->sensor_mode.dng_raw_format.white_level = 4095U; /* max 4095 */
            break;
        default:
            SNS_ERR_TRACE("imx477 Not support ImgMode: %d, should less than %d\n",
                pstSnsState->img_mode, IMX477_MODE_BUTT);
            return;
    }

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
    pstDef->key.bit1_sharpen        = 0U;
    pstDef->sharpen               = &g_stIspYuvSharpen;
    pstDef->key.bit1_edge_mark       = 0U;
    pstDef->edge_mark              = &g_stIspEdgeMark;
    pstDef->key.bit1_ge             = 1U;
    pstDef->ge                    = &g_stIspGe;
    pstDef->key.bit1_anti_false_color = 1U;
    pstDef->anti_false_color        = &g_stIspAntiFalseColor;
    pstDef->key.bit1_ldci           = 1U;
    pstDef->ldci                  = &g_stIspLdci;
    memcpy(&pstDef->noise_calibration,
        &ST_ISP_NOISE_CALIB_RATIO, sizeof(hi_isp_cmos_noise_calibration));
    return HI_SUCCESS;
}

static hi_s32 cmos_get_isp_2to1_line_default(hi_isp_cmos_default *pstDef)
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
    pstDef->key.bit1_sharpen        = 0U;
    pstDef->sharpen               = &g_stIspYuvSharpen;
    pstDef->key.bit1_edge_mark       = 0U;
    pstDef->edge_mark              = &g_stIspEdgeMark;
    pstDef->key.bit1_ge             = 1U;
    pstDef->ge                    = &g_stIspGe;
    pstDef->key.bit1_anti_false_color = 1U;
    pstDef->anti_false_color        = &g_stIspAntiFalseColor;
    pstDef->key.bit1_ldci           = 1U;
    pstDef->ldci                  = &g_stIspLdci;
    memcpy(&pstDef->noise_calibration,
        &ST_ISP_NOISE_CALIB_RATIO, sizeof(hi_isp_cmos_noise_calibration));
    return HI_SUCCESS;
}

static hi_s32 cmos_get_isp_default(hi_vi_pipe viPipe, hi_isp_cmos_default *pstDef)
{
    hi_s32 ret;
    hi_isp_sns_state *pstSnsState = HI_NULL;
    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstDef);
    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
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
    pstDef->lsc             = &ST_CMOS_LSC_NORMAL_LENS;

    switch (pstSnsState->wdr_mode) {
        case HI_WDR_MODE_NONE:
                ret = cmos_get_isp_linear_default(pstDef);
            break;
        case HI_WDR_MODE_2To1_LINE:
                ret = cmos_get_isp_2to1_line_default(pstDef);
            break;
        default:
            break;
    }

    pstDef->sensor_max_resolution.max_width  = SENSOR_MAX_WIDTH;
    pstDef->sensor_max_resolution.max_height = SNESOR_MAX_HIEGHT;
    pstDef->sensor_mode.sensor_id = IMX477_ID;
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
    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);

    if (bEnable) {
        imx477_write_register(viPipe, IMX477_ANA_GAIN_GLOBAL_L, 0x0);
        imx477_write_register(viPipe, IMX477_ANA_GAIN_GLOBAL_H, 0x0);

        imx477_write_register(viPipe, IMX477_DIG_GAIN_GR_L, 0x0);
        imx477_write_register(viPipe, IMX477_DIG_GAIN_GR_H, 0xff);
    } else { /* setup for ISP 'normal mode' */
        pstSnsState->sync_init = HI_FALSE;
    }
    return;
}

static hi_s32 cmos_set_wdr_mode(hi_vi_pipe viPipe, hi_u8 u8Mode)
{
    hi_isp_sns_state *pstSnsState = HI_NULL;
    SNS_CHECK_PIPE_RETURN(viPipe);
    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER(pstSnsState);

    pstSnsState->sync_init    = HI_FALSE;

    switch (u8Mode & 0x3F) {
        case HI_WDR_MODE_NONE:
            pstSnsState->wdr_mode = HI_WDR_MODE_NONE;
            SNS_INFO_TRACE("HI_WDR_MODE_NONE\n");
            break;
        case HI_WDR_MODE_2To1_LINE:
            pstSnsState->wdr_mode = HI_WDR_MODE_2To1_LINE;
            SNS_INFO_TRACE("HI_WDR_MODE_2To1_LINE\n");
            break;
        default:
            SNS_ERR_TRACE("do not support this mode%u,"
                "only support HI_WDR_MODE_NONE or HI_WDR_MODE_2To1_LINE!\n", u8Mode);
            return HI_FAILURE;
    }

    (hi_void)memset(pstSnsState->wdr_int_time, 0, sizeof(pstSnsState->wdr_int_time));

    return HI_SUCCESS;
}

static hi_void cmos_comm_sns_reg_info_init(hi_vi_pipe viPipe, hi_isp_sns_state *pstSnsState)
{
    hi_u32 i;
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].sns_type = HI_ISP_SNS_I2C_TYPE;
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].com_bus.i2c_dev =
        g_aunImx477BusInfo[viPipe].i2c_dev;
    // DelayMax 3
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].cfg2_valid_delay_max = 3U;

    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].reg_num = 9U; // RegNum 9
    if (pstSnsState->wdr_mode == HI_WDR_MODE_2To1_LINE) {
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].reg_num += 6U; /* RegNum add 6 */
    }
    for (i = 0; i < pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].reg_num; i++) {
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[i].update = HI_TRUE;
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[i].dev_addr = IMX477_I2C_ADDR;
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[i].addr_byte_num = IMX477_ADDR_BYTE;
        pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[i].data_byte_num = IMX477_DATA_BYTE;
    }

    /* fps cfg */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[0].reg_addr
        = IMX477_FRM_LENGTH_LINES_L;
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[0].delay_frm_num = 0U;
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[1].reg_addr
        = IMX477_FRM_LENGTH_LINES_H;
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[1].delay_frm_num = 0U;
    /* Again related */
    /* index 2 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[2].reg_addr
        = IMX477_ANA_GAIN_GLOBAL_L;
    /* index 2 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[2].delay_frm_num = 1u;
    /* index 3 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[3].reg_addr
        = IMX477_ANA_GAIN_GLOBAL_H;
    /* index 3 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[3].delay_frm_num = 1U;

    /* Dgain cfg */
    /* index 4 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[4].reg_addr
        = IMX477_DIG_GAIN_GR_L;
    /* index 4 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[4].delay_frm_num = 1u;
    /* index 5 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[5].reg_addr
        = IMX477_DIG_GAIN_GR_H;
    /* index 5 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[5].delay_frm_num = 1U;
    /* index 6 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[6].reg_addr
        = IMX477_DPGA_USE_GLOBAL_GAIN;
    /* index 6 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[6].delay_frm_num = 1u;
    /* index 7 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[7].delay_frm_num = 1u;
    /* index 7 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[7].reg_addr
        = IMX477_COARSE_INTEG_TIME_L;
    /* index 8 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[8].delay_frm_num = 1U;
    /* index 8 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[8].reg_addr
        = IMX477_COARSE_INTEG_TIME_H;

    pstSnsState->sync_init = HI_TRUE;
}

static hi_void cmos_2to1_line_wdr_sns_reg_info_init(hi_vi_pipe viPipe, hi_isp_sns_state *pstSnsState)
{
    SNS_CHECK_PIPE_VOID(viPipe);
    /* Again related */
    /* index 2 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[2].reg_addr
        = IMX477_DOL1_ANA_GAIN_GLOBAL_L;
    /* index 2 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[2].delay_frm_num = 1U;
    /* index 3 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[3].reg_addr
        = IMX477_DOL1_ANA_GAIN_GLOBAL_H;
    /* index 3 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[3].delay_frm_num = 1U;
    /* index 9 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[9].reg_addr
        = IMX477_DOL2_ANA_GAIN_GLOBAL_L;
    /* index 9 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[9].delay_frm_num = 1U;
    /* index 10 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[10].reg_addr
        = IMX477_DOL2_ANA_GAIN_GLOBAL_H;
    /* index 10 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[10].delay_frm_num = 1U;
    /* index 4 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[4].reg_addr
        = IMX477_DOL1_DIG_GAIN_GR_L;
    /* index 4 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[4].delay_frm_num = 1U;
    /* index 5 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[5].reg_addr
        = IMX477_DOL1_DIG_GAIN_GR_H;
    /* index 5 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[5].delay_frm_num = 1U;
    /* index 11 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[11].reg_addr
        = IMX477_DOL2_DIG_GAIN_GR_L;
    /* index 11 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[11].delay_frm_num = 1U;
    /* index 12 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[12].reg_addr
        = IMX477_DOL2_DIG_GAIN_GR_H;
    /* index 12 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[12].delay_frm_num = 1U;
    /* index 7 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[7].delay_frm_num = 1U;
    /* index 7 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[7].reg_addr
        = IMX477_DOL1_COARSE_INTEG_TIME_L;
    /* index 8 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[8].delay_frm_num = 1U;
    /* index 8 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[8].reg_addr
        = IMX477_DOL1_COARSE_INTEG_TIME_H;
    /* index 13 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[13].delay_frm_num = 1U;
    /* index 13 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[13].reg_addr
        = IMX477_DOL2_COARSE_INTEG_TIME_L;
    /* index 14 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[14].delay_frm_num = 1U;
    /* index 14 */
    pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME].i2c_data[14].reg_addr
        = IMX477_DOL2_COARSE_INTEG_TIME_H;

    return;
}

static hi_s32 cmos_get_sns_regs_info(hi_vi_pipe viPipe, hi_isp_sns_regs_info *pstSnsRegsInfo)
{
    hi_s32 ret;
    hi_u32 i;
    hi_isp_sns_state *pstSnsState = HI_NULL;

    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstSnsRegsInfo);
    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER(pstSnsState);

    if ((pstSnsState->sync_init == HI_FALSE) || (pstSnsRegsInfo->config == HI_FALSE)) {
        cmos_comm_sns_reg_info_init(viPipe, pstSnsState);
        if (pstSnsState->wdr_mode == HI_WDR_MODE_2To1_LINE) {
            cmos_2to1_line_wdr_sns_reg_info_init(viPipe, pstSnsState);
        }
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

    pstSnsRegsInfo->config = HI_FALSE;

    memcpy(pstSnsRegsInfo,
        &pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME], sizeof(hi_isp_sns_regs_info));

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
    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER(pstSnsState);

    u8SensorImageMode = pstSnsState->img_mode;
    pstSnsState->sync_init = HI_FALSE;

    if (pstSensorImageMode->sns_mode < IMX477_MODE_BUTT) {
        // The App directly specifies the image mode
        u8SensorImageMode = pstSensorImageMode->sns_mode;
        SNS_INFO_TRACE("IMX477 ISP Firmware specifies the image mode: %d\n", u8SensorImageMode);
    } else {
        return HI_FAILURE;
    }

    if ((pstSnsState->init == HI_TRUE) && (u8SensorImageMode == pstSnsState->img_mode)) {
        return HI_ISP_DO_NOT_NEED_SWITCH_IMAGEMODE; /* Don't need to switch SensorImageMode */
    }
    pstSnsState->img_mode = u8SensorImageMode;
    pstSnsState->fl[0] = pstSnsState->fl_std; // ?
    pstSnsState->fl[1] = pstSnsState->fl[0]; // ?

    return HI_SUCCESS;
}

static hi_void sensor_global_init(hi_vi_pipe viPipe)
{
    hi_s32 ret;
    hi_isp_sns_state *pstSnsState = HI_NULL;

    SNS_CHECK_PIPE_VOID(viPipe);
    IMX477_SENSOR_GET_CTX(viPipe, pstSnsState);
    CMOS_CHECK_POINTER_VOID(pstSnsState);

    pstSnsState->wdr_mode = HI_WDR_MODE_NONE;
    pstSnsState->init = HI_FALSE;
    pstSnsState->sync_init = HI_FALSE;

    pstSnsState->fl_std = 0xa55; /* 0xa55 fl_std */
    pstSnsState->img_mode = IMX477_4056x3040_RAW12_25FPS;
    pstSnsState->fl[0] = pstSnsState->fl_std;
    pstSnsState->fl[1] = pstSnsState->fl_std;

    memset(&pstSnsState->regs_info[ISP_SNS_SAVE_INFO_CUR_FRAME],
        0, sizeof(hi_isp_sns_regs_info));

    memset(&pstSnsState->regs_info[ISP_SNS_SAVE_INFO_PRE_FRAME],
        0, sizeof(hi_isp_sns_regs_info));
}

static hi_s32 cmos_init_sensor_exp_function(hi_isp_sensor_exp_func *pstSensorExpFunc)
{
    hi_s32 ret;
    CMOS_CHECK_POINTER(pstSensorExpFunc);
    memset(pstSensorExpFunc, 0, sizeof(hi_isp_sensor_exp_func));

    pstSensorExpFunc->pfn_cmos_sensor_init                  = imx477_init;
    pstSensorExpFunc->pfn_cmos_sensor_exit                  = imx477_exit;
    pstSensorExpFunc->pfn_cmos_sensor_global_init           = sensor_global_init;
    pstSensorExpFunc->pfn_cmos_set_image_mode               = cmos_set_image_mode;
    pstSensorExpFunc->pfn_cmos_set_wdr_mode                 = cmos_set_wdr_mode;
    pstSensorExpFunc->pfn_cmos_get_isp_default              = cmos_get_isp_default;
    pstSensorExpFunc->pfn_cmos_get_isp_black_level          = cmos_get_isp_black_level;
    pstSensorExpFunc->pfn_cmos_set_pixel_detect             = cmos_set_pixel_detect;
    pstSensorExpFunc->pfn_cmos_get_sns_reg_info             = cmos_get_sns_regs_info;

    return HI_SUCCESS;
}

static hi_s32 imx477_set_bus_info(hi_vi_pipe viPipe, hi_isp_sns_commbus unSNSBusInfo)
{
    SNS_CHECK_PIPE_RETURN(viPipe);
    g_aunImx477BusInfo[viPipe].i2c_dev = unSNSBusInfo.i2c_dev;
    SNS_INFO_TRACE("Config IMX477 sensor on VI PIPE %d: I2C bus %d\n",
        viPipe, g_aunImx477BusInfo[viPipe].i2c_dev);
    return HI_SUCCESS;
}

static hi_s32 sensor_ctx_init(hi_vi_pipe viPipe)
{
    hi_s32 ret;
    hi_isp_sns_state *pastSnsStateCtx = HI_NULL;

    SNS_CHECK_PIPE_RETURN(viPipe);
    IMX477_SENSOR_GET_CTX(viPipe, pastSnsStateCtx);

    if (pastSnsStateCtx == HI_NULL) {
        pastSnsStateCtx = (hi_isp_sns_state *)malloc(sizeof(hi_isp_sns_state));
        if (pastSnsStateCtx == HI_NULL) {
            SNS_ERR_TRACE("Isp[%d] SnsCtx malloc memory failed!\n", viPipe);
            return HI_ERR_ISP_NO_MEM;
        }
    }

    memset(pastSnsStateCtx, 0, sizeof(hi_isp_sns_state));

    IMX477_SENSOR_SET_CTX(viPipe, pastSnsStateCtx);

    return HI_SUCCESS;
}

static hi_void sensor_ctx_exit(hi_vi_pipe viPipe)
{
    hi_isp_sns_state *pastSnsStateCtx = HI_NULL;

    SNS_CHECK_PIPE_VOID(viPipe);
    IMX477_SENSOR_GET_CTX(viPipe, pastSnsStateCtx);
    SENSOR_FREE(pastSnsStateCtx);
    IMX477_SENSOR_RESET_CTX(viPipe);
}

static hi_s32 sensor_register_callback(hi_vi_pipe viPipe,
    hi_isp_3a_alg_lib *pstAeLib, hi_isp_3a_alg_lib *pstAwbLib)
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

    stSnsAttrInfo.sensor_id = IMX477_ID;

    s32Ret  = cmos_init_sensor_exp_function(&stIspRegister.sns_exp);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx477 cmos_init_sensor_exp_function err\n");
        goto fail0;
    }

    s32Ret = hi_mpi_isp_sensor_reg_callback(viPipe, &stSnsAttrInfo, &stIspRegister);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx477 sensor register callback function failed with 0x%x!\n", (hi_u32)s32Ret);
        goto fail0;
    }

    s32Ret  = cmos_init_ae_exp_function(&stAeRegister.sns_exp);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx477 cmos_init_ae_exp_function err\n");
        goto fail1;
    }

    s32Ret = hi_mpi_ae_sensor_reg_callback(viPipe, pstAeLib, &stSnsAttrInfo, &stAeRegister);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx477 sensor register callback function to ae lib failed with 0x%x!\n", (hi_u32)s32Ret);
        goto fail1;
    }

    s32Ret  = cmos_init_awb_exp_function(&stAwbRegister.sns_exp);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx477 cmos_init_awb_exp_function err\n");
        goto fail2;
    }

    s32Ret = hi_mpi_awb_sensor_reg_callback(viPipe, pstAwbLib, &stSnsAttrInfo, &stAwbRegister);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx477 sensor register callback function to awb lib failed with 0x%x!\n", (hi_u32)s32Ret);
        goto fail2;
    }

    return HI_SUCCESS;

fail2:
    (hi_void)hi_mpi_ae_sensor_unreg_callback(viPipe, pstAeLib, IMX477_ID);
fail1:
    (hi_void)hi_mpi_isp_sensor_unreg_callback(viPipe, IMX477_ID);
fail0:
    sensor_ctx_exit(viPipe);
    return HI_FAILURE;
}

static hi_s32 sensor_unregister_callback(hi_vi_pipe viPipe,
    hi_isp_3a_alg_lib *pstAeLib, hi_isp_3a_alg_lib *pstAwbLib)
{
    hi_s32 s32Ret = HI_SUCCESS;
    hi_s32 result = HI_SUCCESS;
    SNS_CHECK_PIPE_RETURN(viPipe);
    CMOS_CHECK_POINTER(pstAeLib);
    CMOS_CHECK_POINTER(pstAwbLib);

    s32Ret = hi_mpi_isp_sensor_unreg_callback(viPipe, IMX477_ID);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx477 sensor unregister callback function failed with 0x%x!\n", (hi_u32)s32Ret);
        result = HI_FAILURE;
    }

    s32Ret = hi_mpi_ae_sensor_unreg_callback(viPipe, pstAeLib, IMX477_ID);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx477 sensor unregister callback function to ae lib failed with 0x%x!\n", (hi_u32)s32Ret);
        result = HI_FAILURE;
    }

    s32Ret = hi_mpi_awb_sensor_unreg_callback(viPipe, pstAwbLib, IMX477_ID);
    if (s32Ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx477 sensor unregister callback function to awb lib failed with 0x%x!\n", (hi_u32)s32Ret);
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

    g_abAERouteExValid[viPipe] = pstInitAttr->ae_route_ex_valid;
    (hi_void)memcpy(&g_astInitAERoute[viPipe],
                      &pstInitAttr->ae_route, sizeof(hi_isp_ae_route));
    (hi_void)memcpy(&g_astInitAERouteEx[viPipe],
                      &pstInitAttr->ae_route_ex, sizeof(hi_isp_ae_route_ex));
    (hi_void)memcpy(&g_astInitAERouteSF[viPipe],
                      &pstInitAttr->ae_route_sf, sizeof(hi_isp_ae_route));
    (hi_void)memcpy(&g_astInitAERouteSFEx[viPipe],
                      &pstInitAttr->ae_route_sf_ex, sizeof(hi_isp_ae_route_ex));

    return HI_SUCCESS;
}

hi_isp_sns_obj g_sns_imx477_obj = {
    .pfn_register_callback    = sensor_register_callback,
    .pfn_un_register_callback  = sensor_unregister_callback,
    .pfn_standby             = imx477_standby,
    .pfn_restart             = imx477_restart,
    .pfn_write_reg            = imx477_write_register,
    .pfn_read_reg             = imx477_read_register,
    .pfn_set_bus_info          = imx477_set_bus_info,
    .pfn_set_init              = sensor_set_init,
    .pfn_identify_module       = imx477_identify_module
};