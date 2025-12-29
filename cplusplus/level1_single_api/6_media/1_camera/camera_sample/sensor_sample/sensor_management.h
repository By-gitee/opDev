#include "hi_mipi_rx.h"
#include "hi_mpi_isp.h"
#include "hi_common_sns.h"
#include "hi_common_3a.h"
#include "hi_mpi_vi.h"
#include "hi_sns_ctrl.h"
#define MAX_SENSOR_NUM 20

#ifndef SENSOR_MANAGEMENT_H
#define SENSOR_MANAGEMENT_H

typedef unsigned int combo_dev_t;
typedef unsigned int sns_rst_source_t;

typedef enum HISAMPLE_SNS_TYPE_E
{
    SONY_IMX219_MIPI_3K_10BIT_NORMAL = 0x60,
    SONY_IMX219_MIPI_1920x1080_10BIT_NORMAL = 0x62,
    SONY_IMX219_MIPI_1920x1080_10BIT_30FPS_NORMAL = 0x64,
    SONY_IMX477_MIPI_1920x1080_10BIT_DOL2_15FPS = 0x71,
    SAMPLE_SNS_TYPE_BUTT,
} SAMPLE_SNS_TYPE_E;

const hi_isp_sns_obj* Sample_Comm_Isp_GetSnsObj(hi_u32 u32SnsId);
hi_s32 Sample_Comm_Vi_GetComboAttrBySns(SAMPLE_SNS_TYPE_E enSnsType, combo_dev_t MipiDev, combo_dev_attr_t* pstComboAttr);
hi_s32 Sample_Comm_Isp_GetIspAttrBySns(SAMPLE_SNS_TYPE_E enSnsType, hi_isp_pub_attr* pstPubAttr);
hi_s32 Sample_Comm_Vi_GetDevAttrBySns(SAMPLE_SNS_TYPE_E enSnsType, hi_vi_dev_attr* pstViDevAttr);
hi_s32 Sample_Comm_Vi_GetChnAttrBySns(SAMPLE_SNS_TYPE_E enSnsType, hi_vi_chn_attr* pstChnAttr);
hi_s32 Sample_Comm_Vi_GetPipeAttrBySns(SAMPLE_SNS_TYPE_E enSnsType, hi_vi_pipe_attr* pstPipeAttr);
hi_s32 Sample_Comm_Vi_GetSnsRstSourceBySns(SAMPLE_SNS_TYPE_E enSnsType, combo_dev_t MipiDev, sns_rst_source_t* sns_reset_port);
hi_s32 Sample_Comm_Vi_GetSnsClkCfgBySns(SAMPLE_SNS_TYPE_E enSnsType, combo_dev_t MipiDev, sns_clk_cfg_t* sns_clk_cfg);
extern const hi_isp_sns_obj* g_enSnsObj[];

// Sensor Object
extern hi_isp_sns_obj g_sns_imx219_obj;
extern hi_isp_sns_obj g_sns_imx477_obj;
#endif
