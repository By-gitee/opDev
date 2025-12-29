#include "sensor_management.h"
#include "sony_imx219_config.h"
#include "sony_imx477_config.h"

hi_s32 Sample_Set_Mipi_2lane(combo_dev_t MipiDev, combo_dev_attr_t* pstComboAttr)
{
    pstComboAttr->devno = MipiDev;
    switch(MipiDev){
        case 0:
            pstComboAttr->mipi_attr.lane_id[0]= 0;
            pstComboAttr->mipi_attr.lane_id[1]= 2;
            break;
        case 1:
            pstComboAttr->mipi_attr.lane_id[0]= 1;
            pstComboAttr->mipi_attr.lane_id[1]= 3;
            break;
        case 2:
            pstComboAttr->mipi_attr.lane_id[0]= 4;
            pstComboAttr->mipi_attr.lane_id[1]= 6;
            break;
        case 3:
            pstComboAttr->mipi_attr.lane_id[0]= 5;
            pstComboAttr->mipi_attr.lane_id[1]= 7;
            break;
        default:
            printf("Function %s() Error. MipiDev: %d\n", __FUNCTION__, MipiDev);
            return HI_FAILURE;
    }
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_GetComboAttrBySns(SAMPLE_SNS_TYPE_E enSnsType, combo_dev_t MipiDev, combo_dev_attr_t* pstComboAttr)
{
    hi_s32 ret;
    switch (enSnsType) {
        case SONY_IMX219_MIPI_3K_10BIT_NORMAL:
            memcpy(pstComboAttr, &MIPI_2lane_CHN0_SENSOR_IMX219_10BIT_24M_NOWDR_ATTR, sizeof(combo_dev_attr_t));
            ret = Sample_Set_Mipi_2lane(MipiDev, pstComboAttr);
            break;
        case SONY_IMX219_MIPI_1920x1080_10BIT_NORMAL:
        case SONY_IMX219_MIPI_1920x1080_10BIT_30FPS_NORMAL:
            memcpy(pstComboAttr, &MIPI_2lane_CHN0_SENSOR_IMX219_10BIT_1920x1080_NOWDR_ATTR, sizeof(combo_dev_attr_t));
            ret = Sample_Set_Mipi_2lane(MipiDev, pstComboAttr);
            break;
        case SONY_IMX477_MIPI_1920x1080_10BIT_DOL2_15FPS:
            memcpy(pstComboAttr, &MIPI_2lane_CHN0_SENSOR_IMX477_10BIT_1920x1080_DOL2_ATTR, sizeof(combo_dev_attr_t));
            pstComboAttr->devno = MipiDev;
            ret = Sample_Set_Mipi_2lane(MipiDev, pstComboAttr);
            break;
        default:
            printf("Function %s() Error. Undefined sensor type: %#x\n", __FUNCTION__, enSnsType);
            return HI_FAILURE;
    }
    return ret;
}

hi_s32 Sample_Comm_Vi_GetSnsRstSourceBySns(SAMPLE_SNS_TYPE_E enSnsType, combo_dev_t MipiDev, sns_rst_source_t* sns_reset_port)
{
    switch (enSnsType) {
        case SONY_IMX219_MIPI_3K_10BIT_NORMAL:
        case SONY_IMX219_MIPI_1920x1080_10BIT_NORMAL:
        case SONY_IMX219_MIPI_1920x1080_10BIT_30FPS_NORMAL:
        case SONY_IMX477_MIPI_1920x1080_10BIT_DOL2_15FPS:
        if (MipiDev == 0 || MipiDev == 1) {
            *sns_reset_port = 0;
        }
        if (MipiDev == 2 || MipiDev == 3) {
            *sns_reset_port = 1;
        }
            break;
        default:
            printf("Function %s() Error. Undefined sensor type: %#x\n", __FUNCTION__, enSnsType);
            return HI_FAILURE;
    }
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_GetSnsClkCfgBySns(SAMPLE_SNS_TYPE_E enSnsType, combo_dev_t MipiDev, sns_clk_cfg_t* sns_clk_cfg)
{
    switch (enSnsType) {
        case SONY_IMX219_MIPI_3K_10BIT_NORMAL:
        case SONY_IMX219_MIPI_1920x1080_10BIT_NORMAL:
        case SONY_IMX219_MIPI_1920x1080_10BIT_30FPS_NORMAL:
        case SONY_IMX477_MIPI_1920x1080_10BIT_DOL2_15FPS:
            memcpy(sns_clk_cfg, &g_mipi_sns_clk_cfg_attr, sizeof(sns_clk_cfg_t));
            break;
        default:
            printf("Function %s() Error. Undefined sensor type: %#x\n", __FUNCTION__, enSnsType);
            return HI_FAILURE;
    }
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Isp_GetIspAttrBySns(SAMPLE_SNS_TYPE_E enSnsType, hi_isp_pub_attr* pstPubAttr)
{
    switch (enSnsType)
    {
        case SONY_IMX219_MIPI_3K_10BIT_NORMAL:
            memcpy(pstPubAttr, &ISP_PUB_ATTR_IMX219_24M_20FPS_raw10, sizeof(hi_isp_pub_attr));
            break;
        case SONY_IMX219_MIPI_1920x1080_10BIT_NORMAL:
            memcpy(pstPubAttr, &ISP_PUB_ATTR_IMX219_1920x1080_20FPS_raw10, sizeof(hi_isp_pub_attr));
            break;
        case SONY_IMX219_MIPI_1920x1080_10BIT_30FPS_NORMAL:
            memcpy(pstPubAttr, &ISP_PUB_ATTR_IMX219_1920x1080_30FPS_raw10, sizeof(hi_isp_pub_attr));
            break;
        case SONY_IMX477_MIPI_1920x1080_10BIT_DOL2_15FPS:
            memcpy(pstPubAttr, &ISP_PUB_ATTR_IMX477_1920x1080_10FPS_DOL2_RAW10,
                sizeof(hi_isp_pub_attr));
            break;
        default:
            printf("Function %s() Error. Undefined sensor type: %#x\n", __FUNCTION__, enSnsType);
            return HI_FAILURE;
    }

    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_GetDevAttrBySns(SAMPLE_SNS_TYPE_E enSnsType, hi_vi_dev_attr* pstViDevAttr)
{
    switch (enSnsType)
    {
        case SONY_IMX219_MIPI_3K_10BIT_NORMAL:
            memcpy(pstViDevAttr, &DEV_ATTR_IMX219_24M, sizeof(hi_vi_dev_attr));
            break;
        case SONY_IMX219_MIPI_1920x1080_10BIT_NORMAL:
        case SONY_IMX219_MIPI_1920x1080_10BIT_30FPS_NORMAL:
            memcpy(pstViDevAttr, &DEV_ATTR_IMX219_1920x1080, sizeof(hi_vi_dev_attr));
            break;
        case SONY_IMX477_MIPI_1920x1080_10BIT_DOL2_15FPS:
            memcpy(pstViDevAttr, &DEV_ATTR_IMX477_1920x1080_DOL2,
                sizeof(hi_vi_dev_attr));
            break;
        default:
            printf("Function %s() Error. Undefined sensor type: %#x\n", __FUNCTION__, enSnsType);
            return HI_FAILURE;
    }

    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_GetPipeAttrBySns(SAMPLE_SNS_TYPE_E enSnsType, hi_vi_pipe_attr* pstPipeAttr)
{
    switch (enSnsType)
    {
        case SONY_IMX219_MIPI_3K_10BIT_NORMAL:
            memcpy(pstPipeAttr, &IMX219_PIPE_ATTR_3264x2448_RAW10_420_3DNR_RFR, sizeof(hi_vi_pipe_attr));
            break;
        case SONY_IMX219_MIPI_1920x1080_10BIT_NORMAL:
        case SONY_IMX219_MIPI_1920x1080_10BIT_30FPS_NORMAL:
            memcpy(pstPipeAttr, &IMX219_PIPE_ATTR_1920x1080_RAW10_420_3DNR_RFR, sizeof(hi_vi_pipe_attr));
            break;
        case SONY_IMX477_MIPI_1920x1080_10BIT_DOL2_15FPS:
            memcpy(pstPipeAttr, &IMX477_PIPE_ATTR_1920x1080_RAW10_420_3DNR_RFR,
                sizeof(hi_vi_pipe_attr));
            break;
        default:
            return HI_FAILURE;
    }

    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_GetChnAttrBySns(SAMPLE_SNS_TYPE_E enSnsType, hi_vi_chn_attr* pstChnAttr)
{
    switch (enSnsType)
    {
        case SONY_IMX219_MIPI_3K_10BIT_NORMAL:
            memcpy(pstChnAttr, &IMX219_CHN_ATTR_3264x2448_420_SDR8_LINEAR, sizeof(hi_vi_chn_attr));
            break;
        case SONY_IMX219_MIPI_1920x1080_10BIT_NORMAL:
        case SONY_IMX219_MIPI_1920x1080_10BIT_30FPS_NORMAL:
            memcpy(pstChnAttr, &IMX219_CHN_ATTR_1920x1080_420_SDR8_LINEAR, sizeof(hi_vi_chn_attr));
            break;
        case SONY_IMX477_MIPI_1920x1080_10BIT_DOL2_15FPS:
            memcpy(pstChnAttr, &IMX477_CHN_ATTR_1920x1080_420_SDR8_LINEAR,
                sizeof(hi_vi_chn_attr));
            break;
        default:
            return HI_FAILURE;
    }
    return HI_SUCCESS;
}

const hi_isp_sns_obj* g_enSnsObj[MAX_SENSOR_NUM] = {
    &g_sns_imx219_obj,
    &g_sns_imx477_obj,
};

const hi_isp_sns_obj* Sample_Comm_Isp_GetSnsObj(hi_u32 u32SnsId)
{
    return g_enSnsObj[u32SnsId];
}