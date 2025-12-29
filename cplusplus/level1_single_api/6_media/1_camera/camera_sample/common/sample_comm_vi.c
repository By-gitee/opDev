#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/poll.h>
#include <sys/time.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <sys/prctl.h>

#include "hi_mipi_rx.h"
#include "sample_comm.h"
#include "sample_comm_vi.h"
#include "sample_comm_isp.h"
#include "sensor_management.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define MIPI_DEV_NODE       "/dev/hi_mipi_rx"
#define MAX_FRAME_WIDTH     8192

static input_mode_t Sample_Comm_Vi_GetSnsInputMode(SAMPLE_SNS_TYPE_E enSnsType)
{
    input_mode_t enInputMode;
    enInputMode = INPUT_MODE_MIPI;

    return enInputMode;
}

hi_s32 Sample_Comm_Vi_ResetSns(sns_rst_source_t SnsDev)
{
    hi_s32 fd;
    hi_s32 s32Ret;

    fd = open(MIPI_DEV_NODE, O_RDWR);

    if (fd < 0)
    {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return -1;
    }

    s32Ret = ioctl(fd, HI_MIPI_RESET_SENSOR, &SnsDev);

    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("HI_MIPI_SET_HS_MODE failed\n");
    }

    close(fd);
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_SetMipiHsMode(lane_divide_mode_t enHsMode)
{
    hi_s32 fd;
    hi_s32 s32Ret;

    fd = open(MIPI_DEV_NODE, O_RDWR);
    if (fd < 0)
    {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return -1;
    }

    s32Ret = ioctl(fd, HI_MIPI_SET_HS_MODE, &enHsMode);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("HI_MIPI_SET_HS_MODE failed\n");
    }

    close(fd);
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_EnableMipiClock(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i = 0;
    hi_s32              s32ViNum = 0;
    hi_s32              s32Ret = HI_SUCCESS;
    hi_s32              fd;
    combo_dev_t           devno = 0;
    input_mode_t        enInputMode;
    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;

    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    fd = open(MIPI_DEV_NODE, O_RDWR);
    if (fd < 0)
    {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return HI_FAILURE;
    }

    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        devno = pstViInfo->stSnsInfo.MipiDev;
        enInputMode = Sample_Comm_Vi_GetSnsInputMode(pstViInfo->stSnsInfo.enSnsType);
        if (INPUT_MODE_SLVS == enInputMode)
        {
            s32Ret = ioctl(fd, HI_MIPI_ENABLE_SLVS_CLOCK, &devno);
        }
        else
        {
            s32Ret = ioctl(fd, HI_MIPI_ENABLE_MIPI_CLOCK, &devno);
        }

        if (HI_SUCCESS != s32Ret)
        {
            SAMPLE_PRT("MIPI_ENABLE_CLOCK %d failed\n", devno);
            goto EXIT;
        }
    }

EXIT:
    close(fd);
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_DisableMipiClock(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i = 0;
    hi_s32              s32ViNum = 0;
    hi_s32              s32Ret = HI_SUCCESS;
    hi_s32              fd;
    combo_dev_t           devno = 0;
    input_mode_t        enInputMode;
    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;

    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    fd = open(MIPI_DEV_NODE, O_RDWR);
    if (fd < 0)
    {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return HI_FAILURE;
    }

    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        devno = pstViInfo->stSnsInfo.MipiDev;
        enInputMode = Sample_Comm_Vi_GetSnsInputMode(pstViInfo->stSnsInfo.enSnsType);
        if (INPUT_MODE_SLVS == enInputMode)
        {
            s32Ret = ioctl(fd, HI_MIPI_DISABLE_SLVS_CLOCK, &devno);
        }
        else
        {
            s32Ret = ioctl(fd, HI_MIPI_DISABLE_MIPI_CLOCK, &devno);
        }

        if (HI_SUCCESS != s32Ret)
        {
            SAMPLE_PRT("MIPI_DISABLE_CLOCK %d failed\n", devno);
            goto EXIT;
        }
    }

EXIT:
    close(fd);
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_ClearMipi(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i = 0;
    hi_s32              s32ViNum = 0;
    hi_s32              s32Ret = HI_SUCCESS;
    hi_s32              fd;
    combo_dev_t         devno = 0;
    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;

    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    fd = open(MIPI_DEV_NODE, O_RDWR);
    if (fd < 0)
    {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return HI_FAILURE;
    }
    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        devno = pstViInfo->stSnsInfo.MipiDev;
        if (HI_SUCCESS != s32Ret)
        {
            SAMPLE_PRT("MIPI_CLEAR %d failed\n", devno);
            goto EXIT;
        }
    }

EXIT:
    close(fd);
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_ResetMipi(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i = 0;
    hi_s32              s32ViNum = 0;
    hi_s32              s32Ret = HI_SUCCESS;
    hi_s32              fd;
    combo_dev_t           devno = 0;
    input_mode_t        enInputMode;
    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;

    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }
    fd = open(MIPI_DEV_NODE, O_RDWR);

    if (fd < 0)
    {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return HI_FAILURE;
    }
    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        devno = pstViInfo->stSnsInfo.MipiDev;
        enInputMode = Sample_Comm_Vi_GetSnsInputMode(pstViInfo->stSnsInfo.enSnsType);
        if (INPUT_MODE_SLVS == enInputMode)
        {
            s32Ret = ioctl(fd, HI_MIPI_RESET_SLVS, &devno);
        }
        else
        {
            s32Ret = ioctl(fd, HI_MIPI_RESET_MIPI, &devno);
        }

        if (HI_SUCCESS != s32Ret)
        {
            SAMPLE_PRT("RESET_MIPI %d failed\n", devno);
            goto EXIT;
        }
    }

EXIT:
    close(fd);
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_UnresetMipi(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i = 0;
    hi_s32              s32ViNum = 0;
    hi_s32              s32Ret = HI_SUCCESS;
    hi_s32              fd;
    combo_dev_t           devno = 0;
    input_mode_t        enInputMode;
    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;

    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    fd = open(MIPI_DEV_NODE, O_RDWR);
    if (fd < 0)
    {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return HI_FAILURE;
    }
    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        devno = pstViInfo->stSnsInfo.MipiDev;
        enInputMode = Sample_Comm_Vi_GetSnsInputMode(pstViInfo->stSnsInfo.enSnsType);
        if (INPUT_MODE_SLVS == enInputMode)
        {
            s32Ret = ioctl(fd, HI_MIPI_UNRESET_SLVS, &devno);
        }
        else
        {
            s32Ret = ioctl(fd, HI_MIPI_UNRESET_MIPI, &devno);
        }

        if (HI_SUCCESS != s32Ret)
        {
            SAMPLE_PRT("UNRESET_MIPI %d failed\n", devno);
            goto EXIT;
        }
    }

EXIT:
    close(fd);
    return s32Ret;

}

hi_s32 sample_comm_vi_config_sensor_clk(SAMPLE_VI_CONFIG_S* viCfg)
{
    hi_s32 ret = HI_SUCCESS;
    if (viCfg == NULL) {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    int fd = open(MIPI_DEV_NODE, O_RDWR);
    if (fd < 0) {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return HI_FAILURE;
    }

    for (hi_s32 i = 0; i < viCfg->s32WorkingViNum; i++) {
        hi_s32 viNum = viCfg->as32WorkingViId[i];
        SAMPLE_VI_INFO_S* viInfo = &viCfg->astViInfo[viNum];
        sns_clk_cfg_t clk_cfg;
        Sample_Comm_Vi_GetSnsClkCfgBySns(viInfo->stSnsInfo.enSnsType,
            viInfo->stSnsInfo.MipiDev, &clk_cfg);
        ret = ioctl(fd, HI_MIPI_CONFIG_SENSOR_CLOCK, &clk_cfg);
        if (ret != 0) {
            printf("HI_MIPI_CONFIG_SENSOR_CLOCK. Clock Source: %d, Clock Frequency: %d Error\n",
                clk_cfg.clk_source, clk_cfg.clk_freq);
            close(fd);
            return HI_FAILURE;
        }
        printf("HI_MIPI_CONFIG_SENSOR_CLOCK. Clock Source: %d, Clock Frequency: %d OK\n",
            clk_cfg.clk_source, clk_cfg.clk_freq);
    }

    close(fd);
    return HI_SUCCESS;
}

hi_s32 sample_comm_vi_enable_sensor_clk(SAMPLE_VI_CONFIG_S* viCfg)
{
    hi_s32 ret = HI_SUCCESS;
    if (viCfg == NULL) {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    int fd = open(MIPI_DEV_NODE, O_RDWR);
    if (fd < 0) {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return HI_FAILURE;
    }

    for (hi_s32 i = 0; i < viCfg->s32WorkingViNum; i++) {
        hi_s32 viNum = viCfg->as32WorkingViId[i];
        SAMPLE_VI_INFO_S* viInfo = &viCfg->astViInfo[viNum];
        sns_clk_cfg_t clk_cfg;
        sns_clk_source_t clk_src;
        Sample_Comm_Vi_GetSnsClkCfgBySns(viInfo->stSnsInfo.enSnsType,
            viInfo->stSnsInfo.MipiDev, &clk_cfg);
        clk_src = clk_cfg.clk_source;
        ret = ioctl(fd, HI_MIPI_ENABLE_SENSOR_CLOCK, &clk_src);
        if (ret != 0) {
            printf("HI_MIPI_ENABLE_SENSOR_CLOCK Clock Source: %d Error\n", clk_src);
            close(fd);
            return HI_FAILURE;
        }
        printf("HI_MIPI_ENABLE_SENSOR_CLOCK Clock Source: %d OK\n", clk_src);
    }

    close(fd);
    return HI_SUCCESS;
}

hi_s32 sample_comm_vi_disable_sensor_clk(SAMPLE_VI_CONFIG_S* viCfg)
{
    hi_s32 ret = HI_SUCCESS;
    if (viCfg == NULL) {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    int fd = open(MIPI_DEV_NODE, O_RDWR);
    if (fd < 0) {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return HI_FAILURE;
    }

    for (hi_s32 i = 0; i < viCfg->s32WorkingViNum; i++) {
        hi_s32 viNum = viCfg->as32WorkingViId[i];
        SAMPLE_VI_INFO_S* viInfo = &viCfg->astViInfo[viNum];
        sns_clk_cfg_t clk_cfg;
        sns_clk_source_t clk_src;
        Sample_Comm_Vi_GetSnsClkCfgBySns(viInfo->stSnsInfo.enSnsType,
            viInfo->stSnsInfo.MipiDev, &clk_cfg);
        clk_src = clk_cfg.clk_source;
        ret = ioctl(fd, HI_MIPI_DISABLE_SENSOR_CLOCK, &clk_src);
        if (ret != 0) {
            printf("HI_MIPI_DISABLE_SENSOR_CLOCK Clock Source: %d Error\n", clk_src);
            close(fd);
            return HI_FAILURE;
        }
        printf("HI_MIPI_DISABLE_SENSOR_CLOCK Clock Source: %d OK\n", clk_src);
    }

    close(fd);
    return HI_SUCCESS;
}

hi_s32 sample_comm_vi_reset_sensor(SAMPLE_VI_CONFIG_S* viCfg)
{
    hi_s32 ret = HI_SUCCESS;
    if (viCfg == NULL) {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    int fd = open(MIPI_DEV_NODE, O_RDWR);
    if (fd < 0) {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return HI_FAILURE;
    }

    for (hi_s32 i = 0; i < viCfg->s32WorkingViNum; i++) {
        hi_s32 viNum = viCfg->as32WorkingViId[i];
        SAMPLE_VI_INFO_S* viInfo = &viCfg->astViInfo[viNum];
        sns_rst_source_t reset_port;
        Sample_Comm_Vi_GetSnsRstSourceBySns(viInfo->stSnsInfo.enSnsType,
            viInfo->stSnsInfo.MipiDev, &reset_port);
        ret = ioctl(fd, HI_MIPI_RESET_SENSOR, &reset_port);
        if (ret != 0) {
            printf("HI_MIPI_RESET_SENSOR Reset Port: %d Error\n", reset_port);
            close(fd);
            return HI_FAILURE;
        }
        // 对于 imx219 模组，特殊情况，如复位前仍在出图
        // 需要在复位后增加一段延时满足sensor复位时序要求
        usleep(100 * 1000); /* 1ms: 1000us * (100) */
        printf("HI_MIPI_RESET_SENSOR Reset Port: %d OK\n", reset_port);
    }

    close(fd);
    return HI_SUCCESS;
}

hi_s32 sample_comm_vi_unreset_sensor(SAMPLE_VI_CONFIG_S* viCfg)
{
    hi_s32 ret = HI_SUCCESS;
    if (viCfg == NULL) {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    int fd = open(MIPI_DEV_NODE, O_RDWR);
    if (fd < 0) {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return HI_FAILURE;
    }

    for (hi_s32 i = 0; i < viCfg->s32WorkingViNum; i++) {
        hi_s32 viNum = viCfg->as32WorkingViId[i];
        SAMPLE_VI_INFO_S* viInfo = &viCfg->astViInfo[viNum];
        sns_rst_source_t reset_port;
        Sample_Comm_Vi_GetSnsRstSourceBySns(viInfo->stSnsInfo.enSnsType,
            viInfo->stSnsInfo.MipiDev, &reset_port);
        ret = ioctl(fd, HI_MIPI_UNRESET_SENSOR, &reset_port);
        if (ret != 0) {
            printf("HI_MIPI_UNRESET_SENSOR failed. reset port: %d\n", reset_port);
            close(fd);
            return HI_FAILURE;
        }
        printf("HI_MIPI_UNRESET_SENSOR Unreset Port: %d OK\n", reset_port);
    }

    close(fd);
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_SetMipiAttr(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i = 0;
    hi_s32              s32ViNum = 0;
    hi_s32              s32Ret = HI_SUCCESS;
    hi_s32              fd;
    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;
    combo_dev_attr_t    stcomboDevAttr;

    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }
    fd = open(MIPI_DEV_NODE, O_RDWR);
    if (fd < 0)
    {
        SAMPLE_PRT("open hi_mipi_rx dev failed\n");
        return HI_FAILURE;
    }

    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        Sample_Comm_Vi_GetComboAttrBySns(pstViInfo->stSnsInfo.enSnsType,
            pstViInfo->stSnsInfo.MipiDev, &stcomboDevAttr);
        stcomboDevAttr.data_rate = MIPI_DATA_RATE_X1;
        SAMPLE_PRT("============= MipiDev %d, SetMipiAttr enWDRMode: %d\n",
            pstViInfo->stSnsInfo.MipiDev, pstViInfo->stDevInfo.enWDRMode);
        s32Ret = ioctl(fd, HI_MIPI_SET_DEV_ATTR, &stcomboDevAttr);
        if (HI_SUCCESS != s32Ret)
        {
            SAMPLE_PRT("MIPI_SET_DEV_ATTR failed\n");
            goto EXIT;
        }
    }

EXIT:
    close(fd);

    return s32Ret;
}

/*****************************************************************************
* function : init mipi
*****************************************************************************/
hi_s32 Sample_Comm_Vi_StartMIPI(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32 s32Ret = HI_SUCCESS;
    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    lane_divide_mode_t lane_divide_mode = pstViConfig->mipi_lane_divide_mode;
    s32Ret = Sample_Comm_Vi_SetMipiHsMode(lane_divide_mode);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("Sample_Comm_Vi_SetMipiHsMode failed!\n");
        return HI_FAILURE;
    }

    s32Ret = Sample_Comm_Vi_EnableMipiClock(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("Sample_Comm_Vi_EnableMipiClock failed!\n");
        return HI_FAILURE;
    }

    s32Ret = Sample_Comm_Vi_ResetMipi(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("Sample_Comm_Vi_ResetMipi failed!\n");
        return HI_FAILURE;
    }

    s32Ret = sample_comm_vi_disable_sensor_clk(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("sample_comm_vi_disable_sensor_clk failed!\n");
        return HI_FAILURE;
    }

    s32Ret = sample_comm_vi_config_sensor_clk(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("sample_comm_vi_config_sensor_clk failed!\n");
        return HI_FAILURE;
    }

    s32Ret = sample_comm_vi_enable_sensor_clk(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("sample_comm_vi_enable_sensor_clk failed!\n");
        return HI_FAILURE;
    }

    s32Ret = sample_comm_vi_reset_sensor(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("sample_comm_vi_reset_sensor failed!\n");
        return HI_FAILURE;
    }

    s32Ret = Sample_Comm_Vi_SetMipiAttr(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("Sample_Comm_Vi_SetMipiAttr failed!\n");
        return HI_FAILURE;
    }

    s32Ret = Sample_Comm_Vi_UnresetMipi(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("Sample_Comm_Vi_UnresetMipi failed!\n");
        return HI_FAILURE;
    }

    s32Ret = sample_comm_vi_unreset_sensor(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("sample_comm_vi_unreset_sensor failed!\n");
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_StopMIPI(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32 s32Ret = HI_SUCCESS;

    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    s32Ret = sample_comm_vi_reset_sensor(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("sample_comm_vi_reset_sensor failed!\n");
        return HI_FAILURE;
    }

    s32Ret = sample_comm_vi_disable_sensor_clk(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("sample_comm_vi_disable_sensor_clk failed!\n");
        return HI_FAILURE;
    }

    s32Ret = Sample_Comm_Vi_ResetMipi(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("Sample_Comm_Vi_ResetMipi failed!\n");
        return HI_FAILURE;
    }

    s32Ret = Sample_Comm_Vi_DisableMipiClock(pstViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("Sample_Comm_Vi_DisableMipiClock failed!\n");
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_SetParam(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i;
    hi_s32              s32ViNum;
    hi_s32              s32Ret;
    hi_vi_pipe             ViPipe;
    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;
    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_StartDev(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32              s32Ret;
    hi_vi_dev              ViDev;
    SAMPLE_SNS_TYPE_E    enSnsType;
    hi_vi_dev_attr       stViDevAttr;

    ViDev       = pstViInfo->stDevInfo.ViDev;
    enSnsType    = pstViInfo->stSnsInfo.enSnsType;
    SAMPLE_PRT("Function %s() enSnsType %d!\n", __FUNCTION__, enSnsType);
    Sample_Comm_Vi_GetDevAttrBySns(enSnsType, &stViDevAttr);
    SAMPLE_PRT("Function %s() hi_vi_intf_mode %d!\n", __FUNCTION__, stViDevAttr.intf_mode);
    SAMPLE_PRT("Function %s() hi_vi_scan_mode %d!\n", __FUNCTION__, stViDevAttr.scan_mode);
    stViDevAttr.wdr_attr.wdr_mode = pstViInfo->stDevInfo.enWDRMode;
    stViDevAttr.data_rate = HI_DATA_RATE_X1;

    SAMPLE_PRT("Function %s() ViDev %d!\n", __FUNCTION__, ViDev);
    s32Ret = hi_mpi_vi_set_dev_attr(ViDev, &stViDevAttr);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("HI_MPI_VI_SetDevAttr failed with %#x!\n", s32Ret);
        return HI_FAILURE;
    }
    s32Ret = hi_mpi_vi_enable_dev(ViDev);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("hi_mpi_vi_enable_dev failed with %#x!\n", s32Ret);
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_StopDev(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32 s32Ret;
    hi_vi_dev ViDev;
    ViDev   = pstViInfo->stDevInfo.ViDev;
    s32Ret  = hi_mpi_vi_disable_dev(ViDev);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("hi_mpi_vi_disable_dev failed with %#x!\n", s32Ret);
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_BindPipeDev(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32              i;
    hi_s32              s32PipeCnt = 0;
    hi_s32              s32Ret;
    hi_vi_dev_bind_pipe  stDevBindPipe = {0};

    // Max Pipe Num 4
    for (i = 0; i < 4; i++)
    {
        if (pstViInfo->stPipeInfo.aPipe[i] >= 0  && pstViInfo->stPipeInfo.aPipe[i] < HI_VI_MAX_PIPE_NUM)
        {
            stDevBindPipe.pipe_id[s32PipeCnt] = pstViInfo->stPipeInfo.aPipe[i];
            s32PipeCnt++;
            stDevBindPipe.num = s32PipeCnt;
        }
    }
    SAMPLE_PRT("Function %s() ViDev: %d\n", __FUNCTION__, pstViInfo->stDevInfo.ViDev);
    SAMPLE_PRT("Function %s() hi_vi_dev_bind_pipe.u32Num: %d\n", __FUNCTION__, stDevBindPipe.num);
    s32Ret = hi_mpi_vi_set_dev_bind_pipe(pstViInfo->stDevInfo.ViDev, &stDevBindPipe);

    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("hi_mpi_vi_set_dev_bind_pipe failed with %#x!\n", s32Ret);
        return HI_FAILURE;
    }

    return s32Ret;
}

static hi_s32 Sample_Comm_Vi_StartSingleViPipe(hi_vi_pipe ViPipe, hi_vi_pipe_attr* pstPipeAttr)
{
    hi_s32 s32Ret;

    s32Ret = hi_mpi_vi_create_pipe(ViPipe, pstPipeAttr);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("hi_mpi_vi_create_pipe failed with %#x!\n", s32Ret);
        return HI_FAILURE;
    }
    s32Ret = hi_mpi_vi_start_pipe(ViPipe);
    if (s32Ret != HI_SUCCESS)
    {
        hi_mpi_vi_destroy_pipe(ViPipe);
        SAMPLE_PRT("hi_mpi_vi_start_pipe failed with %#x!\n", s32Ret);
        return HI_FAILURE;
    }
    return s32Ret;
}


static hi_s32 Sample_Comm_Vi_ModeSwitchCreateSingleViPipe(hi_vi_pipe ViPipe, hi_vi_pipe_attr* pstPipeAttr)
{
    hi_s32 s32Ret;
    s32Ret = hi_mpi_vi_create_pipe(ViPipe, pstPipeAttr);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("hi_mpi_vi_create_pipe failed with %#x!\n", s32Ret);
        return HI_FAILURE;
    }
    return s32Ret;
}


static hi_s32 Sample_Comm_Vi_ModeSwitch_EnableSingleViPipe(hi_vi_pipe ViPipe, hi_vi_pipe_attr* pstPipeAttr)
{
    hi_s32 s32Ret;
    s32Ret = hi_mpi_vi_start_pipe(ViPipe);
    if (s32Ret != HI_SUCCESS)
    {
        hi_mpi_vi_destroy_pipe(ViPipe);
        SAMPLE_PRT("hi_mpi_vi_start_pipe failed with %#x!\n", s32Ret);
        return HI_FAILURE;
    }
    return s32Ret;
}


static hi_s32 Sample_Comm_Vi_StopSingleViPipe(hi_vi_pipe ViPipe)
{
    hi_s32  s32Ret;
    s32Ret = hi_mpi_vi_stop_pipe(ViPipe);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("hi_mpi_vi_stop_pipe failed with %#x!\n", s32Ret);
        return HI_FAILURE;
    }
    s32Ret = hi_mpi_vi_destroy_pipe(ViPipe);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("hi_mpi_vi_destroy_pipe failed with %#x!\n", s32Ret);
        return HI_FAILURE;
    }
    return s32Ret;
}


hi_s32 Sample_Comm_Vi_ModeSwitch_StartViPipe(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32          i, j;
    hi_s32          s32Ret = HI_SUCCESS;
    hi_vi_pipe         ViPipe;
    hi_vi_pipe_attr  stPipeAttr;

    for (i = 0; i < WDR_MAX_PIPE_NUM; i++)
    {
        if (pstViInfo->stPipeInfo.aPipe[i] >= 0  && pstViInfo->stPipeInfo.aPipe[i] < HI_VI_MAX_PIPE_NUM)
        {
            ViPipe = pstViInfo->stPipeInfo.aPipe[i];
            Sample_Comm_Vi_GetPipeAttrBySns(pstViInfo->stSnsInfo.enSnsType, &stPipeAttr);
            {
                s32Ret = Sample_Comm_Vi_ModeSwitchCreateSingleViPipe(ViPipe, &stPipeAttr);
                if (s32Ret != HI_SUCCESS)
                {
                    SAMPLE_PRT("Sample_Comm_Vi_StartSingleViPipe  %d failed!\n", ViPipe);
                    goto EXIT;
                }
            }

        }
    }
    return s32Ret;
EXIT:
    for (j = 0; j < i; j++)
    {
        ViPipe = j;
        Sample_Comm_Vi_StopSingleViPipe(ViPipe);
    }
    return s32Ret;
}


hi_s32 Sample_Comm_Vi_ModeSwitch_EnableViPipe(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32          i, j;
    hi_s32          s32Ret = HI_SUCCESS;
    hi_vi_pipe         ViPipe;
    hi_vi_pipe_attr  stPipeAttr;

    for (i = 0; i < WDR_MAX_PIPE_NUM; i++)
    {
        if (pstViInfo->stPipeInfo.aPipe[i] >= 0  && pstViInfo->stPipeInfo.aPipe[i] < HI_VI_MAX_PIPE_NUM)
        {
            ViPipe = pstViInfo->stPipeInfo.aPipe[i];
            Sample_Comm_Vi_GetPipeAttrBySns(pstViInfo->stSnsInfo.enSnsType, &stPipeAttr);
            {
                s32Ret = Sample_Comm_Vi_ModeSwitch_EnableSingleViPipe(ViPipe, &stPipeAttr);
                if (s32Ret != HI_SUCCESS)
                {
                    SAMPLE_PRT("Sample_Comm_Vi_StartSingleViPipe  %d failed!\n", ViPipe);
                    goto EXIT;
                }
            }

        }
    }
    return s32Ret;

EXIT:
    for (j = 0; j < i; j++)
    {
        ViPipe = j;
        Sample_Comm_Vi_StopSingleViPipe(ViPipe);
    }
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_StartViPipe(SAMPLE_VI_INFO_S* pstViInfo)
{
    SAMPLE_PRT("Enter Function %s()\n", __FUNCTION__);
    hi_s32          i, j;
    hi_s32          s32Ret = HI_SUCCESS;
    hi_vi_pipe         ViPipe;
    hi_vi_pipe_attr  stPipeAttr;

    for (i = 0; i < WDR_MAX_PIPE_NUM; i++)
    {
        if (pstViInfo->stPipeInfo.aPipe[i] >= 0  && pstViInfo->stPipeInfo.aPipe[i] < HI_VI_MAX_PIPE_NUM)
        {
            ViPipe = pstViInfo->stPipeInfo.aPipe[i];
            Sample_Comm_Vi_GetPipeAttrBySns(pstViInfo->stSnsInfo.enSnsType, &stPipeAttr);
            // 在线不压缩前面已防护
            stPipeAttr.compress_mode = pstViInfo->stPipeInfo.enCompressMode;
            printf("set pipe=%d compress mode=%u\n", i, stPipeAttr.compress_mode);
            s32Ret = Sample_Comm_Vi_StartSingleViPipe(ViPipe, &stPipeAttr);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_PRT("Sample_Comm_Vi_StartSingleViPipe  %d failed!\n", ViPipe);
                goto EXIT;
            }
        }
    }
    SAMPLE_PRT("Function %s() success\n", __FUNCTION__);
    return s32Ret;

EXIT:
    for (j = 0; j < i; j++)
    {
        ViPipe = j;
        Sample_Comm_Vi_StopSingleViPipe(ViPipe);
    }
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_StopViPipe(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32  i;
    hi_vi_pipe ViPipe;
    for (i = 0; i < WDR_MAX_PIPE_NUM; i++)
    {
        if (pstViInfo->stPipeInfo.aPipe[i] >= 0  && pstViInfo->stPipeInfo.aPipe[i] < HI_VI_MAX_PIPE_NUM)
        {
            ViPipe = pstViInfo->stPipeInfo.aPipe[i];
            Sample_Comm_Vi_StopSingleViPipe(ViPipe);
        }
    }
    return HI_SUCCESS;
}


hi_s32 Sample_Comm_Vi_ModeSwitch_StartViChn(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32              i;
    hi_bool             bNeedChn;
    hi_s32              s32Ret = HI_SUCCESS;
    hi_vi_pipe             ViPipe;
    hi_vi_chn              ViChn;
    hi_vi_chn_attr       stChnAttr;

    for (i = 0; i < WDR_MAX_PIPE_NUM; i++)
    {
        if (pstViInfo->stPipeInfo.aPipe[i] >= 0  && pstViInfo->stPipeInfo.aPipe[i] < HI_VI_MAX_PIPE_NUM)
        {
            ViPipe = pstViInfo->stPipeInfo.aPipe[i];
            ViChn  = pstViInfo->stChnInfo.ViChn;
            Sample_Comm_Vi_GetChnAttrBySns(pstViInfo->stSnsInfo.enSnsType, &stChnAttr);
            stChnAttr.dynamic_range = pstViInfo->stChnInfo.enDynamicRange;
            stChnAttr.video_format  = pstViInfo->stChnInfo.enVideoFormat;
            stChnAttr.pixel_format  = pstViInfo->stChnInfo.enPixFormat;
            stChnAttr.compress_mode = pstViInfo->stChnInfo.enCompressMode;
            if (HI_WDR_MODE_NONE == pstViInfo->stDevInfo.enWDRMode)
            {
                bNeedChn = HI_TRUE;
            }
            else
            {
                bNeedChn = (i > 0) ? HI_FALSE : HI_TRUE;
            }

            if (bNeedChn)
            {
                s32Ret = hi_mpi_vi_set_chn_attr(ViPipe, ViChn, &stChnAttr);

                if (s32Ret != HI_SUCCESS)
                {
                    SAMPLE_PRT("hi_mpi_vi_set_chn_attr failed with %#x!\n", s32Ret);
                    return HI_FAILURE;
                }
            }
        }
    }

    return s32Ret;
}
hi_s32 Sample_Comm_Vi_StartViChn(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32              i;
    hi_bool             bNeedChn;
    hi_s32              s32Ret = HI_SUCCESS;
    hi_vi_pipe             ViPipe;
    hi_vi_chn              ViChn;
    hi_vi_chn_attr       stChnAttr;

    for (i = 0; i < WDR_MAX_PIPE_NUM; i++)
    {
        if (pstViInfo->stPipeInfo.aPipe[i] >= 0  && pstViInfo->stPipeInfo.aPipe[i] < HI_VI_MAX_PIPE_NUM)
        {
            ViPipe = pstViInfo->stPipeInfo.aPipe[i];
            ViChn  = pstViInfo->stChnInfo.ViChn;

            Sample_Comm_Vi_GetChnAttrBySns(pstViInfo->stSnsInfo.enSnsType, &stChnAttr);
            stChnAttr.dynamic_range = pstViInfo->stChnInfo.enDynamicRange;
            stChnAttr.video_format  = pstViInfo->stChnInfo.enVideoFormat;
            stChnAttr.pixel_format  = pstViInfo->stChnInfo.enPixFormat;
            stChnAttr.compress_mode = pstViInfo->stChnInfo.enCompressMode;
            if (HI_WDR_MODE_NONE == pstViInfo->stDevInfo.enWDRMode)
            {
                bNeedChn = HI_TRUE;
            }
            else
            {
                bNeedChn = (i > 0) ? HI_FALSE : HI_TRUE;
            }

            if (bNeedChn)
            {
                s32Ret = hi_mpi_vi_set_chn_attr(ViPipe, ViChn, &stChnAttr);

                if (s32Ret != HI_SUCCESS)
                {
                    SAMPLE_PRT("hi_mpi_vi_set_chn_attr failed with %#x!\n", s32Ret);
                    return HI_FAILURE;
                }
                s32Ret = hi_mpi_vi_enable_chn(ViPipe, ViChn);
                if (s32Ret != HI_SUCCESS)
                {
                    SAMPLE_PRT("hi_mpi_vi_enable_chn failed with %#x!\n", s32Ret);
                    return HI_FAILURE;
                }
            }
        }
    }

    return s32Ret;
}

static hi_s32  Sample_Comm_Vi_StopSingleViChn(hi_s32 ViPipe, hi_s32 ViChn)
{
    hi_s32 s32Ret;
    s32Ret = hi_mpi_vi_disable_chn(ViPipe, ViChn);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("hi_mpi_vi_disable_chn failed with %#x!\n", s32Ret);
        return HI_FAILURE;
    }
}

hi_s32 Sample_Comm_Vi_StopViChn(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32              i;
    hi_bool             bNeedChn;
    hi_s32              s32Ret = HI_SUCCESS;
    hi_vi_pipe             ViPipe;
    hi_vi_chn              ViChn;
    for (i = 0; i < 4; i++)
    {
        if (pstViInfo->stPipeInfo.aPipe[i] >= 0  && pstViInfo->stPipeInfo.aPipe[i] < HI_VI_MAX_PIPE_NUM)
        {
            ViPipe = pstViInfo->stPipeInfo.aPipe[i];
            ViChn  = pstViInfo->stChnInfo.ViChn;

            if (HI_WDR_MODE_NONE == pstViInfo->stDevInfo.enWDRMode)
            {
                bNeedChn = HI_TRUE;
            }
            else
            {
                bNeedChn = (i > 0) ? HI_FALSE : HI_TRUE;
            }
            if (bNeedChn)
            {
                s32Ret = Sample_Comm_Vi_StopSingleViChn(ViPipe, ViChn);
            }
        }
    }
    return s32Ret;
}

static hi_s32 Sample_Comm_Vi_CreateSingleVi(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32 s32Ret = HI_SUCCESS;
    s32Ret = Sample_Comm_Vi_StartDev(pstViInfo);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_StartDev failed !\n");
        return HI_FAILURE;
    }
    // we should bind pipe,then creat pipe
    s32Ret = Sample_Comm_Vi_BindPipeDev(pstViInfo);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_BindPipeDev failed !\n");
        goto EXIT1;
    }

    s32Ret = Sample_Comm_Vi_StartViPipe(pstViInfo);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_StartViPipe failed !\n");
        goto EXIT1;
    }
    s32Ret = Sample_Comm_Vi_StartViChn(pstViInfo);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_StartViChn failed !\n");
        goto EXIT2;
    }
    return HI_SUCCESS;

EXIT2:
    Sample_Comm_Vi_StopViPipe(pstViInfo);
EXIT1:
    Sample_Comm_Vi_StopDev(pstViInfo);
    return s32Ret;
}

static hi_s32 Sample_Comm_ModeSwitch_VI_CreateSingleVi(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32 s32Ret = HI_SUCCESS;
    s32Ret = Sample_Comm_Vi_StartDev(pstViInfo);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_StartDev failed !\n");
        return HI_FAILURE;
    }
    //we should bind pipe,then creat pipe
    s32Ret = Sample_Comm_Vi_BindPipeDev(pstViInfo);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_BindPipeDev failed !\n");
        goto EXIT1;
    }
    s32Ret = Sample_Comm_Vi_ModeSwitch_StartViPipe(pstViInfo);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_StartViPipe failed !\n");
        goto EXIT1;
    }
    s32Ret = Sample_Comm_Vi_ModeSwitch_StartViChn(pstViInfo);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_StartViChn failed !\n");
        goto EXIT2;
    }
    return HI_SUCCESS;
EXIT2:
    Sample_Comm_Vi_StopViPipe(pstViInfo);
EXIT1:
    Sample_Comm_Vi_StopDev(pstViInfo);
    return s32Ret;
}

static hi_s32 Sample_Comm_Vi_StartPipe_Chn(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32 s32Ret = HI_SUCCESS;
    s32Ret = Sample_Comm_Vi_ModeSwitch_EnableViPipe(pstViInfo);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_StartViPipe failed !\n");
        goto EXIT1;
    }
    s32Ret = Sample_Comm_Vi_StartViChn(pstViInfo);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_StartViChn failed !\n");
        goto EXIT2;
    }
    return HI_SUCCESS;
EXIT2:
    Sample_Comm_Vi_StopViPipe(pstViInfo);
EXIT1:
    Sample_Comm_Vi_StopDev(pstViInfo);
    return s32Ret;
}

static hi_s32 Sample_Comm_Vi_DestroySingleVi(SAMPLE_VI_INFO_S* pstViInfo)
{
    Sample_Comm_Vi_StopViChn(pstViInfo);
    Sample_Comm_Vi_StopViPipe(pstViInfo);
    Sample_Comm_Vi_StopDev(pstViInfo);
    return HI_SUCCESS;
}

static hi_s32 Sample_Comm_Vi_DestroySinglePipe_Chn(SAMPLE_VI_INFO_S* pstViInfo)
{
    Sample_Comm_Vi_StopViChn(pstViInfo);
    Sample_Comm_Vi_StopViPipe(pstViInfo);
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_CreateVi(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i, j;
    hi_s32              s32ViNum;
    hi_s32              s32Ret = HI_SUCCESS;
    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;

    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        s32Ret = Sample_Comm_Vi_CreateSingleVi(pstViInfo);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Sample_Comm_Vi_CreateSingleVi failed !\n");
            goto EXIT;
        }
    }
    return HI_SUCCESS;
EXIT:
    for (j = 0; j < i; j++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[j];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        Sample_Comm_Vi_DestroySingleVi(pstViInfo);
    }
    return s32Ret;
}

hi_s32 Sample_Comm_ModeSwitch_VI_CreateVi(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i, j;
    hi_s32              s32ViNum;
    hi_s32              s32Ret = HI_SUCCESS;
    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;
    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }

    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        s32Ret = Sample_Comm_ModeSwitch_VI_CreateSingleVi(pstViInfo);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Sample_Comm_Vi_CreateSingleVi failed !\n");
            goto EXIT;
        }
    }
    return HI_SUCCESS;
EXIT:
    for (j = 0; j < i; j++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[j];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        Sample_Comm_Vi_DestroySingleVi(pstViInfo);
    }
    return s32Ret;
}

hi_s32 Sample_Comm_ModeSwitch_VI_StartPipe_Chn(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i, j;
    hi_s32              s32ViNum;
    hi_s32              s32Ret = HI_SUCCESS;
    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;

    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }
    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        s32Ret = Sample_Comm_Vi_StartPipe_Chn(pstViInfo);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Sample_Comm_Vi_CreateSingleVi failed !\n");
            goto EXIT;
        }
    }
    return HI_SUCCESS;
EXIT:
    for (j = 0; j < i; j++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[j];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        Sample_Comm_Vi_DestroySinglePipe_Chn(pstViInfo);
    }
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_DestroyVi(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32            i;
    hi_s32            s32ViNum;
    SAMPLE_VI_INFO_S* pstViInfo = HI_NULL;
    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }
    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        Sample_Comm_Vi_DestroySingleVi(pstViInfo);
    }
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_ReStartSns(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32 i;
    hi_s32 s32Ret = HI_SUCCESS;
    hi_vi_pipe ViPipe;
    hi_u32 u32SnsId;
    const hi_isp_sns_obj* pstSnsObj;

    for (i = 0; i < WDR_MAX_PIPE_NUM; i++)
    {
        if (pstViInfo->stPipeInfo.aPipe[i] >= 0  &&
            pstViInfo->stPipeInfo.aPipe[i] < HI_VI_MAX_PIPE_NUM)
        {
            ViPipe      = pstViInfo->stPipeInfo.aPipe[i];
            u32SnsId    = pstViInfo->stSnsInfo.s32SnsId;
            pstSnsObj = Sample_Comm_Isp_GetSnsObj(u32SnsId);
            if (HI_NULL == pstSnsObj)
            {
                SAMPLE_PRT("sensor %d not exist!\n", u32SnsId);
                return HI_FAILURE;
            }
            if (pstSnsObj->pfn_register_callback != HI_NULL)
            {
                pstSnsObj->pfn_restart(ViPipe);
            }
            else
            {
                SAMPLE_PRT("sensor_register_callback failed with HI_NULL!!!\n");
            }
        }
    }
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_StartIsp(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32              i;
    hi_bool             bNeedPipe;
    hi_s32              s32Ret = HI_SUCCESS;
    hi_vi_pipe             ViPipe;
    hi_u32              u32SnsId;
    hi_isp_pub_attr      stPubAttr;
    const hi_isp_sns_obj* pstSnsObj;
    for (i = 0; i < WDR_MAX_PIPE_NUM; i++)
    {
        if (pstViInfo->stPipeInfo.aPipe[i] >= 0  && pstViInfo->stPipeInfo.aPipe[i] < HI_VI_MAX_PIPE_NUM)
        {
            ViPipe      = pstViInfo->stPipeInfo.aPipe[i];
            u32SnsId    = pstViInfo->stSnsInfo.s32SnsId;
            Sample_Comm_Isp_GetIspAttrBySns(pstViInfo->stSnsInfo.enSnsType, &stPubAttr);
            stPubAttr.wdr_mode = pstViInfo->stDevInfo.enWDRMode;
            if (HI_WDR_MODE_NONE == pstViInfo->stDevInfo.enWDRMode)
            {
                bNeedPipe = HI_TRUE;
            }
            else
            {
                bNeedPipe = (i > 0) ? HI_FALSE : HI_TRUE;
            }
            if (HI_TRUE != bNeedPipe)
            {
                continue;
            }
            s32Ret = Sample_Comm_Isp_Sensor_Regiter_callback(ViPipe, u32SnsId);
            if (HI_SUCCESS != s32Ret)
            {
                SAMPLE_PRT("register sensor %d to ISP %d failed\n", u32SnsId, ViPipe);
                Sample_Comm_Isp_Stop(ViPipe);
                return HI_FAILURE;
            }
            s32Ret = Sample_Comm_Isp_BindSns(ViPipe, u32SnsId, pstViInfo->stSnsInfo.enSnsType,
                pstViInfo->stSnsInfo.s32BusId);
            if (HI_SUCCESS != s32Ret)
            {
                SAMPLE_PRT("register sensor %d bus id %d failed\n", u32SnsId,
                    pstViInfo->stSnsInfo.s32BusId);
                Sample_Comm_Isp_Stop(ViPipe);
                return HI_FAILURE;
            }
            s32Ret = Sample_Comm_Isp_Aelib_Callback(ViPipe);
            if (HI_SUCCESS != s32Ret)
            {
                SAMPLE_PRT("Sample_Comm_Isp_Aelib_Callback failed\n");
                Sample_Comm_Isp_Stop(ViPipe);
                return HI_FAILURE;
            }
            s32Ret = Sample_Comm_Isp_Awblib_Callback(ViPipe);
            if (HI_SUCCESS != s32Ret)
            {
                SAMPLE_PRT("Sample_Comm_Isp_Awblib_Callback failed\n");
                Sample_Comm_Isp_Stop(ViPipe);
                return HI_FAILURE;
            }
            s32Ret = hi_mpi_isp_mem_init(ViPipe);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_PRT("Init Ext memory failed with %#x!\n", s32Ret);
                Sample_Comm_Isp_Stop(ViPipe);
                return HI_FAILURE;
            }
            s32Ret = hi_mpi_isp_set_pub_attr(ViPipe, &stPubAttr);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_PRT("SetPubAttr failed with %#x!\n", s32Ret);
                Sample_Comm_Isp_Stop(ViPipe);
                return HI_FAILURE;
            }
            s32Ret = hi_mpi_isp_init(ViPipe);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_PRT("ISP Init failed with %#x!\n", s32Ret);
                Sample_Comm_Isp_Stop(ViPipe);
                return HI_FAILURE;
            }
            s32Ret = Sample_Comm_Isp_Run(&pstViInfo->stPipeInfo.aPipe[i]);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_PRT("ISP Run failed with %#x!\n", s32Ret);
                Sample_Comm_Isp_Stop(ViPipe);
                return HI_FAILURE;
            }
        }
    }
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_StopIsp(SAMPLE_VI_INFO_S* pstViInfo)
{
    hi_s32  i;
    hi_bool bNeedPipe;
    hi_vi_pipe ViPipe;
    for (i = 0; i < WDR_MAX_PIPE_NUM; i++)
    {
        if (pstViInfo->stPipeInfo.aPipe[i] >= 0  &&
            pstViInfo->stPipeInfo.aPipe[i] < HI_VI_MAX_PIPE_NUM)
        {
            ViPipe    = pstViInfo->stPipeInfo.aPipe[i];

            if (HI_WDR_MODE_NONE == pstViInfo->stDevInfo.enWDRMode)
            {
                bNeedPipe = HI_TRUE;
            }
            else
            {
                bNeedPipe = (i > 0) ? HI_FALSE : HI_TRUE;
            }

            if (HI_TRUE != bNeedPipe)
            {
                continue;
            }
            Sample_Comm_Isp_Stop(ViPipe);
        }
    }

    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_CreateIsp(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i;
    hi_s32              s32ViNum;
    hi_s32              s32Ret = HI_SUCCESS;

    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;
    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }
    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        s32Ret = Sample_Comm_Vi_StartIsp(pstViInfo);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Sample_Comm_Vi_StartIsp failed !\n");
            return HI_FAILURE;
        }
    }
    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
    }
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_DestroyIsp(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i;
    hi_s32              s32ViNum;
    hi_s32              s32Ret = HI_SUCCESS;
    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;
    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }
    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        s32Ret = Sample_Comm_Vi_StopIsp(pstViInfo);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Sample_Comm_Vi_StopIsp failed !\n");
            return HI_FAILURE;
        }
    }
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_StartVi(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32 s32Ret = HI_SUCCESS;
    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }
    s32Ret = Sample_Comm_Vi_StartMIPI(pstViConfig);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_StartMIPI failed!\n");
        return HI_FAILURE;
    }
    s32Ret = Sample_Comm_Vi_SetParam(pstViConfig);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_SetParam failed!\n");
        return HI_FAILURE;
    }
    s32Ret = Sample_Comm_Vi_CreateVi(pstViConfig);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_CreateVi failed!\n");
        return HI_FAILURE;
    }
    s32Ret = Sample_Comm_Vi_CreateIsp(pstViConfig);
    if (s32Ret != HI_SUCCESS)
    {
        Sample_Comm_Vi_DestroyVi(pstViConfig);
        SAMPLE_PRT("Sample_Comm_Vi_CreateIsp failed!\n");
        return HI_FAILURE;
    }
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_StopVi(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32 s32Ret = HI_SUCCESS;
    s32Ret = Sample_Comm_Vi_DestroyIsp(pstViConfig);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_DestroyIsp failed !\n");
        return HI_FAILURE;
    }
    s32Ret = Sample_Comm_Vi_DestroyVi(pstViConfig);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_DestroyVi failed !\n");
        return HI_FAILURE;
    }
    s32Ret = Sample_Comm_Vi_StopMIPI(pstViConfig);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Sample_Comm_Vi_StopMIPI failed !\n");
        return HI_FAILURE;
    }
    return s32Ret;
}

static hi_s32 Sample_Comm_Vi_SetSingleIsp(hi_vi_pipe ViPipe,
    hi_isp_pub_attr *stPrePubAttr, hi_isp_pub_attr *stPubAttr)
{
    hi_s32 s32Ret;
    s32Ret = hi_mpi_isp_set_pub_attr(ViPipe, &stPrePubAttr);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("GetPubAttr failed with %#x!\n", s32Ret);
        Sample_Comm_Isp_Stop(ViPipe);
        return HI_FAILURE;
    }
    s32Ret = hi_mpi_isp_set_pub_attr(ViPipe, &stPubAttr);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("SetPubAttr failed with %#x!\n", s32Ret);
        Sample_Comm_Isp_Stop(ViPipe);
        return HI_FAILURE;
    }
    return s32Ret;
}

hi_s32 Sample_Comm_Vi_SwitchISPMode(SAMPLE_VI_CONFIG_S* pstViConfig)
{
    hi_s32              i,j;
    hi_s32              s32ViNum;
    hi_s32              s32Ret = HI_SUCCESS;
    hi_bool             bNeedPipe;
    hi_vi_pipe          ViPipe = 0;
    hi_isp_pub_attr     stPubAttr;
    hi_isp_pub_attr     stPrePubAttr;
    SAMPLE_VI_INFO_S*   pstViInfo = HI_NULL;
    hi_bool bSwitchWDR[HI_VI_MAX_PIPE_NUM] = { HI_FALSE };
    hi_isp_inner_state_info stInnerStateInfo;
    hi_bool bSwitchFinish;
    hi_u32         u32SnsId;
    const hi_isp_sns_obj *pstSnsObj;
    if (!pstViConfig)
    {
        SAMPLE_PRT("%s: null ptr\n", __FUNCTION__);
        return HI_FAILURE;
    }
    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        for (j = 0; j < WDR_MAX_PIPE_NUM; j++)
        {
            if ( pstViInfo->stPipeInfo.aPipe[j] >= 0 && pstViInfo->stPipeInfo.aPipe[j]
                < HI_VI_MAX_PIPE_NUM)
            {
                Sample_Comm_Isp_GetIspAttrBySns(pstViInfo->stSnsInfo.enSnsType, &stPubAttr);
                stPubAttr.wdr_mode =  pstViInfo->stDevInfo.enWDRMode ;
                SAMPLE_PRT("Sample_Comm_Vi_CreateIsp enWDRMode is %d!\n", stPubAttr.wdr_mode);
                if (HI_WDR_MODE_NONE == pstViInfo->stDevInfo.enWDRMode )
                {
                    bNeedPipe = HI_TRUE;
                }
                else
                {
                    bNeedPipe = (j > 0) ? HI_FALSE : HI_TRUE;
                }
                if (HI_TRUE != bNeedPipe)
                {
                    continue;
                }
                ViPipe = pstViInfo->stPipeInfo.aPipe[j];
                s32Ret = Sample_Comm_Vi_SetSingleIsp(ViPipe, &stPrePubAttr, &stPubAttr);
		if (stPrePubAttr.wdr_mode != stPubAttr.wdr_mode)
		{
		    bSwitchWDR[ViPipe] = HI_TRUE;
		}
            }
        }
    }

    while (1)
    {
        bSwitchFinish = HI_TRUE;
        for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
        {
            s32ViNum  = pstViConfig->as32WorkingViId[i];
            pstViInfo = &pstViConfig->astViInfo[s32ViNum];
            ViPipe    = pstViInfo->stPipeInfo.aPipe[0];
            u32SnsId  = pstViInfo->stSnsInfo.s32SnsId;
            hi_mpi_isp_query_inner_state_info(ViPipe, &stInnerStateInfo);
            if (bSwitchWDR[ViPipe] == HI_TRUE)
            {
                bSwitchFinish &= stInnerStateInfo.wdr_switch_finish;
            }
            else
            {
                bSwitchFinish &= stInnerStateInfo.res_switch_finish;
            }
        }
        if (bSwitchFinish == HI_TRUE) {
            SAMPLE_PRT("Switch finish!\n");
            break;
        }
        usleep(1000);
    }

    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
    }

    for (i = 0; i < pstViConfig->s32WorkingViNum; i++)
    {
        s32ViNum  = pstViConfig->as32WorkingViId[i];
        pstViInfo = &pstViConfig->astViInfo[s32ViNum];
        Sample_Comm_Vi_StartPipe_Chn(pstViInfo);
    }
    return s32Ret;
}

/******************************************************************************
* funciton : Get enSize by diffrent sensor
******************************************************************************/
hi_s32 Sample_Comm_Vi_GetFrameRateBySensor(SAMPLE_SNS_TYPE_E enMode, hi_u32* pu32FrameRate)
{
    hi_s32 s32Ret = HI_SUCCESS;
    if (!pu32FrameRate)
    {
        return HI_FAILURE;
    }
    *pu32FrameRate = 30; // 设置sensor帧率为30
    return s32Ret;
}

combo_dev_t Sample_Comm_Vi_GetComboDevBySensor(SAMPLE_SNS_TYPE_E enMode, hi_s32 s32SnsIdx)
{
    combo_dev_t dev = 0;
    return dev;
}

hi_s32 Sample_Comm_Vi_ConvertBitPixel(hi_u8 *pu8Data, hi_u32 u32DataNum,
    hi_u32 u32BitWidth, hi_u16 *pu16OutData)
{
    hi_s32 i, u32Tmp, s32OutCnt;
    hi_u32 u32Val;
    hi_u64 u64Val;
    hi_u8 *pu8Tmp = pu8Data;

    s32OutCnt = 0;
    switch (u32BitWidth)
    {
    case 10: // 10 bit
        {
            /* 4 pixels consist of 5 bytes  */
            u32Tmp = u32DataNum / 4;
            for (i = 0; i < u32Tmp; i++)
            {
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
    case 12: // 12 bit
        {
            /* 2 pixels consist of 3 bytes  */
            u32Tmp = u32DataNum / 2;
            for (i = 0; i < u32Tmp; i++)
            {
                /* byte2 byte1 byte0 */
                pu8Tmp = pu8Data + 3 * i;
                u32Val = pu8Tmp[0] + (pu8Tmp[1] << 8) + (pu8Tmp[2] << 16);
                pu16OutData[s32OutCnt++] = u32Val & 0xfff;
                pu16OutData[s32OutCnt++] = (u32Val >> 12) & 0xfff;
            }
        }
        break;
    case 14: // 14 bit
        {
            /* 4 pixels consist of 7 bytes  */
            u32Tmp = u32DataNum / 4;
            for (i = 0; i < u32Tmp; i++)
            {
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
        SAMPLE_PRT("unsuport bitWidth: %d\n", u32BitWidth);
        return HI_FAILURE;
        break;
    }
    return s32OutCnt;
}

static hi_s32 Sample_Comm_Vi_BitWidth2PixelFormat(hi_u32 u32Nbit, hi_pixel_format *penPixelFormat)
{
    hi_pixel_format enPixelFormat;
    if (8 == u32Nbit)
    {
        enPixelFormat = HI_PIXEL_FORMAT_RGB_BAYER_8BPP;
    }
    else if (10 == u32Nbit)
    {
        enPixelFormat = HI_PIXEL_FORMAT_RGB_BAYER_10BPP;
    }
    else if (12 == u32Nbit)
    {
        enPixelFormat = HI_PIXEL_FORMAT_RGB_BAYER_12BPP;
    }
    else if (14 == u32Nbit)
    {
        enPixelFormat = HI_PIXEL_FORMAT_RGB_BAYER_14BPP;
    }
    else if (16 == u32Nbit)
    {
        enPixelFormat = HI_PIXEL_FORMAT_RGB_BAYER_16BPP;
    }
    else
    {
        return HI_FAILURE;
    }
    *penPixelFormat = enPixelFormat;
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Vi_SaveUncompressRaw(hi_video_frame* pVBuf, hi_u32 u32Nbit, FILE* pfd)
{
    hi_u32  u32Height;
    hi_u64  u64PhyAddr;
    hi_u64  u64Size;
    hi_u8*  pu8VirAddr;
    hi_u16 *pu16Data = NULL;
    hi_u8  *pu8Data;
    hi_pixel_format enPixelFormat = HI_PIXEL_FORMAT_BUTT;
    Sample_Comm_Vi_BitWidth2PixelFormat(u32Nbit, &enPixelFormat);
    if (enPixelFormat != pVBuf->pixel_format)
    {
        SAMPLE_PRT("invalid pixel format:%d, u32Nbit: %d\n", pVBuf->pixel_format, u32Nbit);
        return HI_FAILURE;
    }
    u64Size = (pVBuf->header_stride[0]) * ((hi_u64)pVBuf->height);
    u64PhyAddr = pVBuf->phys_addr[0];
    pu8VirAddr = (hi_u8*)u64PhyAddr;
    pu8Data = pu8VirAddr;
    if ((8 != u32Nbit) && (16 != u32Nbit))
    {
        pu16Data = (hi_u16*)malloc(pVBuf->width * 2U);
        if (NULL == pu16Data)
        {
            SAMPLE_PRT("alloc memory failed\n");
            pu8VirAddr = NULL;
            return HI_FAILURE;
        }
    }
    /* save Y ----------------------------------------------------------------*/
    SAMPLE_PRT("saving......dump data......u32Stride[0]: %d, width: %d\n",
        pVBuf->header_stride[0], pVBuf->width);

    for (u32Height = 0; u32Height < pVBuf->height; u32Height++)
    {
        if (8 == u32Nbit)
        {
            fwrite(pu8Data, pVBuf->width, 1, pfd);
        }
        else if (16 == u32Nbit)
        {
            fwrite(pu8Data, pVBuf->width, 2, pfd);
            fflush(pfd);
        }
        else
        {
            Sample_Comm_Vi_ConvertBitPixel(pu8Data, pVBuf->width, u32Nbit, pu16Data);
            fwrite(pu16Data, pVBuf->width, 2, pfd);
        }
        pu8Data += pVBuf->header_stride[0];
    }
    fflush(pfd);
    SAMPLE_PRT("done u32TimeRef: %d!\n", pVBuf->time_ref);
    if (NULL != pu16Data)
    {
        free(pu16Data);
    }
    pu8VirAddr = NULL;
    return HI_SUCCESS;
}

hi_u32 Sample_Comm_Vi_PixelFormat2BitWidth(hi_pixel_format  enPixelFormat)
{
    switch (enPixelFormat)
    {
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

char* Sample_Comm_Vi_CompressMode2String(hi_compress_mode enCompressMode)
{
    if(HI_COMPRESS_MODE_NONE == enCompressMode)
    {
        return "CMP_NONE";
    }
    else if(HI_COMPRESS_MODE_LINE == enCompressMode)
    {
        return "CMP_LINE";
    }
    else if(HI_COMPRESS_MODE_SEG == enCompressMode)
    {
        return "CMP_SEG";
    }
    else
    {
        return "CMP_XXX";
    }
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
