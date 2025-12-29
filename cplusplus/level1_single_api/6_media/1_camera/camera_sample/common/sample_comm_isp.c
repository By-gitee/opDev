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

#include "sample_comm.h"
#include "sample_comm_isp.h"
#include "sensor_management.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define ISP_MAX_DEV_NUM     8

static pthread_t    g_IspPid[ISP_MAX_DEV_NUM] = {0};
static hi_s32       g_DevId = 0; // 全局变量传参pthread 避免局部变量传参，函数返回后销毁，值不确定
static hi_u32       g_au32IspSnsId[ISP_MAX_DEV_NUM] = {0, 1};

static void* Sample_Comm_Isp_Thread(void* param)
{
    hi_s32 s32Ret;
    hi_isp_dev IspDev;
    hi_char szThreadName[20];

    IspDev = *((hi_isp_dev*)param);

    snprintf(szThreadName, 20, "ISP%d_RUN", IspDev);
    prctl(PR_SET_NAME, szThreadName, 0,0,0);

    SAMPLE_PRT("ISP Dev %d running ! %p. \n", IspDev, param);
    s32Ret = hi_mpi_isp_run(IspDev);

    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("hi_mpi_isp_run failed with %#x!\n", s32Ret);
    }

    return NULL;
}

/******************************************************************************
* funciton : ISP init
******************************************************************************/

hi_s32 Sample_Comm_Isp_Aelib_Callback(hi_isp_dev IspDev)
{
    hi_isp_3a_alg_lib stAeLib;

    stAeLib.id = IspDev;
    strncpy(stAeLib.lib_name, HI_AE_LIB_NAME, sizeof(HI_AE_LIB_NAME));
    CHECK_RET(hi_mpi_ae_register(IspDev, &stAeLib), "aelib register call back");
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Isp_Aelib_UnCallback(hi_isp_dev IspDev)
{
    hi_isp_3a_alg_lib stAeLib;

    stAeLib.id = IspDev;
    strncpy(stAeLib.lib_name, HI_AE_LIB_NAME, sizeof(HI_AE_LIB_NAME));
    CHECK_RET(hi_mpi_ae_unregister(IspDev, &stAeLib), "aelib unregister call back");
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Isp_Awblib_Callback(hi_isp_dev IspDev)
{
    hi_isp_3a_alg_lib stAwbLib;

    stAwbLib.id = IspDev;
    strncpy(stAwbLib.lib_name, HI_AWB_LIB_NAME, sizeof(HI_AWB_LIB_NAME));
    CHECK_RET(hi_mpi_awb_register(IspDev, &stAwbLib), "awblib register call back");
    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Isp_Awblib_UnCallback(hi_isp_dev IspDev)
{
    hi_isp_3a_alg_lib stAwbLib;

    stAwbLib.id = IspDev;
    strncpy(stAwbLib.lib_name, HI_AWB_LIB_NAME, sizeof(HI_AWB_LIB_NAME));
    CHECK_RET(hi_mpi_awb_unregister(IspDev, &stAwbLib), "awblib unregister call back");
    return HI_SUCCESS;
}

/******************************************************************************
* funciton : ISP Run
******************************************************************************/
hi_s32 Sample_Comm_Isp_Run(hi_isp_dev* pIspDev)
{
    hi_s32 s32Ret = 0;
    pthread_attr_t* pstAttr = NULL;
    g_DevId = *pIspDev;
    s32Ret = pthread_create(&g_IspPid[*pIspDev], pstAttr, Sample_Comm_Isp_Thread, (hi_void*)(&g_DevId));
    if (0 != s32Ret)
    {
        SAMPLE_PRT("create isp running thread failed!, error: %d, %s\r\n", s32Ret, strerror(s32Ret));
        goto out;
    }

out:
    if (NULL != pstAttr)
    {
        pthread_attr_destroy(pstAttr);
    }
    return s32Ret;
}

hi_s32 Sample_Comm_Isp_Sensor_Regiter_callback(hi_isp_dev IspDev, hi_u32 u32SnsId)
{
    hi_isp_3a_alg_lib stAeLib;
    hi_isp_3a_alg_lib stAwbLib;
    hi_isp_3a_alg_lib stAfLib;
    const hi_isp_sns_obj* pstSnsObj;
    hi_s32    s32Ret = -1;

    if (MAX_SENSOR_NUM <= u32SnsId)
    {
        SAMPLE_PRT("invalid sensor id: %d\n", u32SnsId);
        return HI_FAILURE;
    }

    pstSnsObj = Sample_Comm_Isp_GetSnsObj(u32SnsId);

    if (HI_NULL == pstSnsObj)
    {
        SAMPLE_PRT("sensor %d not exist!\n", u32SnsId);
        return HI_FAILURE;
    }

    stAeLib.id = IspDev;
    stAwbLib.id = IspDev;
    strncpy(stAeLib.lib_name, HI_AE_LIB_NAME, sizeof(HI_AE_LIB_NAME));
    strncpy(stAwbLib.lib_name, HI_AWB_LIB_NAME, sizeof(HI_AWB_LIB_NAME));

    if (pstSnsObj->pfn_register_callback != HI_NULL)
    {
        s32Ret = pstSnsObj->pfn_register_callback(IspDev, &stAeLib, &stAwbLib);

        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("sensor_register_callback failed with %#x!\n", s32Ret);
            return s32Ret;
        }
    }
    else
    {
        SAMPLE_PRT("sensor_register_callback failed with HI_NULL!\n");
    }
    g_au32IspSnsId[IspDev] = u32SnsId;

    return HI_SUCCESS;
}

hi_s32 Sample_Comm_Isp_Sensor_UnRegiter_callback(hi_isp_dev IspDev)
{
    hi_isp_3a_alg_lib stAeLib;
    hi_isp_3a_alg_lib stAwbLib;
    hi_isp_3a_alg_lib stAfLib;
    hi_u32 u32SnsId;
    const hi_isp_sns_obj* pstSnsObj;
    hi_s32    s32Ret = -1;
    u32SnsId = g_au32IspSnsId[IspDev];

    if (MAX_SENSOR_NUM <= u32SnsId)
    {
        SAMPLE_PRT("%s: invalid sensor id: %d\n", __FUNCTION__, u32SnsId);
        return HI_FAILURE;
    }

    pstSnsObj = Sample_Comm_Isp_GetSnsObj(u32SnsId);

    if (HI_NULL == pstSnsObj)
    {
        return HI_FAILURE;
    }

    stAeLib.id = IspDev;
    stAwbLib.id = IspDev;
    strncpy(stAeLib.lib_name, HI_AE_LIB_NAME, sizeof(HI_AE_LIB_NAME));
    strncpy(stAwbLib.lib_name, HI_AWB_LIB_NAME, sizeof(HI_AWB_LIB_NAME));

    if (pstSnsObj->pfn_un_register_callback != HI_NULL)
    {
        s32Ret = pstSnsObj->pfn_un_register_callback(IspDev, &stAeLib, &stAwbLib);

        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("sensor_unregister_callback failed with %#x!\n", s32Ret);
            return s32Ret;
        }
    }
    else
    {
        SAMPLE_PRT("sensor_unregister_callback failed with HI_NULL!\n");
    }

    return HI_SUCCESS;
}

/******************************************************************************
* funciton : stop ISP, and stop isp thread
******************************************************************************/
hi_void Sample_Comm_Isp_Stop(hi_isp_dev IspDev)
{
    hi_mpi_isp_exit(IspDev);
    if (g_IspPid[IspDev])
    {
        SAMPLE_PRT("%s begin join g_IspPid=%u\n", __func__, g_IspPid[IspDev]);
        pthread_join(g_IspPid[IspDev], NULL);
        g_IspPid[IspDev] = 0;
    }

    Sample_Comm_Isp_Awblib_UnCallback(IspDev);
    Sample_Comm_Isp_Aelib_UnCallback(IspDev);
    Sample_Comm_Isp_Sensor_UnRegiter_callback(IspDev);
    return;
}

hi_void Sample_Comm_All_ISP_Stop(hi_void)
{
    hi_isp_dev IspDev;

    for (IspDev = 0; IspDev < HI_ISP_MAX_PIPE_NUM; IspDev++)
    {
        Sample_Comm_Isp_Stop(IspDev);
    }
}

static hi_isp_sns_type Sample_Comm_GetSnsBusType(SAMPLE_SNS_TYPE_E enSnsType)
{
    hi_isp_sns_type enBusType;

    switch (enSnsType)
    {
        default:
            enBusType = HI_ISP_SNS_I2C_TYPE;
            break;
    }

    return enBusType;
}

hi_s32 Sample_Comm_Isp_BindSns(hi_isp_dev IspDev, hi_u32 u32SnsId, SAMPLE_SNS_TYPE_E enSnsType, hi_s8 s8SnsDev)
{
    hi_isp_sns_commbus uSnsBusInfo;
    hi_isp_sns_type enBusType;
    const hi_isp_sns_obj* pstSnsObj;
    hi_s32 s32Ret;

    if (MAX_SENSOR_NUM <= u32SnsId)
    {
        SAMPLE_PRT("invalid sensor id: %d\n", u32SnsId);
        return HI_FAILURE;
    }

    pstSnsObj = Sample_Comm_Isp_GetSnsObj(u32SnsId);

    if (HI_NULL == pstSnsObj)
    {
        SAMPLE_PRT("sensor %d not exist!\n", u32SnsId);
        return HI_FAILURE;
    }

    enBusType = Sample_Comm_GetSnsBusType(enSnsType);

    if (HI_ISP_SNS_I2C_TYPE == enBusType)
    {
        uSnsBusInfo.i2c_dev = s8SnsDev;
    }
    else
    {
        uSnsBusInfo.ssp_dev.bit4_ssp_dev = s8SnsDev;
        uSnsBusInfo.ssp_dev.bit4_ssp_cs  = 0;
    }

    if (HI_NULL != pstSnsObj->pfn_set_bus_info)
    {
        s32Ret = pstSnsObj->pfn_set_bus_info(IspDev, uSnsBusInfo);

        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("set sensor bus info failed with %#x!\n", s32Ret);
            return s32Ret;
        }
    }
    else
    {
        SAMPLE_PRT("not support set sensor bus info!\n");
        return HI_FAILURE;
    }

    return s32Ret;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
