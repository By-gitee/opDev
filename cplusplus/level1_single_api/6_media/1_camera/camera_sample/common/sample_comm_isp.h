#ifndef Sample_Comm_Isp_H
#define Sample_Comm_Isp_H

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

hi_void Sample_Comm_Isp_Stop(hi_isp_dev IspDev);
hi_void Sample_Comm_All_ISP_Stop(hi_void);
hi_s32 Sample_Comm_Isp_Run(hi_isp_dev* pIspDev);
hi_s32 Sample_Comm_Isp_BindSns(hi_isp_dev IspDev, hi_u32 u32SnsId, SAMPLE_SNS_TYPE_E enSnsType, hi_s8 s8SnsDev);
hi_s32 Sample_Comm_Isp_Sensor_Regiter_callback(hi_isp_dev IspDev, hi_u32 u32SnsId);
hi_s32 Sample_Comm_Isp_Sensor_UnRegiter_callback(hi_isp_dev IspDev);
hi_s32 Sample_Comm_Isp_Aelib_Callback(hi_isp_dev IspDev);
hi_s32 Sample_Comm_Isp_Aelib_UnCallback(hi_isp_dev IspDev);
hi_s32 Sample_Comm_Isp_Awblib_Callback(hi_isp_dev IspDev);
hi_s32 Sample_Comm_Isp_Awblib_UnCallback(hi_isp_dev IspDev);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
#endif // Sample_Comm_Isp_H