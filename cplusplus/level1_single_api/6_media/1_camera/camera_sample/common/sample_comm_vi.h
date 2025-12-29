#ifndef Sample_Comm_Vi_H
#define Sample_Comm_Vi_H

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

hi_s32 Sample_Comm_Vi_GetWDRModeBySensor(SAMPLE_SNS_TYPE_E enMode, hi_wdr_mode* penWDRMode);
hi_s32 Sample_Comm_Vi_GetSizeBySensor(SAMPLE_SNS_TYPE_E enMode, PIC_SIZE_E* penSize);
hi_s32 Sample_Comm_Vi_GetFrameRateBySensor(SAMPLE_SNS_TYPE_E enMode, hi_u32* pu32FrameRate);
hi_s32 Sample_Comm_Vi_StartDev(SAMPLE_VI_INFO_S* pstViInfo);
hi_s32 Sample_Comm_Vi_StartChn(hi_vi_chn ViChn, hi_rect* pstCapRect, hi_size* pstTarSize,
    SAMPLE_VI_CONFIG_S* pstViConfig);
hi_s32 Sample_Comm_Vi_StartMIPI(SAMPLE_VI_CONFIG_S* pstViConfig);
hi_s32 Sample_Comm_Vi_StartVi(SAMPLE_VI_CONFIG_S* pstViConfig);
hi_s32 Sample_Comm_Vi_StopVi(SAMPLE_VI_CONFIG_S* pstViConfig);
hi_s32 Sample_Comm_Vi_SetMipiAttr(SAMPLE_VI_CONFIG_S* pstViConfig);
hi_void Sample_Comm_Vi_GetSensorInfo(SAMPLE_VI_CONFIG_S* pstViConfig);

combo_dev_t Sample_Comm_Vi_GetComboDevBySensor(SAMPLE_SNS_TYPE_E enMode, hi_s32 s32SnsIdx);
hi_s32 Sample_Comm_Vi_SetParam(SAMPLE_VI_CONFIG_S* pstViConfig);
hi_s32 Sample_Comm_Vi_SwitchMode_StopVI(SAMPLE_VI_CONFIG_S* pstViConfigSrc);
hi_s32 Sample_Comm_Vi_SwitchMode(SAMPLE_VI_CONFIG_S* pstViConfigDes);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif // Sample_Comm_Vi_H