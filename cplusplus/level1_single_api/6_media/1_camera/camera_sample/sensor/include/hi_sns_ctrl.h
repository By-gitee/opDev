/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Function of hi_sns_ctrl.h
 * Author: Hisilicon multimedia software group
 * Create: 2023-03-06
 */

#ifndef HI_SNS_CTRL_H
#define HI_SNS_CTRL_H

#include "hi_media_type.h"
#include "hi_common_3a.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */
#define ISP_SNS_SAVE_INFO_MAX 2U
typedef struct {
    hi_bool      init;                   /* HI_TRUE: Sensor init */
    hi_bool      sync_init;              /* HI_TRUE: Sync Reg init */
    hi_u8        img_mode;
    hi_u8        hdr;                 /* HI_TRUE: HDR enbale */
    hi_wdr_mode  wdr_mode;

    hi_isp_sns_regs_info regs_info[ISP_SNS_SAVE_INFO_MAX]; /* [0]: Sensor reg info of cur-frame;
                                                              [1]: Sensor reg info of pre-frame ; */

    hi_u32      fl[ISP_SNS_SAVE_INFO_MAX];                /* [0]: FullLines of cur-frame;
                                                             [1]: Pre FullLines of pre-frame */
    hi_u32      fl_std;                    /* FullLines std */
    hi_u32      wdr_int_time[HI_ISP_WDR_MAX_FRAME_NUM];
    hi_u32      sensor_wb_gain[HI_ISP_BAYER_CHN_NUM];
} hi_isp_sns_state;

typedef enum hiISP_SNS_MIRRORFLIP_TYPE_E {
    ISP_SNS_NORMAL      = 0,
    ISP_SNS_MIRROR      = 1,
    ISP_SNS_FLIP        = 2,
    ISP_SNS_MIRROR_FLIP = 3,
    ISP_SNS_BUTT
} hi_isp_sns_mirrorflip_type;

typedef struct {
    hi_s32  (*pfn_register_callback)(hi_vi_pipe vi_pipe, hi_isp_3a_alg_lib *ae_lib, hi_isp_3a_alg_lib *awb_lib);
    hi_s32  (*pfn_un_register_callback)(hi_vi_pipe vi_pipe, hi_isp_3a_alg_lib *ae_lib, hi_isp_3a_alg_lib *awb_lib);
    hi_s32  (*pfn_set_bus_info)(hi_vi_pipe vi_pipe, hi_isp_sns_commbus sns_bus_info);
    hi_void (*pfn_standby)(hi_vi_pipe vi_pipe);
    hi_void (*pfn_restart)(hi_vi_pipe vi_pipe);
    hi_void (*pfn_mirror_flip)(hi_vi_pipe vi_pipe, hi_isp_sns_mirrorflip_type sns_mirror_flip);
    hi_s32  (*pfn_write_reg)(hi_vi_pipe vi_pipe, hi_u32 addr, hi_u32 data);
    hi_s32  (*pfn_read_reg)(hi_vi_pipe vi_pipe, hi_u32 addr);
    hi_s32  (*pfn_set_init)(hi_vi_pipe vi_pipe, hi_isp_init_attr *init_attr);
    hi_s32  (*pfn_identify_module)(hi_vi_pipe vi_pipe);
} hi_isp_sns_obj;

#define SENSOR_FREE(ptr) \
    do { \
        if ((ptr) != HI_NULL) { \
            free(ptr); \
            (ptr) = HI_NULL; \
        } \
    } while (0)

#define SNS_ERR_TRACE(fmt, ...)                                                                            \
    do {                                                                                                    \
        printf("[sensor]" fmt, ##__VA_ARGS__);      \
    } while (0)

#define SNS_INFO_TRACE(fmt, ...)                                                                            \
    do {                                                                                                    \
        printf("[sensor]" fmt, ##__VA_ARGS__);      \
    } while (0)

#define CMOS_CHECK_POINTER(ptr) \
    do { \
        if ((ptr) == HI_NULL) { \
            SNS_ERR_TRACE("Null Pointer!\n");  \
            return HI_ERR_ISP_NULL_PTR; \
        } \
    } while (0)

#define CMOS_CHECK_POINTER_VOID(ptr) \
    do { \
        if ((ptr) == HI_NULL) { \
            SNS_ERR_TRACE("Null Pointer!\n"); \
            return; \
        } \
    } while (0)

#define SNS_CHECK_PIPE_RETURN(viPipe)                                   \
    do {                                                       \
        if (((viPipe) < 0) || ((viPipe) >= HI_ISP_MAX_PIPE_NUM)) {    \
            SNS_ERR_TRACE("Err viPipe %d!\n", viPipe); \
            return HI_ERR_ISP_ILLEGAL_PARAM;                   \
        }                                                      \
    } while (0)

#define SNS_CHECK_PIPE_VOID(viPipe)                                   \
    do {                                                       \
        if (((viPipe) < 0) || ((viPipe) >= HI_ISP_MAX_PIPE_NUM)) {    \
            SNS_ERR_TRACE("Err viPipe %d!\n", viPipe); \
            return;                   \
        }                                                      \
    } while (0)

#define SNS_DIV_0_TO_1(a)   (((a) == 0) ? 1 : (a))
#define SNS_DIV_0_TO_1_FLOAT(a) ((((a) < 1E-10) && ((a) > (-1E-10))) ? 1 : (a))
#define sns_unused(x) ((hi_void)(x))

#define HIGHER_4BITS(x) (((x) & 0xf0000) >> 16U)
#define HIGH_8BITS(x) (((x) & 0xff00) >> 8U)
#define LOW_8BITS(x)  ((x) & 0x00ff)

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif /* HI_SNS_CTRL_H */
