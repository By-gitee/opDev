/**
* @File vo_home.h
* @Description hdmi sample app
*
* Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include "hi_media_common.h"
#include "hi_common_vo.h"
#include "hi_mpi_vo.h"
#include "hi_mpi_hdmi.h"
typedef hi_u32 hi_vb_pool;
typedef hi_u32 hi_vb_blk;

#define VO_CHECK_RET(express, name)                                                              \
    do {                                                                                         \
        hi_s32 _ret;                                                                             \
        _ret = express;                                                                          \
        if (_ret != HI_SUCCESS) {                                                                \
            printf("%s failed at %s: LINE: %d with %#x!\n", name, __FUNCTION__, __LINE__, _ret); \
        }                                                                                        \
    } while (0)

#define FILE_NAME_LEN 512
#define VO_MST_ALIGN_16 16
#define VO_MST_ALIGN_2 2
#define VO_ALIGN_BACK(x, a) ((a) * (((x) / (a))))
#define VO_TEST_ALIGN_BACK(x, a) (((a) * ((((x) + (a) - 1) / (a)))))
#define VO_LAYER_VHD0 0
#define DEV_DHD0 0

typedef struct {
    hi_vo_intf_sync intf_sync;
    hi_char *name;
    hi_u32 width;
    hi_u32 height;
    hi_u32 frame_rate;
} vo_mst_sync_info;

typedef enum {
    VB_REMAP_MODE_NONE = 0,    // no remap
    VB_REMAP_MODE_NOCACHE = 1, // no cache remap
    VB_REMAP_MODE_CACHED = 2,  // cache remap, if you use this mode, you should flush cache by yourself
    VB_REMAP_MODE_BUTT
} vo_vb_remap_mode;

typedef struct {
    hi_u64 blk_size;
    hi_u32 blk_cnt;
    vo_vb_remap_mode remap_mode;
} hi_vb_pool_config;

hi_void hi_mpi_hdmi_init_sample(void);
hi_s32 hi_mpi_hdmi_avi_infoframe_colorspace(int hdmi_timing, int pix_clk);