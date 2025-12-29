/**
* @File vo_comm.h
* @Description Provide util function for reading RAW & YUV data from the VI PIPE & CHN
*
* Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef __VO_COMM_H__
#define __VO_COMM_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <fcntl.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>

#include "hi_media_common.h"
#include "hi_common_vo.h"
#include "hi_mpi_vo.h"
#include "hi_mipi_tx.h"

#define VO_CHECK_RET(express, name)                                                              \
    do {                                                                                         \
        hi_s32 _ret;                                                                             \
        _ret = express;                                                                          \
        if (_ret != HI_SUCCESS) {                                                                \
            printf("%s failed at %s: LINE: %d with %#x!\n", name, __FUNCTION__, __LINE__, _ret); \
        }                                                                                        \
    } while (0)

#define FILE_NAME_LEN 512
#define VO_MST_ALIGN_16  16
#define VO_MST_ALIGN_2   2
#define VO_ALIGN_BACK(x, a) ((a) * (((x) / (a))))
#define VO_TEST_ALIGN_BACK(x, a) ((a) * (((x) + (a) - 1) / (a)))

#define DEV_DHD0 0
#define VO_LAYER_VHD0 0

#define CMD_COUNT 16

// typedef VB_BLK hi_vb_blk;

typedef struct {
    hi_vo_intf_sync intf_sync;
    hi_char *name;
    hi_u32 width;
    hi_u32 height;
    hi_u32 frame_rate;
} vo_mst_sync_info;

typedef struct {
    cmd_info_t cmd_info;
    hi_u32 usleep_value;
}mipi_tx_cmd_info;

typedef enum {
    HI_MIPI_TX_OUT_576P50       = HI_VO_OUT_576P50,
    HI_MIPI_TX_OUT_1024X768_60  = HI_VO_OUT_1024x768_60,
    HI_MIPI_TX_OUT_720P50       = HI_VO_OUT_720P50,
    HI_MIPI_TX_OUT_720P60       = HI_VO_OUT_720P60,
    HI_MIPI_TX_OUT_1280X1024_60 = HI_VO_OUT_1280x1024_60,
    HI_MIPI_TX_OUT_1080P24      = HI_VO_OUT_1080P24,
    HI_MIPI_TX_OUT_1080P25      = HI_VO_OUT_1080P25,
    HI_MIPI_TX_OUT_1080P30      = HI_VO_OUT_1080P30,
    HI_MIPI_TX_OUT_1080P50      = HI_VO_OUT_1080P50,
    HI_MIPI_TX_OUT_1080P60      = HI_VO_OUT_1080P60,
    HI_MIPI_TX_OUT_3840X2160_24 = HI_VO_OUT_3840x2160_24,
    HI_MIPI_TX_OUT_3840X2160_25 = HI_VO_OUT_3840x2160_25,
    HI_MIPI_TX_OUT_3840X2160_30 = HI_VO_OUT_3840x2160_30,
    HI_MIPI_TX_OUT_3840X2160_50 = HI_VO_OUT_3840x2160_50,
    HI_MIPI_TX_OUT_3840X2160_60 = HI_VO_OUT_3840x2160_60,

    HI_MIPI_TX_OUT_720X1280_60  = HI_VO_OUT_720x1280_60,
    HI_MIPI_TX_OUT_1080X1920_60 = HI_VO_OUT_1080x1920_60,

    HI_MIPI_TX_OUT_USER = HI_VO_OUT_USER,

    HI_MIPI_TX_OUT_BUTT = HI_VO_OUT_BUTT,
} mipi_tx_intf_sync;

#define ARGB_FILE_MAX_LENGTH 64

typedef struct {
    mipi_tx_intf_sync intf_sync;

    hi_u32 cmd_count;
    mipi_tx_cmd_info *cmd_info;

    combo_dev_cfg_t combo_dev_cfg;
} sample_mipi_tx_config;

typedef enum {
    VB_REMAP_MODE_NONE = 0,    /* no remap */
    VB_REMAP_MODE_NOCACHE = 1, /* no cache remap */
    VB_REMAP_MODE_CACHED = 2,  /* cache remap, if you use this mode, you should flush cache by yourself */
    VB_REMAP_MODE_BUTT
} vo_vb_remap_mode;

typedef struct {
    hi_u64 blk_size;
    hi_u32 blk_cnt;
    vo_vb_remap_mode remap_mode;
} hi_vb_pool_config;

#endif