/**
* @file imx477_sensor_config.c
*
* Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "hi_mipi_rx.h"
#include "hi_common_isp.h"
#include "hi_common_vi.h"

#include "sensor_management.h"

combo_dev_attr_t MIPI_2lane_CHN0_SENSOR_IMX477_12BIT_1920x1080_NOWDR_ATTR =
{
    .devno = 0,
    .input_mode = INPUT_MODE_MIPI,
    .data_rate = MIPI_DATA_RATE_X1,
    .img_rect = {0, 0, 1920, 1080},

    {
        .mipi_attr =
        {
            DATA_TYPE_RAW_12BIT,
            HI_MIPI_WDR_MODE_NONE,
            {0, 2, -1, -1, -1, -1, -1, -1}
        }
    }
};

combo_dev_attr_t MIPI_2lane_CHN0_SENSOR_IMX477_10BIT_1920x1080_DOL2_ATTR =
{
    .devno = 0,
    .input_mode = INPUT_MODE_MIPI,
    .data_rate = MIPI_DATA_RATE_X1,
    .img_rect = {0, 0, 1920, 1080},

    {
        .mipi_attr =
        {
            DATA_TYPE_RAW_10BIT,
            HI_MIPI_WDR_MODE_NONE,
            {0, 2, -1, -1, -1, -1, -1, -1}
        }
    }
};

hi_isp_pub_attr ISP_PUB_ATTR_IMX477_1920x1080_25FPS_NOWDR_RAW12 =
{
    {0, 0, 1920, 1080},
    {1920, 1080},
    30,
    HI_ISP_BAYER_RGGB,
    HI_WDR_MODE_NONE,
    3,   // IMX477_4056x3040_RAW10_10FPS
};

hi_isp_pub_attr ISP_PUB_ATTR_IMX477_1920x1080_10FPS_DOL2_RAW10 =
{
    {0, 0, 1920, 1080},
    {1920, 1080},
    15,
    HI_ISP_BAYER_RGGB,
    HI_WDR_MODE_2To1_LINE,
    4,   // IMX477_1920x1080_DOL2_RAW10_15fps
};

hi_vi_dev_attr DEV_ATTR_IMX477_1920x1080_NOWDR =
{
    HI_VI_MODE_MIPI,

    HI_VI_SCAN_PROGRESSIVE,

    HI_VI_DATA_SEQ_YUYV,

    HI_VI_DATA_TYPE_RAW,
    {1920, 1080},
    {
        HI_WDR_MODE_NONE,
        1920
    },
    HI_DATA_RATE_X1
};

hi_vi_dev_attr DEV_ATTR_IMX477_1920x1080_DOL2 =
{
    HI_VI_MODE_MIPI,

    HI_VI_SCAN_PROGRESSIVE,

    HI_VI_DATA_SEQ_YUYV,

    HI_VI_DATA_TYPE_RAW,
    {1920, 1080},
    {
        HI_WDR_MODE_2To1_LINE,
        1920
    },
    HI_DATA_RATE_X1
};

hi_vi_pipe_attr IMX477_PIPE_ATTR_1920x1080_RAW10_420_3DNR_RFR =
{
    HI_VI_PIPE_BYPASS_NONE,
    HI_FALSE,
    {1920, 1080},
    HI_PIXEL_FORMAT_RGB_BAYER_10BPP,
    HI_COMPRESS_MODE_NONE,
    HI_DATA_BIT_WIDTH_10,
    { -1, -1}
};

hi_vi_chn_attr IMX477_CHN_ATTR_1920x1080_420_SDR8_LINEAR =
{
    {1920, 1080},
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420,
    HI_DYNAMIC_RANGE_SDR8,
    HI_VIDEO_FORMAT_LINEAR,
    HI_COMPRESS_MODE_NONE,
    0,
    { -1, -1}
};
