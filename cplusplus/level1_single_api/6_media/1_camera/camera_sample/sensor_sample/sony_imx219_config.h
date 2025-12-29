/**
* @file imx219_sensor_config.c
*
* Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "hi_mipi_rx.h"
#include "hi_mpi_isp.h"
#include "hi_mpi_vi.h"

#include "sensor_management.h"

sns_clk_cfg_t g_mipi_sns_clk_cfg_attr = {
    .clk_source = 0,
    .clk_freq = SENSOR_CLK_37P125MHz
};

combo_dev_attr_t MIPI_2lane_CHN0_SENSOR_IMX219_10BIT_24M_NOWDR_ATTR =
{
    .devno = 0,
    .input_mode = INPUT_MODE_MIPI,
    .data_rate = MIPI_DATA_RATE_X1,
    .img_rect = {0, 0, 3264, 2448},

    {
        .mipi_attr =
        {
            DATA_TYPE_RAW_10BIT,
            HI_MIPI_WDR_MODE_NONE,
            {0, 2, -1, -1, -1, -1, -1, -1}
        }
    }
};

combo_dev_attr_t MIPI_2lane_CHN0_SENSOR_IMX219_10BIT_1920x1080_NOWDR_ATTR =
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

hi_isp_pub_attr ISP_PUB_ATTR_IMX219_24M_20FPS_raw10 =
{
    {0, 0, 3264, 2448},
    {3264, 2448},
    20,
    HI_ISP_BAYER_RGGB,
    HI_WDR_MODE_NONE,
    0,   // IMX219_RES_MODE_E raw10模式对应
};

hi_isp_pub_attr ISP_PUB_ATTR_IMX219_1920x1080_20FPS_raw10 =
{
    {0, 0, 1920, 1080},
    {1920, 1080},
    20,
    HI_ISP_BAYER_RGGB,
    HI_WDR_MODE_NONE,
    0,   // IMX219_RES_MODE_E raw10模式对应
};

hi_isp_pub_attr ISP_PUB_ATTR_IMX219_1920x1080_30FPS_raw10 =
{
    {0, 0, 1920, 1080},
    {1920, 1080},
    30,
    HI_ISP_BAYER_RGGB,
    HI_WDR_MODE_NONE,
    2,   // IMX219_RES_MODE_E raw10 1920x1080 30 fps 模式对应
};

// hi_vi_dev_attr 3264x2448
hi_vi_dev_attr DEV_ATTR_IMX219_24M =
{
    HI_VI_MODE_MIPI,

    HI_VI_SCAN_PROGRESSIVE,

    HI_VI_DATA_SEQ_YUYV,

    HI_VI_DATA_TYPE_RAW,
    {3264, 2448},
    {
        HI_WDR_MODE_NONE,
        2448
    },
    HI_DATA_RATE_X1
};

hi_vi_dev_attr DEV_ATTR_IMX219_1920x1080 =
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

hi_vi_pipe_attr IMX219_PIPE_ATTR_3264x2448_RAW10_420_3DNR_RFR =
{
    HI_VI_PIPE_BYPASS_NONE,
    HI_FALSE,
    {3264, 2448},
    HI_PIXEL_FORMAT_RGB_BAYER_10BPP,
    HI_COMPRESS_MODE_NONE,
    HI_DATA_BIT_WIDTH_10,
    { -1, -1}
};

hi_vi_pipe_attr IMX219_PIPE_ATTR_1920x1080_RAW10_420_3DNR_RFR =
{
    HI_VI_PIPE_BYPASS_NONE,
    HI_FALSE,
    {1920, 1080},
    HI_PIXEL_FORMAT_RGB_BAYER_10BPP,
    HI_COMPRESS_MODE_NONE,
    HI_DATA_BIT_WIDTH_10,
    { -1, -1}
};

hi_vi_chn_attr IMX219_CHN_ATTR_3264x2448_420_SDR8_LINEAR =
{
    {3264, 2448},
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420,
    HI_DYNAMIC_RANGE_SDR8,
    HI_VIDEO_FORMAT_LINEAR,
    HI_COMPRESS_MODE_NONE,
    // 0,      0,
    0,
    { -1, -1}
};

hi_vi_chn_attr IMX219_CHN_ATTR_1920x1080_420_SDR8_LINEAR =
{
    {1920, 1080},
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420,
    HI_DYNAMIC_RANGE_SDR8,
    HI_VIDEO_FORMAT_LINEAR,
    HI_COMPRESS_MODE_NONE,
    // 0,      0,
    0,
    { -1, -1}
};