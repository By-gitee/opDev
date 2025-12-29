/**
* @File vo_init.c
* @Description Provide util function for reading RAW & YUV data from the VI PIPE & CHN
*
* Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "func.h"
#include "vo_comm.h"

hi_void vo_get_sync_info(vo_mst_sync_info *sync_info)
{
    sync_info->intf_sync = HI_VO_OUT_800x480_60;
    sync_info->name = "800x480@60";
    sync_info->width = 800;
    sync_info->height = 480;
    sync_info->frame_rate = 60;
}

/* init device*/
static hi_void vo_init_dev(hi_s32 dev, hi_vo_intf_type intf_type, hi_vo_intf_sync intf_sync)
{
    hi_vo_pub_attr pub_attr;

    pub_attr.bg_color = 0xffffff;
    pub_attr.intf_sync = intf_sync;
    pub_attr.intf_type = intf_type;
    VO_CHECK_RET(hi_mpi_vo_set_pub_attr(dev, &pub_attr), "hi_mpi_vo_set_pub_attr");
    VO_CHECK_RET(hi_mpi_vo_enable(dev), "hi_mpi_vo_enable");
}

/* init layer */
static hi_void vo_init_layer(hi_s32 layer, hi_u32 img_height, hi_u32 img_width, hi_u32 frame_rate)
{
    hi_vo_video_layer_attr layer_attr;

    layer_attr.double_frame_en = HI_FALSE;
    layer_attr.cluster_mode_en = HI_FALSE;
    layer_attr.dst_dynamic_range = HI_DYNAMIC_RANGE_SDR8;
    layer_attr.pixel_format = HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    layer_attr.display_buf_len = 4;
    layer_attr.partition_mode = HI_VO_PARTITION_MODE_MULTI;
    layer_attr.compress_mode = 0;
    layer_attr.display_rect.width = img_width;
    layer_attr.display_rect.height = img_height;
    layer_attr.display_rect.x = 0;
    layer_attr.display_rect.y = 0;
    layer_attr.img_size.width = img_width;
    layer_attr.img_size.height = img_height;
    layer_attr.display_frame_rate = frame_rate;
    VO_CHECK_RET(hi_mpi_vo_set_video_layer_attr(layer, &layer_attr), "hi_mpi_vo_set_video_layer_attr");
    VO_CHECK_RET(hi_mpi_vo_enable_video_layer(layer), "hi_mpi_vo_enable_video_layer");
}

/* init channel */
static hi_void vo_init_chn(hi_s32 layer, hi_u32 img_height, hi_u32 img_width, hi_u32 chn_num)
{
    hi_vo_chn_attr chn_attr[chn_num];

    hi_u32 square_sort = 1;
    for (int chn = 0; chn < (square_sort * square_sort); chn++) {
        chn_attr[chn].rect.x = VO_ALIGN_BACK((img_width / square_sort) * (chn % square_sort), VO_MST_ALIGN_2);
        chn_attr[chn].rect.y = VO_ALIGN_BACK((img_height / square_sort) * (chn / square_sort), VO_MST_ALIGN_2);
        chn_attr[chn].rect.width = VO_ALIGN_BACK(img_width / square_sort, VO_MST_ALIGN_2);
        chn_attr[chn].rect.height = VO_ALIGN_BACK(img_height / square_sort, VO_MST_ALIGN_2);
        chn_attr[chn].priority = 0;
        chn_attr[chn].deflicker_en = HI_FALSE;
    }

    for (int chn = 0; chn < chn_num; chn++) {
        VO_CHECK_RET(hi_mpi_vo_set_chn_attr(layer, chn, &chn_attr[chn]), "hi_mpi_vo_set_chn_attr");
        VO_CHECK_RET(hi_mpi_vo_enable_chn(layer, chn), "hi_mpi_vo_enable_chn");
    }
}

hi_void vo_init(hi_s32 dev, hi_s32 layer, hi_vo_intf_type intf_type, hi_vo_intf_sync intf_sync, vo_mst_sync_info sync_info)
{
    vo_init_dev(dev, intf_type, intf_sync);
    vo_init_layer(layer, sync_info.height, sync_info.width, sync_info.frame_rate);
    vo_init_chn(layer, sync_info.height, sync_info.width, 1);
}

hi_void vo_deinit(hi_s32 dev, hi_s32 layer)
{
    hi_u32 chn_num = 1;
    for (int chn = 0; chn < chn_num; chn++) {
        hi_mpi_vo_disable_chn(layer, chn);
    }
    hi_mpi_vo_disable_video_layer(layer);
    VO_CHECK_RET(hi_mpi_vo_disable(dev), "hi_mpi_vo_disable");
}