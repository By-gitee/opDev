/**
* @File fun.h
* @Description Provide util function for reading RAW & YUV data from the VI PIPE & CHN
*
* Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef __FUNC_H__
#define __FUNC_H__

#include "vo_comm.h"

hi_void vo_get_sync_info(vo_mst_sync_info *sync_info);
hi_void vo_sys_init(hi_u32 img_height, hi_u32 img_width);
hi_void vo_sys_exit(hi_void);
hi_void vo_init(hi_s32 dev, hi_s32 layer, hi_vo_intf_type intf_type, hi_vo_intf_sync intf_sync, vo_mst_sync_info sync_info);
hi_void vo_deinit(hi_s32 dev, hi_s32 layer);
hi_s32 start_mipi_tx();
hi_s32 close_mipi_tx();
hi_s32 vo_create_vb_pool(hi_u32 img_height, hi_u32 img_width, hi_u32 *blk_handle);
hi_void vo_start(hi_u32 img_height, hi_u32 img_width, hi_u32 vb_blk, char *file_name);
#endif