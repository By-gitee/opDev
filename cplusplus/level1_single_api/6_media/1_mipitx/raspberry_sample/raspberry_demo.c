/**
* @File raspberry_demo.h
* @Description Provide util function for reading RAW & YUV data from the VI PIPE & CHN
*
* Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <stdio.h>
#include <sys/wait.h>
#include <sys/types.h>
#include "func.h"
#include "vo_comm.h"

hi_s32 g_exit_flag = 0;

void handler(hi_s32 flag)
{
    g_exit_flag = 1;
}

hi_s32 main(hi_s32 argc, hi_s8 *argv[])
{
    vo_mst_sync_info sync_info;
    hi_u32 vb_pool_val = -1;
    hi_s32 ret;
    char file_name[FILE_NAME_LEN];

    if (argc < 2) {
        printf("The input filename is mandatory!\n");
        return -1;
    } else {
        strcpy(file_name, argv[1]);
    }

    signal(SIGINT, handler);

    /* start vo */
    vo_get_sync_info(&sync_info);
    vo_sys_init(sync_info.height, sync_info.width);
    vo_init(DEV_DHD0, VO_LAYER_VHD0, HI_VO_INTF_MIPI, HI_VO_OUT_800x480_60, sync_info);

    ret = vo_create_vb_pool(sync_info.height, sync_info.width, &vb_pool_val);
    if (ret != HI_SUCCESS) {
        printf("start vo ... Failed!\n");
        return -1;
    }
    vo_start(sync_info.height, sync_info.width, vb_pool_val, file_name);

    /* start mipitx */
    ret = start_mipi_tx();
    if (ret != HI_SUCCESS) {
        printf("start mipitx ... Failed!\n");
        return -1;
    }

    if (!g_exit_flag) printf("\nIf you wanna stop, enter CTRL+C!\n");
    while (!g_exit_flag) {
        sleep(1);
    }

    close_mipi_tx();

    vo_deinit(DEV_DHD0, VO_LAYER_VHD0);

    vo_sys_exit();

    return -1;
}