/**
* @File vo_mem.h
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

static hi_u64 vo_mst_get_vb_blk_size(hi_u32 width, hi_u32 height)
{
    hi_u64 vb_blk_size;
    hi_u32 align_width;
    hi_u32 align_height;
    hi_u32 head_size;

    align_width = VO_TEST_ALIGN_BACK(width, VO_MST_ALIGN_16);
    align_height = VO_TEST_ALIGN_BACK(height, VO_MST_ALIGN_2);

    /* compress header stride 16 */
    head_size = VO_MST_ALIGN_16 * align_height;
    vb_blk_size = (align_width * align_height + head_size) * 2;
    return vb_blk_size;
}

/* vb init */
hi_void vo_sys_init(hi_u32 img_height, hi_u32 img_width)
{
    hi_mpi_sys_exit();
    VO_CHECK_RET(hi_mpi_sys_init();, "sys init");
}

/* vb deinit */
hi_void vo_sys_exit(hi_void)
{
    VO_CHECK_RET(hi_mpi_sys_exit(), "sys exit");
}

/* vb pool create */
hi_s32 vo_create_vb_pool(hi_u32 img_height, hi_u32 img_width, hi_u32 *blk_handle)
{
    hi_vb_pool_config pool_cfg;
    hi_u32 pool_val;

    (hi_void)memset(&pool_cfg, 0, sizeof(hi_vb_pool_config));

    pool_cfg.blk_size = vo_mst_get_vb_blk_size(img_width, img_height);
    pool_cfg.blk_cnt = 10;
    pool_cfg.remap_mode = 0;

    pool_val = hi_mpi_vo_create_pool(pool_cfg.blk_size);

    *blk_handle = pool_val;
    if (pool_val == -1) {
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

static hi_void vo_init_user_frame(hi_u32 vb_blk, hi_u32 img_height, hi_u32 img_width, hi_video_frame_info *user_frame)
{
    hi_u32 luma_size = 0;

    user_frame->v_frame.field = 4;
    user_frame->v_frame.compress_mode = 0;
    user_frame->v_frame.pixel_format = HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    user_frame->v_frame.video_format = 0;
    user_frame->v_frame.color_gamut = 0;
    user_frame->v_frame.dynamic_range = 0;
    user_frame->v_frame.height = img_height;
    user_frame->v_frame.width = img_width;
    user_frame->v_frame.width_stride[0] = VO_TEST_ALIGN_BACK(img_width, VO_MST_ALIGN_16);
    user_frame->v_frame.width_stride[1] = VO_TEST_ALIGN_BACK(img_width, VO_MST_ALIGN_16);
    user_frame->v_frame.time_ref = 0;
    user_frame->v_frame.pts = 0;

    luma_size = VO_TEST_ALIGN_BACK(user_frame->v_frame.width, VO_MST_ALIGN_16) * user_frame->v_frame.height;

    user_frame->pool_id = (vb_blk >> 16U);
    user_frame->v_frame.phys_addr[0] = hi_mpi_vo_handle_to_phys_addr(vb_blk);
    user_frame->v_frame.phys_addr[1] = user_frame->v_frame.phys_addr[0] + luma_size;
    user_frame->v_frame.header_phys_addr[0] = user_frame->v_frame.phys_addr[0];
    user_frame->v_frame.header_phys_addr[1] = user_frame->v_frame.phys_addr[1];
}

static hi_s32 loadRawImage(FILE *fp, hi_u8 *addr, hi_u32 width, hi_u32 height, hi_u32 format)
{
    hi_u32 buf_size = width * height * 3U / 2U;
    hi_u32 vb_size = buf_size;

    void *viraddr = (void*)addr;
    if (fread(viraddr, 1, vb_size, fp) != buf_size) {
       printf("buf_size should be: %d\n!", buf_size);
       return HI_FAILURE;
    }
    return HI_SUCCESS;
}

/* read image and display */
hi_void vo_start(hi_u32 img_height, hi_u32 img_width, hi_u32 vb_blk, char *file_name)
{
    hi_s32 ret;
    FILE *fd;
    hi_video_frame_info user_frame;

    vo_init_user_frame(vb_blk, img_height, img_width, &user_frame);

    fd = fopen(file_name, "rb");
    if (fd == NULL) {
        printf("open file %s fail \n", file_name);
        return;
    }

    ret = loadRawImage(fd, user_frame.v_frame.phys_addr[0], img_width, img_height, user_frame.v_frame.pixel_format);
    if (ret == HI_SUCCESS) {
        ret = hi_mpi_vo_send_frame(0, 0, &user_frame, 0);
        if (ret != HI_SUCCESS) {
            printf("Display one frame failed, ret = 0x%x!\n", ret);
        } else {
            printf("Display one frame success!\n");
        }
    }

    fclose(fd);

    sleep(1);
    return;
}