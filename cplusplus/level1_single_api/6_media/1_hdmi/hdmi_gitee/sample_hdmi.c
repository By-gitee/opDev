/**
* @File sample_hdmi.c
* @Description hdmi sample app
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
#include <fcntl.h>
#include "vo_comm.h"

hi_void vo_mipi_get_sync_info(hi_vo_intf_sync intf_sync, vo_mst_sync_info *sync_info)
{
    sync_info->intf_sync = intf_sync;
    sync_info->name = "1080P@60";
    sync_info->width = 1920;
    sync_info->height = 1080;
    sync_info->frame_rate = 60;
}

/* init device */
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
    layer_attr.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    layer_attr.display_buf_len = 4;
    layer_attr.partition_mode = HI_VO_PARTITION_MODE_SINGLE;
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
static hi_void vo_init_chn(hi_s32 layer, hi_u32 img_height, hi_u32 img_width)
{
    hi_vo_chn_attr chn_attr;

    chn_attr.rect.x = 0;
    chn_attr.rect.y = 0;
    chn_attr.rect.width = VO_ALIGN_BACK(img_width, VO_MST_ALIGN_2);
    chn_attr.rect.height = VO_ALIGN_BACK(img_height, VO_MST_ALIGN_2);
    chn_attr.priority = 0;
    chn_attr.deflicker_en = HI_FALSE;

    VO_CHECK_RET(hi_mpi_vo_set_chn_attr(layer, 0, &chn_attr), "hi_mpi_vo_set_chn_attr");
    VO_CHECK_RET(hi_mpi_vo_enable_chn(layer, 0), "hi_mpi_vo_enable_chn");
}

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
    vb_blk_size = (align_width * align_height + head_size) * 2; // HI_PIXEL_FORMAT_YVU_SEMIPLANAR_422
    return vb_blk_size;
}

/* ************** VB INIT ************** */
hi_void vo_mipi_sys_init(hi_u32 img_height, hi_u32 img_width)
{
    hi_mpi_sys_exit();
    VO_CHECK_RET(hi_mpi_sys_init(), "sys init");
}

/* ************** VB DEINIT ************** */
hi_void vo_mipi_sys_exit(hi_void)
{
    VO_CHECK_RET(hi_mpi_sys_exit(), "sys exit");
}

/* ************** VB POOL CREATE ************** */
hi_s32 vo_mipi_create_vb_pool(hi_u32 img_height, hi_u32 img_width, hi_u32 *blk_handle)
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

static hi_void vo_init_user_frame(hi_u32 vb_pool_val, hi_u32 img_height, hi_u32 img_width,
    hi_video_frame_info *user_frame)
{
    hi_u32 luma_size = 0;
    hi_u32 chroma_size = 0;

    user_frame->v_frame.field = 4;
    user_frame->v_frame.compress_mode = 0;
    user_frame->v_frame.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    user_frame->v_frame.video_format = 0;
    user_frame->v_frame.color_gamut = 0;
    user_frame->v_frame.dynamic_range = 0;
    user_frame->v_frame.height = img_height;
    user_frame->v_frame.width = img_width;

    user_frame->v_frame.width_stride[0] = img_width;
    user_frame->v_frame.width_stride[1] = img_width;
    user_frame->v_frame.width_stride[0] = VO_TEST_ALIGN_BACK(img_width, VO_MST_ALIGN_2);
    user_frame->v_frame.width_stride[1] = VO_TEST_ALIGN_BACK(img_width, VO_MST_ALIGN_2);

    user_frame->v_frame.time_ref = 0;
    user_frame->v_frame.pts = 0;

    luma_size = user_frame->v_frame.width * user_frame->v_frame.height;
    luma_size = VO_TEST_ALIGN_BACK(user_frame->v_frame.width, 2) * user_frame->v_frame.height;

    chroma_size = luma_size / 2;

    user_frame->pool_id = (vb_pool_val >> 16);
    user_frame->v_frame.phys_addr[0] = hi_mpi_vo_handle_to_phys_addr(vb_pool_val);
    user_frame->v_frame.phys_addr[1] = user_frame->v_frame.phys_addr[0] + luma_size;
    user_frame->v_frame.header_phys_addr[0] = user_frame->v_frame.phys_addr[0];
    user_frame->v_frame.header_phys_addr[1] = user_frame->v_frame.phys_addr[1];
}

hi_s32 load_vo_mipi_RawImage(FILE *fp, hi_u8 *addr, hi_u32 width, hi_u32 height, hi_u32 format)
{
    hi_u32 stride;
    hi_u32 offset;
    hi_u32 bufSize = width * height * 3 / 2;
    hi_u32 vbSize = width * height * 3;

    void *viraddr = (void *)addr;
    (hi_void)memset(viraddr, 0, (bufSize));
    size_t buf_size = fread(viraddr, 1, vbSize, fp);
    if (buf_size != bufSize) {
        printf("buf_size is: %d, but should be: %d\n!", buf_size, bufSize);
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

hi_void vo_hlt_start(hi_u32 img_height, hi_u32 img_width, hi_u32 vb_pool_val, char *file_name)
{
    img_height = 1080;
    img_width = 1920;

    hi_s32 ret;
    hi_u32 vb_blk = vb_pool_val;
    FILE *fdd;
    hi_video_frame_info user_frame;
    hi_u64 vb_blk_size = img_width * img_height * 2;

    vo_init_user_frame(vb_blk, img_height, img_width, &user_frame);

    fdd = fopen(file_name, "rb");
    if (fdd == NULL) {
        printf("open file %s fail \n", file_name);
        return;
    }

    ret = load_vo_mipi_RawImage(fdd, user_frame.v_frame.phys_addr[0], user_frame.v_frame.width,
        user_frame.v_frame.height, user_frame.v_frame.pixel_format);
    if (ret == HI_SUCCESS) {
        ret = hi_mpi_vo_send_frame(0, 0, &user_frame, 0);
        if (ret != HI_SUCCESS) {
            printf("Display one frame failed, ret = 0x%x!\n", ret);
        } else {
            printf("Display one frame success!\n");
        }
    }

    fclose(fdd);
    sleep(1);
    return;
}

hi_void vo_hdmi_init(hi_s32 dev, hi_s32 layer, hi_vo_intf_type intf_type, hi_vo_intf_sync intf_sync,
    vo_mst_sync_info sync_info)
{
    vo_init_dev(dev, intf_type, intf_sync);
    vo_init_layer(layer, sync_info.height, sync_info.width, sync_info.frame_rate);
    vo_init_chn(layer, sync_info.height, sync_info.width);
    hi_mpi_hdmi_init_sample();
    hi_mpi_hdmi_avi_infoframe_colorspace(HI_HDMI_VIDEO_FORMAT_1080P_60, 148500);
}

hi_void vo_hdmi_deinit(hi_s32 dev, hi_s32 layer)
{
    hi_mpi_vo_disable_chn(layer, 0);
    hi_mpi_vo_disable_video_layer(layer);
    VO_CHECK_RET(hi_mpi_vo_disable(dev), "hi_mpi_vo_disable");

    hi_mpi_hdmi_stop(0);
    hi_mpi_hdmi_close(0);
    hi_mpi_hdmi_deinit();
}

int main(int argc, char *argv[])
{
    hi_u32 vb_pool_val = -1;
    vo_mst_sync_info sync_info;
    int waiting_time = 10;
    char file_name[FILE_NAME_LEN];
    if (argc < 2) {
        printf("The input filename is mandatory!\n");
        return 0;
    } else {
        strcpy(file_name, argv[1]);
        if (argc == 3) {
            waiting_time = atoi(argv[2]);
        }
    }

    vo_mipi_get_sync_info(HI_VO_OUT_1080P60, &sync_info);
    vo_mipi_sys_init(sync_info.height, sync_info.width);
    vo_hdmi_init(DEV_DHD0, VO_LAYER_VHD0, HI_VO_INTF_HDMI, HI_VO_OUT_1080P60 | HI_VO_INTF_VGA, sync_info);

    vo_mipi_create_vb_pool(sync_info.height, sync_info.width, &vb_pool_val);

    vo_hlt_start(sync_info.height, sync_info.width, vb_pool_val, file_name);
    sleep(waiting_time);

    vo_hdmi_deinit(DEV_DHD0, VO_LAYER_VHD0);
    hi_mpi_vo_destroy_pool(vb_pool_val);
    vo_mipi_sys_exit();

    printf("main finish!\n");
    return 0;
}