#include "sample_comm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

static sample_vo_sync_info g_vo_sample_sync_info[HI_VO_OUT_BUTT] = {
    { HI_VO_OUT_PAL,        720, 576, 25 },
    { HI_VO_OUT_NTSC,       720, 480, 30 },
    { HI_VO_OUT_960H_PAL,   960, 576, 25 },
    { HI_VO_OUT_960H_NTSC,  960, 480, 30 },

    { HI_VO_OUT_640x480_60,     640,  480,  60 },
    { HI_VO_OUT_480P60,         720,  480,  60 },
    { HI_VO_OUT_576P50,         720,  576,  50 },
    { HI_VO_OUT_800x600_60,     800,  600,  60 },
    { HI_VO_OUT_1024x768_60,    1024, 768,  60 },
    { HI_VO_OUT_720P50,         1280, 720,  50 },
    { HI_VO_OUT_720P60,         1280, 720,  60 },
    { HI_VO_OUT_1280x800_60,    1280, 800,  60 },
    { HI_VO_OUT_1280x1024_60,   1280, 1024, 60 },
    { HI_VO_OUT_1366x768_60,    1366, 768,  60 },
    { HI_VO_OUT_1400x1050_60,   1400, 1050, 60 },
    { HI_VO_OUT_1440x900_60,    1440, 900,  60 },
    { HI_VO_OUT_1680x1050_60,   1680, 1050, 60 },

    { HI_VO_OUT_1080P24,    1920, 1080, 24 },
    { HI_VO_OUT_1080P25,    1920, 1080, 25 },
    { HI_VO_OUT_1080P30,    1920, 1080, 30 },
    { HI_VO_OUT_1080I50,    1920, 1080, 25 },
    { HI_VO_OUT_1080I60,    1920, 1080, 30 },
    { HI_VO_OUT_1080P50,    1920, 1080, 50 },
    { HI_VO_OUT_1080P60,    1920, 1080, 60 },

    { HI_VO_OUT_1600x1200_60,   1600, 1200, 60 },
    { HI_VO_OUT_1920x1200_60,   1920, 1200, 60 },
    { HI_VO_OUT_1920x2160_30,   1920, 2160, 30 },
    { HI_VO_OUT_2560x1440_30,   2560, 1440, 30 },
    { HI_VO_OUT_2560x1440_60,   2560, 1440, 60 },
    { HI_VO_OUT_2560x1600_60,   2560, 1600, 60 },

    { HI_VO_OUT_3840x2160_24, 3840, 2160, 24 },
    { HI_VO_OUT_3840x2160_25, 3840, 2160, 25 },
    { HI_VO_OUT_3840x2160_30, 3840, 2160, 30 },
    { HI_VO_OUT_3840x2160_50, 3840, 2160, 50 },
    { HI_VO_OUT_3840x2160_60, 3840, 2160, 60 },
    { HI_VO_OUT_4096x2160_24, 4096, 2160, 24 },
    { HI_VO_OUT_4096x2160_25, 4096, 2160, 25 },
    { HI_VO_OUT_4096x2160_30, 4096, 2160, 30 },
    { HI_VO_OUT_4096x2160_50, 4096, 2160, 50 },
    { HI_VO_OUT_4096x2160_60, 4096, 2160, 60 },
    { HI_VO_OUT_7680x4320_30, 7680, 4320, 30 },

    { HI_VO_OUT_240x320_50, 240, 320, 50 },
    { HI_VO_OUT_320x240_50, 320, 240, 50 },
    { HI_VO_OUT_240x320_60, 240, 320, 60 },
    { HI_VO_OUT_320x240_60, 320, 240, 60 },
    { HI_VO_OUT_800x600_50, 800, 600, 50 },

    { HI_VO_OUT_720x1280_60, 720, 1280, 60 },
    { HI_VO_OUT_1080x1920_60, 1080, 1920, 60 },
    { HI_VO_OUT_USER,  0,    0,    0 }
};

static const hi_hdmi_video_format g_vo_hdmi_sync[HI_VO_OUT_BUTT] = {
    HI_HDMI_VIDEO_FORMAT_PAL,
    HI_HDMI_VIDEO_FORMAT_NTSC,
    HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE,
    HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE,
    HI_HDMI_VIDEO_FORMAT_861D_640X480_60,
    HI_HDMI_VIDEO_FORMAT_480P_60,
    HI_HDMI_VIDEO_FORMAT_576P_50,
    HI_HDMI_VIDEO_FORMAT_VESA_800X600_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1024X768_60,
    HI_HDMI_VIDEO_FORMAT_720P_50,
    HI_HDMI_VIDEO_FORMAT_720P_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1280X800_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1280X1024_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1366X768_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1400X1050_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1440X900_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1680X1050_60,
    HI_HDMI_VIDEO_FORMAT_1080P_24,
    HI_HDMI_VIDEO_FORMAT_1080P_25,
    HI_HDMI_VIDEO_FORMAT_1080P_30,
    HI_HDMI_VIDEO_FORMAT_1080i_50,
    HI_HDMI_VIDEO_FORMAT_1080i_60,
    HI_HDMI_VIDEO_FORMAT_1080P_50,
    HI_HDMI_VIDEO_FORMAT_1080P_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1600X1200_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1920X1200_60,
    HI_HDMI_VIDEO_FORMAT_1920x2160_30,
    HI_HDMI_VIDEO_FORMAT_2560x1440_30,
    HI_HDMI_VIDEO_FORMAT_2560x1440_60,
    HI_HDMI_VIDEO_FORMAT_2560x1600_60,
    HI_HDMI_VIDEO_FORMAT_3840X2160P_24,
    HI_HDMI_VIDEO_FORMAT_3840X2160P_25,
    HI_HDMI_VIDEO_FORMAT_3840X2160P_30,
    HI_HDMI_VIDEO_FORMAT_3840X2160P_50,
    HI_HDMI_VIDEO_FORMAT_3840X2160P_60,
    HI_HDMI_VIDEO_FORMAT_4096X2160P_24,
    HI_HDMI_VIDEO_FORMAT_4096X2160P_25,
    HI_HDMI_VIDEO_FORMAT_4096X2160P_30,
    HI_HDMI_VIDEO_FORMAT_4096X2160P_50,
    HI_HDMI_VIDEO_FORMAT_4096X2160P_60,
    HI_HDMI_VIDEO_FORMAT_7680X4320P_30,
    HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE,
    HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE,
    HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE,
    HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE,
    HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE,
    HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE,
    HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE,
    HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE,
};

static sample_vo_wnd_info g_vo_sample_wnd_info[VO_MODE_BUTT] = {
    { VO_MODE_1MUX,   1,  1, 0, 0 },
    { VO_MODE_2MUX,   2,  2, 0, 0 },
    { VO_MODE_4MUX,   4,  2, 0, 0 },
    { VO_MODE_8MUX,   8,  3, 0, 0 },
    { VO_MODE_9MUX,   9,  3, 0, 0 },
    { VO_MODE_16MUX,  16, 4, 0, 0 },
    { VO_MODE_25MUX,  25, 5, 0, 0 },
    { VO_MODE_36MUX,  36, 6, 0, 0 },
    { VO_MODE_49MUX,  49, 7, 0, 0 },
    { VO_MODE_64MUX,  64, 8, 0, 0 },
    { VO_MODE_2X4,    8,  4, 4, 2 }
};

static hi_s32 sample_comm_vo_start_dev(hi_vo_dev vo_dev, hi_vo_pub_attr *pub_attr)
{
    hi_s32 ret;
    check_null_ptr_return(pub_attr);

    ret = hi_mpi_vo_set_pub_attr(vo_dev, pub_attr);
    if (ret != HI_SUCCESS) {
        sample_print("failed with %#x!\n", ret);
        return HI_FAILURE;
    }

    ret = hi_mpi_vo_enable(vo_dev);
    if (ret != HI_SUCCESS) {
        sample_print("failed with %#x!\n", ret);
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

static hi_s32 sample_comm_vo_get_width_height(hi_vo_intf_sync intf_sync, hi_u32 *width,
                                              hi_u32 *height, hi_u32 *frame_rate)
{
    check_null_ptr_return(width);
    check_null_ptr_return(height);
    check_null_ptr_return(frame_rate);

    if (intf_sync > HI_VO_OUT_USER) {
        sample_print("vo intf_sync %d not support!\n", intf_sync);
        return HI_FAILURE;
    }

    *width = g_vo_sample_sync_info[intf_sync].width;
    *height = g_vo_sample_sync_info[intf_sync].height;
    *frame_rate = g_vo_sample_sync_info[intf_sync].frame_rate;
    return HI_SUCCESS;
}

static hi_s32 sample_comm_vo_get_vo_layer_attr(sample_vo_cfg *vo_config, hi_vo_video_layer_attr *layer_attr)
{
    hi_s32 ret;
    hi_rect def_disp_rect = { 0, 0, 1920, 1080 };
    hi_size def_img_size = { 1920, 1080 };

    check_null_ptr_return(vo_config);
    check_null_ptr_return(layer_attr);

    ret = sample_comm_vo_get_width_height(vo_config->intf_sync, &layer_attr->display_rect.width,
        &layer_attr->display_rect.height, &layer_attr->display_frame_rate);
    if (ret != HI_SUCCESS) {
        sample_print("sample_comm_vo_get_width_height failed!\n");
        return ret;
    }

    layer_attr->cluster_mode_en = HI_FALSE;
    layer_attr->double_frame_en = HI_FALSE;
    layer_attr->pixel_format = vo_config->pix_format;
    layer_attr->display_rect.x = 0;
    layer_attr->display_rect.y = 0;

    /* set display rectangle if changed. */
    if (memcmp(&vo_config->disp_rect, &def_disp_rect, sizeof(hi_rect)) != 0) {
        layer_attr->display_rect.width = vo_config->disp_rect.width;
        layer_attr->display_rect.height = vo_config->disp_rect.height;
    }

    layer_attr->img_size.width = layer_attr->display_rect.width;
    layer_attr->img_size.height = layer_attr->display_rect.height;

    /* set image size if changed. */
    if (memcmp(&vo_config->image_size, &def_img_size, sizeof(hi_size)) != 0) {
        layer_attr->img_size.width = vo_config->image_size.width;
        layer_attr->img_size.height = vo_config->image_size.height;
    }

    layer_attr->dst_dynamic_range = vo_config->dst_dynamic_range;
    layer_attr->display_buf_len = vo_config->dis_buf_len;
    layer_attr->partition_mode = vo_config->vo_part_mode;
    layer_attr->compress_mode = vo_config->compress_mode;
    return HI_SUCCESS;
}

hi_s32 sample_comm_vo_start_layer(hi_vo_layer vo_layer, const hi_vo_video_layer_attr *layer_attr)
{
    hi_s32 ret;

    check_null_ptr_return(layer_attr);

    ret = hi_mpi_vo_set_video_layer_attr(vo_layer, layer_attr);
    if (ret != HI_SUCCESS) {
        sample_print("vo_layer(%d) failed with %#x!\n", vo_layer, ret);
        return HI_FAILURE;
    }

    sample_print("layer_attr: layer_attr.display_rect.x =%d,layer_attr.display_rect.y=%d,"
                 "layer_attr.display_rect.width=%d, layer_attr.display_rect.height=%d "
                 "layer_attr.img_size.width=%d,layer_attr.img_size.height=%d,partition_mode:%d\n",
                 layer_attr->display_rect.x, layer_attr->display_rect.y, 
                 layer_attr->display_rect.width, layer_attr->display_rect.height,
                 layer_attr->img_size.width, layer_attr->img_size.height, layer_attr->partition_mode);

    ret = hi_mpi_vo_enable_video_layer(vo_layer);
    if (ret != HI_SUCCESS) {
        sample_print("failed with %#x!\n", ret);
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

static hi_s32 sample_comm_vo_get_wnd_info(sample_vo_mode mode, sample_vo_wnd_info *wnd_info)
{
    check_null_ptr_return(wnd_info);

    if (mode >= VO_MODE_BUTT) {
        sample_print("vo mode %d not support!\n", mode);
        return HI_FAILURE;
    }

    wnd_info->mode = mode;
    wnd_info->wnd_num = g_vo_sample_wnd_info[mode].wnd_num;
    wnd_info->square = g_vo_sample_wnd_info[mode].square;
    wnd_info->row = g_vo_sample_wnd_info[mode].row;
    wnd_info->col = g_vo_sample_wnd_info[mode].col;
    return HI_SUCCESS;
}

hi_s32 sample_comm_vo_get_chn_attr(sample_vo_wnd_info *wnd_info, hi_vo_video_layer_attr *layer_attr, hi_s32 chn,
    hi_vo_chn_attr *chn_attr)
{
    hi_u32 width;
    hi_u32 height;

    check_null_ptr_return(wnd_info);
    check_null_ptr_return(layer_attr);
    check_null_ptr_return(chn_attr);

    width = layer_attr->img_size.width;
    height = layer_attr->img_size.height;
    if (wnd_info->mode != VO_MODE_2X4) {
        chn_attr->rect.x = HI_ALIGN_DOWN((width / wnd_info->square) * (chn % wnd_info->square), 2);  /* 2: 2 align */
        chn_attr->rect.y = HI_ALIGN_DOWN((height / wnd_info->square) * (chn / wnd_info->square), 2); /* 2: 2 align */
        chn_attr->rect.width = HI_ALIGN_DOWN(width / wnd_info->square, 2);                           /* 2: 2 align */
        chn_attr->rect.height = HI_ALIGN_DOWN(height / wnd_info->square, 2);                         /* 2: 2 align */
    } else {
        chn_attr->rect.x = HI_ALIGN_DOWN((width / wnd_info->col) * (chn % wnd_info->col), 2);  /* 2: 2 align */
        chn_attr->rect.y = HI_ALIGN_DOWN((height / wnd_info->row) * (chn / wnd_info->col), 2); /* 2: 2 align */
        chn_attr->rect.width = HI_ALIGN_DOWN(width / wnd_info->col, 2);                        /* 2: 2 align */
        chn_attr->rect.height = HI_ALIGN_DOWN(height / wnd_info->row, 2);                      /* 2: 2 align */
    }

    chn_attr->priority = 0;
    chn_attr->deflicker_en = HI_FALSE;
    return HI_SUCCESS;
}

hi_s32 sample_comm_vo_start_chn(hi_vo_layer vo_layer, sample_vo_mode mode, hi_vo_chn_param *chn_param)
{
    hi_s32 ret;
    hi_s32 i;
    sample_vo_wnd_info wnd_info;
    hi_vo_chn_attr chn_attr;
    hi_vo_video_layer_attr layer_attr;

    memset(&wnd_info, 0, sizeof(sample_vo_wnd_info));

    ret = sample_comm_vo_get_wnd_info(mode, &wnd_info);
    if (ret != HI_SUCCESS) {
        sample_print("failed with %#x!\n", ret);
        return ret;
    }

    ret = hi_mpi_vo_get_video_layer_attr(vo_layer, &layer_attr);
    if (ret != HI_SUCCESS) {
        sample_print("failed with %#x!\n", ret);
        return HI_FAILURE;
    }

    sample_print("sample_comm_vo_start_chn,wnd_num is %d!\n", wnd_info.wnd_num);

    for (i = 0; i < (hi_s32)wnd_info.wnd_num; i++) {
        ret = sample_comm_vo_get_chn_attr(&wnd_info, &layer_attr, i, &chn_attr);
        if (ret != HI_SUCCESS) {
            sample_print("failed with %#x!\n", ret);
            return HI_FAILURE;
        }

        sample_print("chn_attr--- x:%d, y: %d, width:%d, height:%d \n",
            chn_attr.rect.x, chn_attr.rect.y, chn_attr.rect.width, chn_attr.rect.height);

        ret = hi_mpi_vo_set_chn_attr(vo_layer, i, &chn_attr);
        if (ret != HI_SUCCESS) {
            sample_print("failed with %#x!\n", ret);
            return HI_FAILURE;
        }

        ret = hi_mpi_vo_set_chn_param(vo_layer, i, chn_param);
        if (ret != HI_SUCCESS) {
            sample_print("hi_mpi_vo_set_chn_param failed with %#x!\n", ret);
            return HI_FAILURE;
        }

        ret = hi_mpi_vo_enable_chn(vo_layer, i);
        if (ret != HI_SUCCESS) {
            sample_print("failed with %#x!\n", ret);
            return HI_FAILURE;
        }
    }

    return HI_SUCCESS;
}

hi_s32 sample_comm_vo_stop_layer(hi_vo_layer vo_layer)
{
    hi_s32 ret;

    ret = hi_mpi_vo_disable_video_layer(vo_layer);
    if (ret != HI_SUCCESS) {
        sample_print("failed with %#x!\n", ret);
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

hi_s32 sample_comm_vo_stop_dev(hi_vo_dev vo_dev)
{
    hi_s32 ret;

    ret = hi_mpi_vo_disable(vo_dev);
    if (ret != HI_SUCCESS) {
        sample_print("failed with %#x!\n", ret);
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

hi_s32 sample_comm_vo_hdmi_init(hi_void)
{
    check_return(hi_mpi_hdmi_init(), "hi_mpi_hdmi_init");
    return HI_SUCCESS;
}

static hi_void sample_comm_vo_hdmi_convert_sync(hi_vo_intf_sync intf_sync, hi_hdmi_video_format *video_fmt)
{
    *video_fmt = HI_HDMI_VIDEO_FORMAT_1080P_60;
    if ((intf_sync >= HI_VO_OUT_PAL) && (intf_sync < HI_VO_OUT_BUTT)) {
        *video_fmt = g_vo_hdmi_sync[intf_sync];
    }
}

hi_s32 sample_comm_vo_hdmi_start(hi_hdmi_id hdmi_id, hi_vo_intf_sync intf_sync)
{
    hi_hdmi_attr attr;
    hi_hdmi_video_format video_fmt = HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE;

    memset(&attr, 0, sizeof(hi_hdmi_attr));

    sample_comm_vo_hdmi_convert_sync(intf_sync, &video_fmt);

    sample_print("sample_comm_vo_hdmi_start begin, hdmi: %d\n",hdmi_id);
    check_return(hi_mpi_hdmi_open(hdmi_id), "hi_mpi_hdmi_open");

    check_return(hi_mpi_hdmi_get_attr(hdmi_id, &attr), "hi_mpi_hdmi_get_attr");
    attr.video_format = video_fmt;
    attr.pix_clk =  148500;  /* 148500: hdmi pix clk */
    check_return(hi_mpi_hdmi_set_attr(hdmi_id, &attr), "hi_mpi_hdmi_set_attr");
    check_return(hi_mpi_hdmi_start(hdmi_id), "hi_mpi_hdmi_start");
    sample_print("sample_comm_vo_hdmi_start end\n");
    return HI_SUCCESS;
}

hi_void sample_comm_vo_config_intf_type(sample_vo_cfg *vo_config, hi_vo_pub_attr *vo_pub_attr)
{
    /* start hdmi device. note : do this after vo device started. */
    sample_print("sample_comm_vo_config_intf_type,vo_config->vo_intf_type is %d!\n",vo_config->vo_intf_type);
    if ((vo_config->vo_intf_type & HI_VO_INTF_HDMI) || (vo_config->vo_intf_type & HI_VO_INTF_HDMI1)) {
        sample_comm_vo_hdmi_init();
    }

    if (vo_config->vo_intf_type & HI_VO_INTF_HDMI) {
        (hi_void)sample_comm_vo_hdmi_start(vo_config->hdmi_id, vo_config->intf_sync);
    }

    if (vo_config->vo_intf_type & HI_VO_INTF_HDMI1) {
        (hi_void)sample_comm_vo_hdmi_start(vo_config->hdmi1_id, vo_config->intf_sync);
    }
}

hi_s32 sample_comm_vo_start_vo(sample_vo_cfg *vo_config)
{
    /* VO device vo_dev# information declaration. */
    hi_s32 ret;
    hi_vo_dev vo_dev;
    hi_vo_layer vo_layer;
    hi_vo_pub_attr vo_pub_attr = { 0 };
    hi_vo_video_layer_attr layer_attr = { 0 };

    check_null_ptr_return(vo_config);

    vo_dev = vo_config->vo_dev;
    vo_layer = vo_config->vo_layer;

    /* set and start VO device vo_dev#. */
    vo_pub_attr.intf_type = vo_config->vo_intf_type;
    vo_pub_attr.intf_sync = vo_config->intf_sync;
    vo_pub_attr.bg_color = vo_config->bg_color;
    ret = sample_comm_vo_start_dev(vo_dev, &vo_pub_attr);
    if (ret != HI_SUCCESS) {
        sample_print("sample_comm_vo_start_dev failed!\n");
        return ret;
    }

    ret = sample_comm_vo_get_vo_layer_attr(vo_config, &layer_attr);
    if (ret != HI_SUCCESS) {
        sample_print("sample_comm_vo_start get video layer failed!\n");
        (hi_void)sample_comm_vo_stop_dev(vo_dev);
        return ret;
    }

    ret = sample_comm_vo_start_layer(vo_layer, &layer_attr);
    if (ret != HI_SUCCESS) {
        sample_print("sample_comm_vo_start set video layer failed!\n");
        (hi_void)sample_comm_vo_stop_dev(vo_dev);
        return ret;
    }

    /* start vo channels. */
    ret = sample_comm_vo_start_chn(vo_layer, vo_config->vo_mode, &vo_config->chn_param);
    if (ret != HI_SUCCESS) {
        sample_print("sample_comm_vo_start_chn failed!\n");
        (hi_void)sample_comm_vo_stop_layer(vo_layer);
        (hi_void)sample_comm_vo_stop_dev(vo_dev);
        return ret;
    }

    sample_print("sample_comm_vo_config_intf_type begin!\n");
    sample_comm_vo_config_intf_type(vo_config, &vo_pub_attr);
    sample_print("sample_comm_vo_config_intf_type end!\n");

    return HI_SUCCESS;
}

hi_s32 sample_comm_vo_hdmi_stop(hi_hdmi_id hdmi_id)
{
    check_return(hi_mpi_hdmi_stop(hdmi_id), "hi_mpi_hdmi_stop");
    check_return(hi_mpi_hdmi_close(hdmi_id), "hi_mpi_hdmi_close");
    return HI_SUCCESS;
}

hi_s32 sample_comm_vo_hdmi_deinit(hi_void)
{
    check_return(hi_mpi_hdmi_deinit(), "hi_mpi_hdmi_deinit");
    return HI_SUCCESS;
}

hi_s32 sample_comm_vo_stop_chn(hi_vo_layer vo_layer, sample_vo_mode mode)
{
    hi_s32 ret;
    hi_s32 i;
    sample_vo_wnd_info wnd_info;

    memset(&wnd_info, 0, sizeof(sample_vo_wnd_info));

    ret = sample_comm_vo_get_wnd_info(mode, &wnd_info);
    if (ret != HI_SUCCESS) {
        sample_print("failed with %#x!\n", ret);
        return ret;
    }

    for (i = 0; i < (hi_s32)wnd_info.wnd_num; i++) {
        ret = hi_mpi_vo_disable_chn(vo_layer, i);
        if (ret != HI_SUCCESS) {
            sample_print("failed with %#x!\n", ret);
            return HI_FAILURE;
        }
    }

    return ret;
}

hi_s32 sample_comm_vo_stop_vo(const sample_vo_cfg *vo_config)
{
    hi_s32 ret;
    hi_vo_dev vo_dev;
    hi_vo_layer vo_layer;
    sample_vo_mode vo_mode;

    check_null_ptr_return(vo_config);

    vo_dev = vo_config->vo_dev;
    vo_layer = vo_config->vo_layer;
    vo_mode = vo_config->vo_mode;

    if (vo_config->vo_intf_type & HI_VO_INTF_HDMI) {
        sample_comm_vo_hdmi_stop(vo_config->hdmi_id);
    }

    if (vo_config->vo_intf_type & HI_VO_INTF_HDMI1) {
        sample_comm_vo_hdmi_stop(vo_config->hdmi1_id);
    }

    if ((vo_config->vo_intf_type & HI_VO_INTF_HDMI) || (vo_config->vo_intf_type & HI_VO_INTF_HDMI1)) {
        sample_comm_vo_hdmi_deinit();
    }

    ret = sample_comm_vo_stop_chn(vo_layer, vo_mode);
    if (ret != HI_SUCCESS) {
        sample_print("sample_comm_vo_stop_vo stop chn failed!\n");
        return ret;
    }

    ret = sample_comm_vo_stop_layer(vo_layer);
    if (ret != HI_SUCCESS) {
        sample_print("sample_comm_vo_stop_vo stop layer failed!\n");
        return ret;
    }

    ret = sample_comm_vo_stop_dev(vo_dev);
    if (ret != HI_SUCCESS) {
        sample_print("sample_comm_vo_stop_vo stop dev failed!\n");
        return ret;
    }

    return HI_SUCCESS;
}