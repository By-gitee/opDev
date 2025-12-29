/**
* @File hi_mpi_vo.c
* @Description hdmi info init functions
*
* Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <unistd.h>
#include "vo_comm.h"

hi_void hi_mpi_hdmi_init_sample(void)
{
    hi_s32 ret = hi_mpi_hdmi_init();
    if (ret != HI_SUCCESS) {
        printf("hi_mpi_hdmi_init_sample error: %d", ret);
        return;
    }
    ret = hi_mpi_hdmi_open(0);
    if (ret != HI_SUCCESS) {
        printf("hi_mpi_hdmi_open error: %d\n", ret);
        return;
    }
}

hi_s32 hi_mpi_hdmi_set_info(hi_hdmi_attr attr, int hdmi_timing)
{
    hi_s32 ret = 0;
    hi_hdmi_infoframe infoframe;

    ret = hi_mpi_hdmi_set_attr(0, &attr);
    if (ret != HI_SUCCESS) {
        printf("hi_mpi_hdmi_set_attr error: %d\n", ret);
        return HI_FAILURE;
    }

    infoframe.infoframe_type = HI_INFOFRAME_TYPE_AVI;
    hi_hdmi_avi_infoframe *avi = &infoframe.infoframe_unit.avi_infoframe;
    user_set_avi_infoframe_pattern(avi, hdmi_timing);
    avi->color_space = HI_HDMI_COLOR_SPACE_YCBCR444;
    ret = hi_mpi_hdmi_set_infoframe(0, &infoframe);
    if (ret != HI_SUCCESS) {
        printf("[avi]hi_mpi_hdmi_set_infoframe error: %d\n", ret);
        return HI_FAILURE;
    }

    infoframe.infoframe_type = HI_INFOFRAME_TYPE_AUDIO;
    hi_hdmi_audio_infoframe *audio = &infoframe.infoframe_unit.audio_infoframe;
    user_set_audio_infoframe(audio);
    ret = hi_mpi_hdmi_set_infoframe(0, &infoframe);
    if (ret != HI_SUCCESS) {
        printf("[audio]hi_mpi_hdmi_set_infoframe error: %d\n", ret);
        return HI_FAILURE;
    }
    sleep(1);

    return HI_SUCCESS;
}

hi_s32 hi_mpi_hdmi_avi_infoframe_colorspace(int hdmi_timing, int pix_clk)
{
    hi_s32 ret = 0;

    hi_hdmi_attr attr;
    attr.hdmi_en = HI_TRUE;
    attr.video_format = HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE;
    attr.deep_color_mode = HI_HDMI_DEEP_COLOR_24BIT;
    attr.audio_en = HI_TRUE;
    attr.sample_rate = HI_HDMI_SAMPLE_RATE_48K;
    attr.bit_depth = HI_HDMI_BIT_DEPTH_16;
    attr.auth_mode_en = HI_FALSE;
    attr.deep_color_adapt_en = HI_TRUE; // 根据vo输出自动调整hdmi色域打开
    attr.pix_clk = pix_clk;

    ret = hi_mpi_hdmi_set_info(attr, hdmi_timing);
    if (ret != HI_SUCCESS) {
        return HI_FAILURE;
    }

    ret = hi_mpi_hdmi_set_info(attr, hdmi_timing);
    if (ret != HI_SUCCESS) {
        return HI_FAILURE;
    }

    ret = hi_mpi_hdmi_start(0);
    if (ret != HI_SUCCESS) {
        printf("hi_mpi_hdmi_start error: %d\n", ret);
        return HI_FAILURE;
    }
    sleep(1);

    return HI_SUCCESS;
}

void user_set_avi_infoframe_pattern(hi_hdmi_avi_infoframe *avi, int timing)
{
    avi->timing_mode = timing;

    avi->color_space = HI_HDMI_COLOR_SPACE_RGB444;
    avi->active_info_present = HI_FALSE; // Active Format Aspect Ratio
    avi->bar_info = HI_HDMI_BAR_INFO_NOT_VALID;
    avi->scan_info = HI_HDMI_SCAN_INFO_NO_DATA;
    avi->colorimetry = HI_HDMI_COMMON_COLORIMETRY_ITU601;
    avi->ex_colorimetry = HI_HDMI_COMMON_COLORIMETRY_XVYCC_601;
    avi->aspect_ratio = HI_HDMI_PIC_ASPECT_RATIO_4TO3;
    avi->active_aspect_ratio = HI_HDMI_ACTIVE_ASPECT_RATIO_SAME_PIC;
    avi->pic_scaling = HI_HDMI_PIC_NON_UNIFORM_SCALING;
    avi->rgb_quant = HI_HDMI_RGB_QUANT_FULL_RANGE;
    avi->is_it_content = HI_FALSE;
    avi->pixel_repetition = HI_HDMI_PIXEL_REPET_NO;
    avi->content_type = HI_HDMI_CONTNET_PHOTO;
    avi->ycc_quant = HI_HDMI_YCC_QUANT_FULL_RANGE;
    avi->line_n_end_of_top_bar = 0;
    avi->line_n_start_of_bot_bar = 0;
    avi->pixel_n_end_of_left_bar = 0;
    avi->pixel_n_start_of_right_bar = 0;
}

void user_set_audio_infoframe(hi_hdmi_audio_infoframe *audio)
{
    audio->chn_cnt = HI_HDMI_AUDIO_CHN_CNT_2;
    audio->coding_type = HI_HDMI_AUDIO_CODING_PCM;
    audio->sample_size = HI_HDMI_AUDIO_SAMPLE_SIZE_16;
    audio->sampling_freq = HI_HDMI_AUDIO_SAMPLE_FREQ_48000;
    audio->chn_alloc = 0; /* Channel/Speaker Allocation.Range [0,255] */
    audio->level_shift = HI_HDMI_LEVEL_SHIFT_VAL_0_DB;
    audio->lfe_playback_level = HI_HDMI_LFE_PLAYBACK_NO;
    audio->down_mix_inhibit = HI_FALSE;
}