#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/poll.h>
#include <sys/time.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <getopt.h>

#include "sample_comm.h"
#include "region_param_parser_ext.h"

#define SAMPLE_STREAM_PATH "./"

char gInputFileName[500] = "infile";
hi_u32 gStreamWidth = 1920;
hi_u32 gStreamHeight = 1088;
hi_u32 gOutWidth = 1920;
hi_u32 gOutHeight = 1088;
hi_u32 gCropWidth = 1920;
hi_u32 gCropHeight = 1088;
hi_u32 gCropX = 0;
hi_u32 gCropY = 0;
hi_u32 gCropMode = 0;
hi_bool gCropEnable = HI_FALSE;
hi_u32 gAspectRatioMode = 0;
hi_u32 gAspectRatioWidth = 1920;
hi_u32 gAspectRatioHeight = 1088;
hi_u32 gAspectRatioX = 0;
hi_u32 gAspectRatioY = 0;
hi_u32 gBgColor = 0;
hi_bool gVpssChn0Enable = HI_TRUE;
hi_bool gVpssChn1Enable = HI_FALSE;
hi_u32 gAutoNum = 1;
hi_u32 gTestType = 0;
hi_u32 gWriteFile = 1;
hi_u32 gUserNum = 1;
hi_u32 gRefNum = 5;
hi_u32 gDisplayNum = 2;
hi_bool gRegionEnable = HI_FALSE;
hi_u32 gGroupId = VPSS_AI_START_GRP;
hi_u32 gOutFormat = (hi_u32)HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;

static vdec_display_cfg g_vdec_display_cfg = {
    .pic_size = PIC_1080P,
    .intf_sync = HI_VO_OUT_1080P60,
    .intf_type = HI_VO_INTF_HDMI,
};

static hi_size g_disp_size;

pthread_t g_vpss_get_thread[HI_VPSS_MAX_GET_FRAME_CHN_NUM] = {0};
uint32_t  g_vpss_get_exit_state[HI_VPSS_MAX_GET_FRAME_CHN_NUM] = {0};
uint32_t g_temp_vpss_grp_num[HI_VPSS_MAX_GET_FRAME_CHN_NUM] =  {0};

static sample_vo_grp_vo_mode_info g_vo_grp_vo_mode_info[VO_MODE_BUTT] = {
    { 1, VO_MODE_1MUX},
    { 2, VO_MODE_2MUX},
    { 4, VO_MODE_4MUX},
    { 8, VO_MODE_8MUX},
    { 9, VO_MODE_9MUX},
    { 16, VO_MODE_16MUX},
    { 25, VO_MODE_25MUX},
    { 36, VO_MODE_36MUX},
    { 49, VO_MODE_49MUX},
    { 64, VO_MODE_64MUX},
    { 888, VO_MODE_2X4}
};

static hi_s32 g_sample_exit = 0;

static hi_s32 sample_getchar()
{
    int c;
    if (g_sample_exit == 1) {
        return 'e';
    }

    c = getchar();

    if (g_sample_exit == 1) {
        return 'e';
    }
    return c;
}

static hi_s32 sample_init_module_vb(sample_vdec_attr *sample_vdec, hi_u32 vdec_chn_num, hi_payload_type type,
    hi_u32 len)
{
    hi_u32 i;
    hi_s32 ret;
    for (i = 0; (i < vdec_chn_num) && (i < len); i++) {
        sample_vdec[i].type                        = type;
        sample_vdec[i].width                       = gStreamWidth;
        sample_vdec[i].height                      = gStreamHeight;
        sample_vdec[i].mode                        = HI_VDEC_SEND_MODE_FRAME;
        sample_vdec[i].sample_vdec_video.dec_mode  = HI_VIDEO_DEC_MODE_IPB;
        sample_vdec[i].sample_vdec_video.bit_width = HI_DATA_BIT_WIDTH_8;
        if (type == HI_PT_JPEG) {
            sample_vdec[i].sample_vdec_video.ref_frame_num = 0;
        } else {
            sample_vdec[i].sample_vdec_video.ref_frame_num = gRefNum;
        }
        sample_vdec[i].display_frame_num                   = gDisplayNum;
        sample_vdec[i].frame_buf_cnt = (type == HI_PT_JPEG) ? (sample_vdec[i].display_frame_num + 1) :
            (sample_vdec[i].sample_vdec_video.ref_frame_num + sample_vdec[i].display_frame_num + 1);
        if (type == HI_PT_JPEG) {
            sample_vdec[i].sample_vdec_picture.pixel_format = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
            sample_vdec[i].sample_vdec_picture.alpha        = 255; /* 255:pic alpha value */
        }
    }
    return ret;
}

static hi_s32 sample_init_sys_and_vb(sample_vdec_attr *sample_vdec, hi_u32 vdec_chn_num,
                                     hi_payload_type type, hi_u32 len)
{
    hi_s32 ret;
    ret = sample_comm_sys_get_pic_size(g_vdec_display_cfg.pic_size, &g_disp_size);
    if (ret != HI_SUCCESS) {
        sample_print("sys get pic size fail for %#x!\n", ret);
        return ret;
    }
    ret = sample_comm_sys_init();
    if (ret != HI_SUCCESS) {
        sample_print("init sys fail for %#x!\n", ret);
        sample_comm_sys_exit();
        return ret;
    }
    ret = sample_init_module_vb(&sample_vdec[0], vdec_chn_num, type, len);
    if (ret != HI_SUCCESS) {
        sample_print("init mod vb fail for %#x!\n", ret);
        sample_comm_sys_exit();
        return ret;
    }
    return ret;
}

static hi_s32 sample_init_region(hi_void)
{
    hi_s32 ret = HI_SUCCESS;
    hi_u32 mosaic_region_num = get_mosaic_region_num();
    if (mosaic_region_num != 0) {
        ret = sample_comm_region_create(mosaic_region_num, HI_RGN_MOSAIC);
    }
    return ret;
}

static hi_s32 sample_start_vdec(sample_vdec_attr *sample_vdec, hi_u32 vdec_chn_num, hi_u32 len)
{
    hi_s32 ret;
    hi_u32 i;

    ret = sample_comm_vdec_start(vdec_chn_num, &sample_vdec[0], len);
    if (ret != HI_SUCCESS) {
        sample_print("start VDEC fail for %#x!\n", ret);
        sample_comm_vdec_stop(vdec_chn_num);
        return ret;
    }
    return ret;
}

static hi_s32 sample_vpss_grp_attach_mosaic_region(hi_mpp_chn *mpp_chn)
{
    hi_u32 mosaic_region_num = get_mosaic_region_num();
    if (mosaic_region_num == 0) {
        return HI_SUCCESS;
    }
    return sample_comm_region_attach_to_chn(mosaic_region_num, HI_RGN_MOSAIC, mpp_chn);
}

static hi_void sample_config_vpss_crop_info(hi_vpss_crop_info* crop_info)
{
    crop_info->enable = gCropEnable;
    crop_info->crop_mode = (hi_coord)gCropMode;
    crop_info->crop_rect.x = HI_ALIGN_UP(gCropX, 2); /* 4,2:align */
    crop_info->crop_rect.y = HI_ALIGN_UP(gCropY, 2); /* 4,2:align */
    crop_info->crop_rect.width = HI_ALIGN_UP(gCropWidth, 2); /* 2,2:align */
    crop_info->crop_rect.height = HI_ALIGN_UP(gCropHeight, 2); /* 2,2:align */
    sample_print("vpss_crop enable: %u, mode: %u, x:%u, y:%u, width:%u, height:%u\n",
        crop_info->enable, crop_info->crop_mode, crop_info->crop_rect.x, crop_info->crop_rect.y,
        crop_info->crop_rect.width, crop_info->crop_rect.height);
}

static hi_void sample_stop_vpss(hi_u32 vpss_start_grp_num, hi_vpss_grp vpss_grp, hi_bool *vpss_chn_enable)
{
    hi_s32 i;
    for (i = vpss_grp; i >= vpss_start_grp_num; i--) {
        vpss_grp = i;
        sample_common_vpss_stop(vpss_grp, &vpss_chn_enable[0]);
        if (i == 0) {
            break;
        }
    }
}

static hi_s32 sample_vdec_bind_vpss(hi_u32 vpss_start_grp_num, hi_u32 vpss_grp_num, hi_bool is_pip)
{
    hi_u32 i;
    hi_s32 ret;
    for (i = 0; i < vpss_grp_num; i++) {
        ret = sample_comm_vdec_bind_vpss((is_pip ? 0 : i), (i + vpss_start_grp_num));
        if (ret != HI_SUCCESS) {
            sample_print("vdec bind vpss fail for %#x!\n", ret);
            return ret;
        }
    }
    return HI_SUCCESS;
}

static hi_s32 sample_vdec_unbind_vpss(hi_u32 vpss_start_grp_num, hi_u32 vpss_grp_num, hi_bool is_pip)
{
    hi_u32 i;
    hi_s32 ret;
    for (i = 0; i < vpss_grp_num; i++) {
        ret = sample_comm_vdec_un_bind_vpss((is_pip ? 0 : i), (i + vpss_start_grp_num));
        if (ret != HI_SUCCESS) {
            sample_print("vdec unbind vpss fail for %#x!\n", ret);
            return ret;
        }
    }
    return HI_SUCCESS;
}

static hi_void sample_config_vpss_grp_attr(hi_vpss_grp_attr *vpss_grp_attr)
{
    vpss_grp_attr->max_width = gStreamHeight;
    vpss_grp_attr->max_height = gStreamHeight;
    vpss_grp_attr->frame_rate.src_frame_rate = -1;
    vpss_grp_attr->frame_rate.dst_frame_rate = -1;
    vpss_grp_attr->pixel_format  = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    vpss_grp_attr->nr_en   = HI_FALSE;
}

static hi_void sample_config_vpss_chn_attr(hi_bool *vpss_chn_enable, 
                                           hi_vpss_chn_attr vpss_chn_attr[VPSS_MAX_PHYS_CHN_NUM],
                                           hi_vpss_chn_mode chn_mode)
{
    vpss_chn_enable[0] = gVpssChn0Enable;
    vpss_chn_enable[1] = gVpssChn1Enable;
    for (hi_u32 i = 0; i < VPSS_MAX_PHYS_CHN_NUM; i++) {
        vpss_chn_attr[i].width                 = gOutWidth;
        vpss_chn_attr[i].height                = gOutHeight;
        vpss_chn_attr[i].chn_mode              = chn_mode;
        vpss_chn_attr[i].compress_mode         = HI_COMPRESS_MODE_NONE;
        vpss_chn_attr[i].pixel_format          = (hi_pixel_format)gOutFormat;
        vpss_chn_attr[i].frame_rate.src_frame_rate = -1;
        vpss_chn_attr[i].frame_rate.dst_frame_rate = -1;
        vpss_chn_attr[i].depth                     = 8; /* 8: vpss chn depth */
        vpss_chn_attr[i].aspect_ratio.mode                 = (hi_aspect_ratio_type)gAspectRatioMode;
        vpss_chn_attr[i].aspect_ratio.video_rect.width     = gAspectRatioWidth;
        vpss_chn_attr[i].aspect_ratio.video_rect.height    = gAspectRatioHeight;
        vpss_chn_attr[i].aspect_ratio.video_rect.x         = gAspectRatioX;
        vpss_chn_attr[i].aspect_ratio.video_rect.y         = gAspectRatioY;
        vpss_chn_attr[i].aspect_ratio.bg_color             = gBgColor;
        sample_print("vpss_chn_attr[%u] aspect_ratio mode: %u, x:%u, y:%u, width:%u, height:%u, color:%#x\n",
            i, gAspectRatioMode, gAspectRatioX, gAspectRatioY, gAspectRatioWidth, gAspectRatioHeight, gBgColor);
    }
}

static hi_s32 sample_start_vpss(hi_vpss_grp *vpss_grp, hi_u32 vpss_start_grp_num,
                                hi_u32 vpss_grp_num, hi_bool *vpss_chn_enable, hi_vpss_chn_mode chn_mode)
{
    hi_u32 i;
    hi_s32 ret;
    hi_vpss_chn_attr vpss_chn_attr[VPSS_MAX_PHYS_CHN_NUM];
    hi_vpss_grp_attr vpss_grp_attr = {0};
    hi_vpss_crop_info crop_info;
    hi_mpp_chn mpp_chn;

    (hi_void)memset(&vpss_chn_attr[0], 0, VPSS_MAX_PHYS_CHN_NUM * sizeof(hi_vpss_chn_attr));
    (hi_void)memset(&mpp_chn, 0, sizeof(hi_mpp_chn));
    (hi_void)memset(vpss_chn_enable, 0, VPSS_MAX_PHYS_CHN_NUM * sizeof(hi_bool));

    sample_config_vpss_grp_attr(&vpss_grp_attr);
    sample_config_vpss_crop_info(&crop_info);
    sample_config_vpss_chn_attr(vpss_chn_enable, vpss_chn_attr, chn_mode);

    for (i = vpss_start_grp_num; i < vpss_start_grp_num + vpss_grp_num; i++) {
        *vpss_grp = i;
        ret = sample_common_vpss_start(*vpss_grp, &vpss_chn_enable[0], &vpss_grp_attr, &crop_info, vpss_chn_attr);
        if (ret != HI_SUCCESS) {
            sample_print("start VPSS fail for %#x!\n", ret);
            sample_stop_vpss(vpss_start_grp_num, *vpss_grp, &vpss_chn_enable[0]);
            return ret;
        }
        /************************************************
        VPSS grp attach region
        *************************************************/
        if (gRegionEnable) {
            mpp_chn.mod_id = HI_ID_VPSS;
            mpp_chn.dev_id = *vpss_grp;
            mpp_chn.chn_id = 0;
            ret = sample_vpss_grp_attach_mosaic_region(&mpp_chn);
            if (ret != HI_SUCCESS) {
                sample_print("start attach region fail for %#x!\n", ret);
                sample_stop_vpss(vpss_start_grp_num, *vpss_grp, &vpss_chn_enable[0]);
                return ret;
            }
        }
    }

    ret = sample_vdec_bind_vpss(vpss_start_grp_num, vpss_grp_num, HI_FALSE);
    if (ret != HI_SUCCESS) {
        sample_vdec_unbind_vpss(vpss_start_grp_num, vpss_grp_num, HI_FALSE);
        sample_stop_vpss(vpss_start_grp_num, *vpss_grp, &vpss_chn_enable[0]);
    }

    return ret;
}

static hi_s32 vpss_yuv_8bit_dump(hi_video_frame* pVBuf, FILE* pfd)
{
    hi_s32 s32Ret = HI_SUCCESS;
    char* pVBufVirt_Y;
    hi_u64 phy_addr;
    hi_pixel_format  enPixelFormat = pVBuf->pixel_format;
    hi_u32 u32Size;
    hi_char* pUserPageAddr[2];

    if (pVBuf->width_stride[0] == 0 || pVBuf->height_stride[0] == 0) {
        pVBuf->width_stride[0] = pVBuf->width;
        pVBuf->height_stride[0] = pVBuf->height;
    }

    if ((HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420 == enPixelFormat) ||
        (HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420 == enPixelFormat)) {
        u32Size = (pVBuf->width_stride[0]) * (pVBuf->height) * 3 / 2;  /* 3: yuv  2: 420 */
    } else if ((HI_PIXEL_FORMAT_YVU_SEMIPLANAR_422 == enPixelFormat) ||
               (HI_PIXEL_FORMAT_YUV_SEMIPLANAR_422 == enPixelFormat)) {
        u32Size = pVBuf->width_stride[0] * (pVBuf->height) * 2; /* 2: yuv 422 */
    } else if (HI_PIXEL_FORMAT_YUV_400 == enPixelFormat) {
        u32Size = pVBuf->width_stride[0] * (pVBuf->height);
    }
    sample_print("enPixelFormat = %d, width = %d, height = %d, u32Size = %u\n",
        enPixelFormat, pVBuf->width, pVBuf->height, u32Size);

    phy_addr = pVBuf->phys_addr[0];

    pUserPageAddr[0] = (hi_char*) phy_addr;

    pVBufVirt_Y = pUserPageAddr[0];
    fwrite(pVBufVirt_Y, u32Size, 1, pfd);
    fflush(pfd);
    fflush(stderr);

    pUserPageAddr[0] = HI_NULL;
    return s32Ret;
}

static hi_void write_frame_to_file(uint32_t grpid, uint32_t chn, hi_video_frame_info *frame, uint32_t frameNum)
{
    if (gWriteFile == HI_FALSE) {
        return;
    }
    constexpr uint32_t buf_len = 60;
    char buff[buf_len] = {0};
    snprintf(buff, (buf_len - 1), "output_grp%u_chn%u_w%u_h%u_frame%u.yuv",
        grpid, chn, frame->v_frame.width, frame->v_frame.height, frameNum);
    FILE* fp = fopen(buff, "wb");
    if(fp != HI_NULL) {
        vpss_yuv_8bit_dump(&frame->v_frame, fp);
        fclose(fp);
    }
}

static hi_void get_and_save_frame(uint32_t grpid, uint32_t chn, hi_video_frame_info *frame, uint32_t frameNum)
{
    constexpr int32_t time_out = 2000;
    int32_t ret = HI_SUCCESS;
    ret = hi_mpi_vpss_get_chn_frame(grpid, chn, frame, time_out);
    if (ret == HI_SUCCESS) {
        write_frame_to_file(grpid, chn, frame, frameNum);
        // Release Frame
        ret = hi_mpi_vpss_release_chn_frame(grpid, chn, frame);
        if (ret != HI_SUCCESS) {
            sample_print("[%s][%d] Chn %u hi_mpi_vdec_release_frame Fail, Error Code = %x \n",
                __FUNCTION__, __LINE__, grpid, ret);
        }
    } else {
        sample_print("[%s][%d] grp %u hi_mpi_vpss_get_chn_frame finish!\n", __FUNCTION__, __LINE__, grpid);
    }
}

static hi_void* sample_vpss_get_frame_thread(hi_void *args)
{
    uint32_t grpid = *(uint32_t*)args;
    hi_video_frame_info frame;
    uint32_t frame_num[2] = {0};
    uint32_t grp_recv_frame_num = 0;
    hi_u32 vdec_recv_frame_num = 0;

    aclError acl_ret = aclrtSetCurrentContext(g_context);
    if (acl_ret != ACL_SUCCESS) {
        SAMPLE_PRT("set current context failed");
        return (void*)-1;
    }

    while (1) {
        if ((gVpssChn0Enable == HI_FALSE) && (gVpssChn1Enable == HI_FALSE)) {
            break;
        }
        
        if (g_vpss_get_exit_state[grpid] == 1) {
            vdec_recv_frame_num = sample_comm_vdec_get_recv_frames_cnt(grpid - gGroupId);
            grp_recv_frame_num = (gVpssChn0Enable == HI_TRUE) ? frame_num[0] : frame_num[1];
            if (grp_recv_frame_num >= vdec_recv_frame_num) {
                break;
            }
        }
        for (uint32_t chn = 0; chn < VPSS_MAX_PHYS_CHN_NUM; ++chn) {
            if ((gVpssChn0Enable == HI_FALSE && chn == 0) || (gVpssChn1Enable == HI_FALSE && chn == 1)) {
                continue;
            }
            ++frame_num[chn];
            get_and_save_frame(grpid, chn, &frame, frame_num[chn]);
        }
    }
    sample_print("[%s][%d] grp %u vpss_get_frame Thread Exit \n", __FUNCTION__, __LINE__, grpid);
    return (void*)HI_SUCCESS;
}

static void sample_vpss_stop_get_frame_thread(hi_u32 grp_num)
{
    for (uint32_t i = gGroupId; i < gGroupId + grp_num; i++) {
        // Set thread state to 1, then get frame thread will end
        g_vpss_get_exit_state[i] = 1;
    }

     for (uint32_t i = 0; i < grp_num; i++) {
        if (g_vpss_get_thread[i] != 0) {
            pthread_join(g_vpss_get_thread[i], HI_NULL);
            g_vpss_get_thread[i] = 0;
        }
    }
}

static hi_void sample_vpss_create_get_frame_thread(hi_u32 vpss_start_grp_num,hi_u32 grp_num)
{
    int32_t ret = 0;    
    for (int32_t i = 0; i < grp_num; i++) {
        g_vpss_get_thread[i] = 0;
        g_temp_vpss_grp_num[vpss_start_grp_num + i] = vpss_start_grp_num + i;
        ret = pthread_create(&g_vpss_get_thread[i], 0, sample_vpss_get_frame_thread,
            (hi_void *)&g_temp_vpss_grp_num[vpss_start_grp_num+i]);
        if (ret != 0) {
            sample_print("grp %d vpss_create_get_frame_thread fail for %#x!\n",
                g_temp_vpss_grp_num[vpss_start_grp_num + i] , ret); 
        }
        sample_print("grp %d vpss_create_get_frame_thread succeed!\n",
            g_temp_vpss_grp_num[vpss_start_grp_num + i]); 
    }
}

static hi_void sample_vdec_cmd_ctrl(hi_u32 chn_num, vdec_thread_param *vdec_send, pthread_t *vdec_thread,
    hi_u32 send_arr_len, hi_u32 thread_arr_len)
{
    sample_comm_vdec_cmd_not_circle_send(chn_num, vdec_send, vdec_thread, send_arr_len, thread_arr_len);
    return;
}

static hi_void sample_send_stream_to_vdec(sample_vdec_attr *sample_vdec, hi_u32 arr_len, hi_u32 vdec_chn_num,
    const char *stream_name)
{
    hi_u32 i;
    vdec_thread_param vdec_send[HI_VDEC_MAX_CHN_NUM];
    pthread_t vdec_thread[HI_VDEC_MAX_CHN_NUM] = {0}; /* 2:thread */
    if (arr_len > HI_VDEC_MAX_CHN_NUM) {
        sample_print("array size(%u) of vdec_send need < %u!\n", arr_len, HI_VDEC_MAX_CHN_NUM);
        return;
    }
    for (i = 0; (i < vdec_chn_num) && (i < arr_len); i++) {
        if (snprintf(vdec_send[i].c_file_name, sizeof(vdec_send[i].c_file_name), stream_name) < 0) {
            return;
        }
        if (snprintf(vdec_send[i].c_file_path, sizeof(vdec_send[i].c_file_path), "%s", SAMPLE_STREAM_PATH) < 0) {
            return;
        }
        vdec_send[i].type          = sample_vdec[i].type;
        vdec_send[i].stream_mode   = sample_vdec[i].mode;
        vdec_send[i].chn_id        = i;
        vdec_send[i].interval_time = 1000; /* 1000: interval time */
        vdec_send[i].pts_init      = 0;
        vdec_send[i].pts_increase  = 16666; /* 16666: pts increase */
        vdec_send[i].e_thread_ctrl = THREAD_CTRL_START;
        vdec_send[i].circle_send   = HI_FALSE;
        vdec_send[i].milli_sec     = 0;
        vdec_send[i].min_buf_size  = (sample_vdec[i].width * sample_vdec[i].height * 3) >> 1; /* 3:yuv */
    }

    sample_comm_vdec_start_send_stream(vdec_chn_num, &vdec_send[0], &vdec_thread[0],
        HI_VDEC_MAX_CHN_NUM, HI_VDEC_MAX_CHN_NUM);

    sample_vdec_cmd_ctrl(vdec_chn_num, &vdec_send[0], &vdec_thread[0],
        HI_VDEC_MAX_CHN_NUM, HI_VDEC_MAX_CHN_NUM);

    sample_comm_vdec_stop_send_stream(vdec_chn_num, &vdec_send[0], &vdec_thread[0],
        HI_VDEC_MAX_CHN_NUM, HI_VDEC_MAX_CHN_NUM);
}

static hi_s32 sample_get_vo_mode(hi_u32 vpss_grp_num)
{
    for(int i = 0; i< VO_MODE_BUTT; i++) {
        if(g_vo_grp_vo_mode_info[i].grp_num == vpss_grp_num) {
            return g_vo_grp_vo_mode_info[i].mode;
        }
    }
    return 0;
}

static hi_s32 sample_vpss_bind_vo(sample_vo_cfg vo_config, hi_u32 vpss_grp_num, hi_bool is_pip)
{
    hi_u32 i;
    hi_vo_layer vo_layer;
    hi_s32 ret;
    vo_layer = vo_config.vo_dev;
    for (i = 0; i < vpss_grp_num; i++) {
        if (is_pip) {
            vo_layer = (i == 0) ? 0 : 2; /* 2:layer id */
        }
        ret = sample_comm_vpss_bind_vo(i, 0, vo_layer, is_pip ? 0 : i);
        if (ret != HI_SUCCESS) {
            sample_print("vpss bind vo fail for %#x!\n", ret);
            return ret;
        }
    }
    return HI_SUCCESS;
}

static hi_s32 sample_vpss_unbind_vo(hi_u32 vpss_grp_num, sample_vo_cfg vo_config)
{
    hi_u32 i;
    hi_vo_layer vo_layer = vo_config.vo_dev;
    hi_s32 ret;
    for (i = 0; i < vpss_grp_num; i++) {
        ret = sample_comm_vpss_un_bind_vo(i, 0, vo_layer, i);
        if (ret != HI_SUCCESS) {
            sample_print("vpss unbind vo fail for %#x!\n", ret);
            return ret;
        }
    }
    return HI_SUCCESS;
}

static hi_s32 sample_start_vo(sample_vo_cfg *vo_config, hi_u32 vpss_grp_num, hi_bool is_pip)
{
    hi_s32 ret;
    vo_config->vo_dev            = SAMPLE_VO_DEV_UHD;
    vo_config->vo_layer          = 0;
    vo_config->vo_intf_type      = g_vdec_display_cfg.intf_type;
    vo_config->intf_sync         = g_vdec_display_cfg.intf_sync;
    vo_config->pic_size          = g_vdec_display_cfg.pic_size;
    vo_config->bg_color          = COLOR_RGB_BLUE;
    vo_config->dis_buf_len       = 4; /* 4:buf length */
    vo_config->dst_dynamic_range = HI_DYNAMIC_RANGE_SDR8;
    vo_config->vo_mode           = (sample_vo_mode)sample_get_vo_mode(vpss_grp_num);
    vo_config->pix_format        = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    vo_config->disp_rect.x       = 0;
    vo_config->disp_rect.y       = 0;
    vo_config->disp_rect.width   = g_disp_size.width;
    vo_config->disp_rect.height  = g_disp_size.height;
    vo_config->image_size.width  = g_disp_size.width;
    vo_config->image_size.height = g_disp_size.height;
    vo_config->vo_part_mode      = HI_VO_PARTITION_MODE_MULTI;
    vo_config->compress_mode     = HI_COMPRESS_MODE_NONE;
    vo_config->hdmi_id           = (hi_hdmi_id)0;
    vo_config->chn_param.aspect_ratio.mode                 = (hi_aspect_ratio_type)gAspectRatioMode;
    vo_config->chn_param.aspect_ratio.video_rect.width     = gAspectRatioWidth;
    vo_config->chn_param.aspect_ratio.video_rect.height    = gAspectRatioHeight;
    vo_config->chn_param.aspect_ratio.video_rect.x         = gAspectRatioX;
    vo_config->chn_param.aspect_ratio.video_rect.y         = gAspectRatioY;
    vo_config->chn_param.aspect_ratio.bg_color             = gBgColor;

    ret = sample_comm_vo_start_vo(vo_config);
    if (ret != HI_SUCCESS) {
        sample_print("start VO fail for %#x!\n", ret);
        sample_comm_vo_stop_vo(vo_config);
        return ret;
    }

    ret = sample_vpss_bind_vo(*vo_config, vpss_grp_num, is_pip);
    if (ret != HI_SUCCESS) {
        sample_vpss_unbind_vo(vpss_grp_num, *vo_config);
        sample_comm_vo_stop_vo(vo_config);
    }

    return ret;
}

static hi_s32 sample_h264_user_vdec_vpss_vo(hi_void)
{
    hi_s32 ret;
    hi_u32 chn_num;
    hi_vpss_grp vpss_grp;
    hi_vpss_grp vpss_grp_ai;
    sample_vdec_attr sample_vdec[HI_VDEC_MAX_CHN_NUM];
    sample_vo_cfg vo_config;
    hi_bool vpss_chn_enable[VPSS_MAX_PHYS_CHN_NUM];

    chn_num = gUserNum;

    /************************************************
    step1:  init SYS, init common VB(for VPSS and VO), init module VB(for VDEC)
    *************************************************/
    ret = sample_init_sys_and_vb(&sample_vdec[0], chn_num, HI_PT_H264, HI_VDEC_MAX_CHN_NUM);
    if (ret != HI_SUCCESS) {
        return ret;
    }

    /************************************************
    step2:  init region
    *************************************************/
    ret = sample_init_region();
    if (ret != HI_SUCCESS) {
        goto stop_sys;
    }

    /************************************************
    step3:  init VDEC
    *************************************************/
    ret = sample_start_vdec(&sample_vdec[0], chn_num, HI_VDEC_MAX_CHN_NUM);
    if (ret != HI_SUCCESS) {
        goto stop_sys;
    }

    /************************************************
    step4:  start VPSS
    *************************************************/
    ret = sample_start_vpss(&vpss_grp_ai, gGroupId, chn_num,
                            &vpss_chn_enable[0], HI_VPSS_CHN_MODE_USER);
    if (ret != HI_SUCCESS) {
        goto stop_vdec;
    }
    sample_vpss_create_get_frame_thread(gGroupId, chn_num);

    /************************************************
    step5:  send stream to VDEC
    *************************************************/
    sleep(1);
    sample_send_stream_to_vdec(&sample_vdec[0], HI_VDEC_MAX_CHN_NUM, chn_num, gInputFileName);

    sample_vpss_stop_get_frame_thread(chn_num);

stop_vpss:
    ret = sample_vdec_unbind_vpss(gGroupId, chn_num, HI_FALSE); 
    sample_stop_vpss(gGroupId, vpss_grp_ai, &vpss_chn_enable[0]);

stop_vdec:
    sample_comm_vdec_stop(chn_num);
stop_sys:
    sample_comm_sys_exit();
    return ret;
}

static hi_s32 sample_h264_auto_vdec_vpss_vo(hi_void)
{
    hi_s32 ret;
    hi_u32 chn_num;
    hi_vpss_grp vpss_grp;
    sample_vdec_attr sample_vdec[HI_VDEC_MAX_CHN_NUM];
    sample_vo_cfg vo_config;
    hi_bool vpss_chn_enable[VPSS_MAX_PHYS_CHN_NUM];

    chn_num = gAutoNum; 

    /************************************************
    step1:  init SYS, init common VB(for VPSS and VO), init module VB(for VDEC)
    *************************************************/
    ret = sample_init_sys_and_vb(&sample_vdec[0], chn_num, HI_PT_H264, HI_VDEC_MAX_CHN_NUM);
    if (ret != HI_SUCCESS) {
        return ret;
    }

    /************************************************
    step2:  init region
    *************************************************/
    ret = sample_init_region();
    if (ret != HI_SUCCESS) {
        goto stop_sys;
    }

    /************************************************
    step3:  init VDEC
    *************************************************/
    ret = sample_start_vdec(&sample_vdec[0], chn_num, HI_VDEC_MAX_CHN_NUM);
    if (ret != HI_SUCCESS) {
        goto stop_sys;
    }

    /************************************************
    step4:  start VPSS
    *************************************************/
    ret = sample_start_vpss(&vpss_grp, 0, chn_num, &vpss_chn_enable[0], HI_VPSS_CHN_MODE_AUTO);
    if (ret != HI_SUCCESS) {
        goto stop_vdec;
    }
    /************************************************
    step5:  start VO
    *************************************************/
    ret = sample_start_vo(&vo_config, chn_num, HI_FALSE);
    if (ret != HI_SUCCESS) {
        goto stop_vpss;
    }

    /************************************************
    step6:  send stream to VDEC
    *************************************************/
    sleep(1);
    sample_send_stream_to_vdec(&sample_vdec[0], HI_VDEC_MAX_CHN_NUM, chn_num, gInputFileName);

stop_vpss:
    ret = sample_vdec_unbind_vpss(0, chn_num, HI_FALSE);
    sample_stop_vpss(0, vpss_grp, &vpss_chn_enable[0]);
    ret = sample_vpss_unbind_vo(chn_num, vo_config);
    sample_comm_vo_stop_vo(&vo_config);

stop_vdec:
    sample_comm_vdec_stop(chn_num);
stop_sys:
    sample_comm_sys_exit();
    return ret;
}

constexpr int opt_img_width = 301;
constexpr int opt_img_height = 302;
constexpr int opt_auto_num = 303;
constexpr int opt_out_width = 304;
constexpr int opt_out_height = 305;
constexpr int opt_in_image_file = 306;
constexpr int opt_send_usleep = 307;
constexpr int opt_rgn_param = 308;
constexpr int opt_write_file = 309;
constexpr int opt_test_type = 310;
constexpr int opt_user_num = 311;
constexpr int opt_ref_num = 312;
constexpr int opt_display_num = 313;
constexpr int opt_crop_width = 314;
constexpr int opt_crop_height = 315;
constexpr int opt_crop_x = 316;
constexpr int opt_crop_y = 317;
constexpr int opt_crop_mode  = 318;
constexpr int opt_crop_enable  = 319;
constexpr int opt_aspect_ratio_width  = 320;
constexpr int opt_aspect_ratio_height  = 321;
constexpr int opt_aspect_ratio_x  = 322;
constexpr int opt_aspect_ratio_y = 323;
constexpr int opt_aspect_ratio_mode  = 324;
constexpr int opt_bg_color  = 325;
constexpr int opt_vpss_chn0_enable = 326;
constexpr int opt_vpss_chn1_enable  = 327;
constexpr int opt_user_group_id  = 328;
constexpr int opt_out_format  = 329;

static struct option long_options[] =
{
    {"img_width"              , 1, nullptr, opt_img_width},
    {"img_height"             , 1, nullptr, opt_img_height},
    {"auto_num"               , 1, nullptr, opt_auto_num},            // auto mode channel num: vdec + vpss + vo
    {"out_width"              , 1, nullptr, opt_out_width},
    {"out_height"             , 1, nullptr, opt_out_height},
    {"in_image_file"          , 1, nullptr, opt_in_image_file},
    {"send_usleep"            , 1, nullptr, opt_send_usleep},
    {"rgn_param"              , 1, nullptr, opt_rgn_param},
    {"write_file"             , 1, nullptr, opt_write_file},
    {"test_type"              , 1, nullptr, opt_test_type},
    {"user_num"               , 1, nullptr, opt_user_num},            // user mode channel num: vdec + vpss
    {"ref_num"                , 1, nullptr, opt_ref_num},
    {"display_num"            , 1, nullptr, opt_display_num},
    {"crop_width"             , 1, nullptr, opt_crop_width},          
    {"crop_height"            , 1, nullptr, opt_crop_height},         
    {"crop_x"                 , 1, nullptr, opt_crop_x},              
    {"crop_y"                 , 1, nullptr, opt_crop_y},              
    {"crop_mode"              , 1, nullptr, opt_crop_mode},           //  0:Absolute coordinates, 1:Relative coordinates
    {"crop_enable"            , 1, nullptr, opt_crop_enable},         
    {"aspect_ratio_width"     , 1, nullptr, opt_aspect_ratio_width}, 
    {"aspect_ratio_height"    , 1, nullptr, opt_aspect_ratio_height},
    {"aspect_ratio_x"         , 1, nullptr, opt_aspect_ratio_x},     
    {"aspect_ratio_y"         , 1, nullptr, opt_aspect_ratio_y},     
    {"aspect_ratio_mode"      , 1, nullptr, opt_aspect_ratio_mode},   // Amplitude ratio type 0:none, 1:automatic, 2:manual
    {"bg_color"               , 1, nullptr, opt_bg_color},            // Background
    {"vpss_chn0_enable"       , 1, nullptr, opt_vpss_chn0_enable},
    {"vpss_chn1_enable"       , 1, nullptr, opt_vpss_chn1_enable},
    {"user_group_id"          , 1, nullptr, opt_user_group_id},
    {"out_format"             , 1, nullptr, opt_out_format},
};

static bool is_argv_in_options(std::string &argv_str)
{
    for (int option_index = 0; option_index < sizeof(long_options) / sizeof(struct option); option_index++) {
        std::string option_str = std::string("--") + std::string(long_options[option_index].name);
        if (argv_str == option_str) {
            return true;
        }
    }
    return false;
}

static hi_s32 check_argv(int argc, char *argv[])
{
    for (int argc_index = 0; argc_index < argc; argc_index++) {
        std::string argv_str(argv[argc_index]);
        if (argv_str.find(std::string("--")) != std::string::npos) {
            bool find_flag = is_argv_in_options(argv_str);
            if (find_flag == false) {
                printf("argv:%s not support.\n", argv_str.c_str());
                return -1;
            }
        }
    }
    return 0;
}

static hi_s32 get_option(int argc, char *argv[])
{
    if (check_argv(argc, argv) != 0) {
        return -1;
    }

    while (1)
    {
        hi_s32 option_index = 0;
        
        hi_s32 option_value = -1;
        option_value = getopt_long(argc, argv, "", long_options, &option_index);
        if (option_value == -1) {
            break;
        }
        switch (option_value)
        {
            case opt_img_width:
                gStreamWidth = atoi(optarg);
                break;
            case opt_img_height:
                gStreamHeight = atoi(optarg);
                break;
            case opt_auto_num:
                gAutoNum = atoi(optarg);
                break;
            case opt_out_width:
                gOutWidth = atoi(optarg);
                break;
            case opt_out_height:
                gOutHeight = atoi(optarg);
                break;
            case opt_in_image_file:
                strcpy(gInputFileName, optarg);
                break;
            case opt_send_usleep:
                sample_comm_vdec_set_send_usleep((hi_u32)atoi(optarg));
                break;
            case opt_rgn_param:
                parse_region_param(optarg);
                gRegionEnable = HI_TRUE;
                break;
            case opt_write_file:
                gWriteFile = atoi(optarg);
                break;
            case opt_test_type:
                gTestType = atoi(optarg);
                break;
            case opt_user_num:
                gUserNum = atoi(optarg);
                break;
            case opt_ref_num:
                gRefNum = atoi(optarg);
                break;
            case opt_display_num:
                gDisplayNum = atoi(optarg);
                break;
            case opt_crop_width:
                gCropWidth = atoi(optarg);
                break;
            case opt_crop_height:
                gCropHeight = atoi(optarg);
                break;
            case opt_crop_x:
                gCropX = atoi(optarg);
                break;
            case opt_crop_y:
                gCropY = atoi(optarg);
                break;
            case opt_crop_mode:
                gCropMode = atoi(optarg);
                break;
            case opt_crop_enable:
                gCropEnable = (hi_bool)atoi(optarg);
                break;
            case opt_aspect_ratio_width:
                gAspectRatioWidth = atoi(optarg);
                break;
            case opt_aspect_ratio_height:
                gAspectRatioHeight = atoi(optarg);
                break;
            case opt_aspect_ratio_x:
                gAspectRatioX = atoi(optarg);
                break;
            case opt_aspect_ratio_y:
                gAspectRatioY = atoi(optarg);
                break;
            case opt_aspect_ratio_mode:
                gAspectRatioMode = atoi(optarg);
                break;
            case opt_bg_color:
                gBgColor = strtol(optarg, HI_NULL, 16); /* 16: hexadecimal */
                break;
            case opt_vpss_chn0_enable:
                gVpssChn0Enable = (hi_bool)atoi(optarg);
                break;
            case opt_vpss_chn1_enable:
                gVpssChn1Enable = (hi_bool)atoi(optarg);
                break;
            case opt_user_group_id:
                gGroupId = atoi(optarg);
                break;
            case opt_out_format:
                gOutFormat = atoi(optarg);
                break;
            default:
                break;
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    if (get_option(argc, argv) != 0) {
        return 0;
    }

    if (gTestType == 0) {
        sample_h264_user_vdec_vpss_vo();
    } else if (gTestType == 1) {
        sample_h264_auto_vdec_vpss_vo();
    } else {
        printf("gTestType should be 0: user type, 1: auto type\n");
    }

    return 0;
}