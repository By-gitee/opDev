#include "sample_comm.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/prctl.h>

static hi_u32 g_send_usleep = 31000;
static hi_u32 g_vdec_recv_frame_num[HI_VPSS_MAX_GET_FRAME_CHN_NUM] = {0};

hi_void sample_comm_vdec_set_send_usleep(hi_u32 send_usleep)
{
    g_send_usleep = send_usleep;
}

hi_u32 sample_comm_vdec_get_recv_frames_cnt(hi_u32 vdec_chn)
{
    if (vdec_chn >= HI_VPSS_MAX_GET_FRAME_CHN_NUM) {
        return 0;
    }
    return g_vdec_recv_frame_num[vdec_chn];
}

hi_void sample_comm_vdec_config_attr(hi_vdec_chn_attr *chn_attr, sample_vdec_attr *sample_vdec)
{
    hi_pic_buf_attr buf_attr  = { 0 };

    chn_attr->type            = sample_vdec->type;
    chn_attr->mode            = sample_vdec->mode;
    chn_attr->pic_width       = sample_vdec->width;
    chn_attr->pic_height      = sample_vdec->height;
    chn_attr->stream_buf_size = sample_vdec->width * sample_vdec->height;
    chn_attr->frame_buf_cnt   = sample_vdec->frame_buf_cnt;

    buf_attr.align  = 0;
    buf_attr.height = sample_vdec->height;
    buf_attr.width  = sample_vdec->width;

    if (sample_vdec->type == HI_PT_H264 || sample_vdec->type == HI_PT_H265) {
        buf_attr.bit_width    = sample_vdec->sample_vdec_video.bit_width;
        buf_attr.pixel_format = HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
        chn_attr->video_attr.ref_frame_num    = sample_vdec->sample_vdec_video.ref_frame_num;
        chn_attr->video_attr.temporal_mvp_en  = HI_TRUE;
        if ((sample_vdec->type == HI_PT_H264) &&
            (sample_vdec->sample_vdec_video.dec_mode != HI_VIDEO_DEC_MODE_IPB)) {
            chn_attr->video_attr.temporal_mvp_en = HI_FALSE;
        }
        chn_attr->frame_buf_size = hi_vdec_get_pic_buf_size(chn_attr->type, &buf_attr);
        chn_attr->video_attr.tmv_buf_size = hi_vdec_get_tmv_buf_size(chn_attr->type,
            sample_vdec->height, sample_vdec->width);
    }

    return;
}

hi_s32 sample_comm_vdec_start(hi_s32 chn_num, sample_vdec_attr *sample_vdec, hi_u32 arr_len)
{
    hi_s32 i;
    hi_vdec_chn_attr chn_attr[HI_VDEC_MAX_CHN_NUM];
    hi_vdec_chn_param chn_param;

    check_null_ptr_return(sample_vdec);
    if (arr_len > HI_VDEC_MAX_CHN_NUM) {
        sample_print("array size(%u) of chn_attr need < %u!\n", arr_len, HI_VDEC_MAX_CHN_NUM);
        return HI_FAILURE;
    }

    for (i = 0; (i < chn_num) && (i < (hi_s32)arr_len); i++) {
        sample_comm_vdec_config_attr(&chn_attr[i], &sample_vdec[i]);

        check_chn_return(hi_mpi_vdec_create_chn(i, &chn_attr[i]), i, "vdec create chn");
        check_chn_return(hi_mpi_vdec_get_chn_param(i, &chn_param), i, "vdec get chn param");

        if (sample_vdec[i].type == HI_PT_H264 || sample_vdec[i].type == HI_PT_H265) {
            chn_param.video_param.dec_mode = sample_vdec[i].sample_vdec_video.dec_mode;
            sample_print("hi_mpi_vdec_get_chn_param, compress_mode is %d,video_format %d,display_frame_num is %d\n",
                chn_param.video_param.compress_mode, chn_param.video_param.video_format,chn_param.display_frame_num);
            chn_param.display_frame_num = DISPLAY_NUM;
            if (chn_param.video_param.dec_mode == HI_VIDEO_DEC_MODE_IPB) {
                chn_param.video_param.out_order = HI_VIDEO_OUT_ORDER_DISPLAY;
            } else {
                chn_param.video_param.out_order = HI_VIDEO_OUT_ORDER_DEC;
            }
        } else {
            chn_param.pic_param.pixel_format = sample_vdec[i].sample_vdec_picture.pixel_format;
            chn_param.pic_param.alpha = sample_vdec[i].sample_vdec_picture.alpha;
        }
        chn_param.display_frame_num = sample_vdec[i].display_frame_num;

        check_chn_return(hi_mpi_vdec_set_chn_param(i, &chn_param), i, "vdec set chn param");
        check_chn_return(hi_mpi_vdec_start_recv_stream(i), i, "vdec start recv stream");
    }

    return HI_SUCCESS;
}

hi_s32 sample_comm_vdec_bind_vpss(hi_vdec_chn vdec_chn, hi_vpss_grp vpss_grp)
{
    hi_mpp_chn src_chn;
    hi_mpp_chn dest_chn;

    src_chn.mod_id = HI_ID_VDEC;
    src_chn.dev_id = 0;
    src_chn.chn_id = vdec_chn;

    dest_chn.mod_id = HI_ID_VPSS;
    dest_chn.dev_id = vpss_grp;
    dest_chn.chn_id = 0;

    check_return(hi_mpi_sys_bind(&src_chn, &dest_chn), "hi_mpi_sys_bind(VDEC-VPSS)");

    return HI_SUCCESS;
}

hi_s32 sample_comm_vdec_un_bind_vpss(hi_vdec_chn vdec_chn, hi_vpss_grp vpss_grp)
{
    hi_mpp_chn src_chn;
    hi_mpp_chn dest_chn;

    src_chn.mod_id = HI_ID_VDEC;
    src_chn.dev_id = 0;
    src_chn.chn_id = vdec_chn;

    dest_chn.mod_id = HI_ID_VPSS;
    dest_chn.dev_id = vpss_grp;
    dest_chn.chn_id = 0;

    check_return(hi_mpi_sys_unbind(&src_chn, &dest_chn), "hi_mpi_sys_unbind(VDEC-VPSS)");

    return HI_SUCCESS;
}

static hi_s32 sample_comm_vdec_check_send_stream_param(vdec_thread_param *thread_param,
    hi_char *c_stream_file, hi_u32 arr_len)
{
    if (thread_param->c_file_path == HI_NULL || thread_param->c_file_name == HI_NULL) {
        sample_print("chn %d stream file path or stream file name is NULL\n", thread_param->chn_id);
        return HI_FAILURE;
    }

    if (arr_len <= 1) {
        sample_print("chn %d arr length might be overflow\n", thread_param->chn_id);
        return HI_FAILURE;
    }

    if (snprintf(c_stream_file, arr_len, "%s/%s", thread_param->c_file_path, thread_param->c_file_name) < 0) {
        sample_print("chn %d config stream file failed!\n", thread_param->chn_id);
        return HI_FAILURE;
    }

    if (thread_param->min_buf_size <= 0) {
        sample_print("chn %d min_buf_size should greater than zero!\n", thread_param->chn_id);
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

static hi_void sample_comm_vdec_send_h264_frame_process(hi_s32 *read_len, hi_u8 *buf,
    vdec_thread_param *thread_param, hi_s32 used_bytes)
{
    hi_s32 i;
    hi_bool find_start = HI_FALSE;
    hi_bool find_end = HI_FALSE;
    hi_bool new_pic;
    /* H264 frame start marker */
    if (*read_len > thread_param->min_buf_size) {
        sample_print("chn %d read_len %d is bigger than buf_size %u!\n", thread_param->chn_id, *read_len,
            thread_param->min_buf_size);
        return;
    }
    for (i = 0; i < *read_len - 8; i++) { /* 8:h264 frame start code length */
        int tmp = buf[i + 3] & 0x1F; /* 3:index  0x1F:frame start marker */
        new_pic = (hi_bool)(buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1 && /* 1 2:index */
            (((tmp == 0x5 || tmp == 0x1) && ((buf[i + 4] & 0x80) == 0x80)) || /* 4:index 0x5 0x80:frame start mark */
            (tmp == 20 && (buf[i + 7] & 0x80) == 0x80))); /* 20 0x1 0x80:frame start marker 7:index */
        if (new_pic == HI_TRUE) {
            find_start = HI_TRUE;
            i += 8; /* 8:h264 frame start code length */
            break;
        }
    }

    for (; i < *read_len - 8; i++) { /* 8:h264 frame start code length */
        int tmp = buf[i + 3] & 0x1F; /* 3:index  0x1F:frame start marker */
        new_pic = (hi_bool)(buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1 && /* 1 2:index */
            (tmp == 15 || tmp == 7 || tmp == 8 || tmp == 6 || /* 15 7 8 6:frame start marker */
            ((tmp == 5 || tmp == 1) && ((buf[i + 4] & 0x80) == 0x80)) || /* 4:index 5 0x80:frame start marker */
            (tmp == 20 && (buf[i + 7] & 0x80) == 0x80))); /* 7:index 20 0x80:frame start marker */
        if (new_pic == HI_TRUE) {
            find_end = HI_TRUE;
            break;
        }
    }

    if (i > 0) {
        *read_len = i;
    }
    if (find_start == HI_FALSE) {
        sample_print("chn %d can not find H264 start code! read_len %d, used_bytes %d!\n", thread_param->chn_id,
            *read_len, used_bytes);
    }
    if (find_end == HI_FALSE) {
        *read_len = i + 8; /* 8:h264 frame start code length */
    }
    return;
}

static hi_u32 sample_comm_vdec_handle_send_stream_mode(hi_s32 *read_len, hi_u8 *buf,
    vdec_thread_param *thread_param, hi_s32 used_bytes, hi_bool *end_of_stream)
{
    hi_u32 start = 0;
    if (thread_param->stream_mode == HI_VDEC_SEND_MODE_FRAME && thread_param->type == HI_PT_H264) {
        sample_comm_vdec_send_h264_frame_process(read_len, buf, thread_param, used_bytes);
    } else {
        if ((*read_len != 0) && (*read_len < thread_param->min_buf_size)) {
            *end_of_stream = HI_TRUE;
        }
    }
    return start;
}

static hi_void sample_comm_vdec_handle_send_stream(vdec_thread_param *thread_param, hi_vdec_stream *stream,
    hi_bool *end_of_stream, hi_u64 *pts)
{
    hi_s32 ret;

    do {
        ret = hi_mpi_vdec_send_stream(thread_param->chn_id, stream, HI_NULL, thread_param->milli_sec);
        if(ret != HI_SUCCESS) {
            sample_print("chn %d vdec_send_stream failed with ret: %#x\n", thread_param->chn_id, ret);
            break;
        } else {
            usleep(thread_param->interval_time);
        }
    } while ((ret != HI_SUCCESS) && (thread_param->e_thread_ctrl == THREAD_CTRL_START));

    *end_of_stream = HI_FALSE;
    *pts += thread_param->pts_increase;
    return;
}

static hi_void sample_comm_vdec_send_stream_process(vdec_thread_param *thread_param, FILE *fp_strm, hi_u8 *buf)
{
    hi_bool end_of_stream;
    hi_s32 used_bytes = 0;
    hi_s32 read_len;
    hi_u64 pts = thread_param->pts_init;
    hi_u32 start;
    hi_vdec_stream stream;

    while (1) {
        if (thread_param->e_thread_ctrl == THREAD_CTRL_STOP) {
            break;
        } else if (thread_param->e_thread_ctrl == THREAD_CTRL_PAUSE) {
            sleep(1);
            continue;
        }

        end_of_stream = HI_FALSE;
        fseek(fp_strm, used_bytes, SEEK_SET);
        read_len = fread(buf, 1, thread_param->min_buf_size, fp_strm);
        if (read_len == 0) {
            if (thread_param->circle_send == HI_TRUE) {
                (hi_void)memset(&stream, 0, sizeof(hi_vdec_stream));
                stream.end_of_stream = HI_TRUE;
                hi_mpi_vdec_send_stream(thread_param->chn_id, &stream, HI_NULL, -1);

                used_bytes = 0;
                fseek(fp_strm, 0, SEEK_SET);
                read_len = fread(buf, 1, thread_param->min_buf_size, fp_strm);
            } else {
                break;
            }
        }

        start = sample_comm_vdec_handle_send_stream_mode(&read_len, buf, thread_param, used_bytes, &end_of_stream);

        stream.pts = pts;
        stream.addr = buf + start;
        stream.len = read_len;
        stream.end_of_frame = (thread_param->stream_mode == HI_VDEC_SEND_MODE_FRAME) ? HI_TRUE : HI_FALSE;
        stream.end_of_stream = end_of_stream;
        stream.need_display = HI_TRUE;

        sample_comm_vdec_handle_send_stream(thread_param, &stream, &end_of_stream, &pts);
        usleep(g_send_usleep);
        used_bytes += read_len + start;
    }
    return;
}

hi_void *sample_comm_vdec_send_stream(hi_void *args)
{
    vdec_thread_param *thread_param = (vdec_thread_param *)args;
    FILE *fp_strm = HI_NULL;
    hi_u8 *buf = HI_NULL;
    hi_vdec_stream stream;
    hi_char c_stream_file[FILE_NAME_LEN] = {0};
    hi_char real_path[PATH_MAX] = {0};
    hi_char *path = HI_NULL;

    aclError acl_ret = aclrtSetCurrentContext(g_context);
    if (acl_ret != ACL_SUCCESS) {
        SAMPLE_PRT("set current context failed");
        return HI_NULL;
    }
   
    prctl(PR_SET_NAME, "video_send_stream", 0, 0, 0);

    if (sample_comm_vdec_check_send_stream_param(thread_param, c_stream_file, FILE_NAME_LEN) != HI_SUCCESS) {
        return HI_NULL;
    }

    path = realpath(c_stream_file, real_path);
    if (path == HI_NULL) {
        sample_print("chn %d Invalid stream path. Please check!\n", thread_param->chn_id);
        return HI_NULL;
    }

    fp_strm = fopen(real_path, "rb");
    if (fp_strm == HI_NULL) {
        sample_print("chn %d can't open file %s in send stream thread!\n", thread_param->chn_id, c_stream_file);
        goto end1;
    }

    buf = (hi_u8 *)malloc(thread_param->min_buf_size);
    if (buf == HI_NULL) {
        sample_print("chn %d can't alloc %d in send stream thread!\n",
            thread_param->chn_id, thread_param->min_buf_size);
        goto end;
    }
    fflush(stdout);

    sample_comm_vdec_send_stream_process(thread_param, fp_strm, buf);

    /* send the flag of stream end */
    (hi_void)memset(&stream, 0, sizeof(hi_vdec_stream));
    stream.end_of_stream = HI_TRUE;
    hi_mpi_vdec_send_stream(thread_param->chn_id, &stream, HI_NULL, -1);

    fflush(stdout);
    if (buf != HI_NULL) {
        free(buf);
        buf = HI_NULL;
    }
end:
    fclose(fp_strm);
    fp_strm = HI_NULL;
end1:
    path = HI_NULL;
    return HI_NULL;
}

static hi_void sample_comm_vdec_print_chn_status(hi_s32 chn, hi_vdec_chn_status status)
{
    printf("\033[0;33m ---------------------------------------------------------------\033[0;39m\n");
    printf("\033[0;33m chn:%d, type:%d, start:%d, decode_frames:%d, left_pics:%d, left_bytes:%d, "
        "left_frames:%d, recv_frames:%d  \033[0;39m\n",
        chn, (status).type, (status).is_started,  (status).dec_stream_frames, (status).left_decoded_frames,
        (status).left_stream_bytes, (status).left_stream_frames, (status).recv_stream_frames);
    printf("\033[0;33m format_err:%d,    pic_size_err_set:%d,  stream_unsprt:%d,  pack_err:%d, "
        "set_pic_size_err:%d,  ref_err_set:%d,  pic_buf_size_err_set:%d  \033[0;39m\n",
        (status).dec_err.format_err, (status).dec_err.set_pic_size_err, (status).dec_err.stream_unsupport,
        (status).dec_err.pack_err, (status).dec_err.set_protocol_num_err, (status).dec_err.set_ref_num_err,
        (status).dec_err.set_pic_buf_size_err);
    printf("\033[0;33m -----------------------------------------------------------------\033[0;39m\n");
    return;
}

hi_void sample_comm_vdec_cmd_not_circle_send(hi_u32 chn_num, vdec_thread_param *vdec_send,
    pthread_t *vdec_thread, hi_u32 send_arr_len, hi_u32 thread_arr_len)
{
    hi_s32 ret;
    hi_u32 i;
    hi_vdec_chn_status status;
    printf("decoding..............\n");
    for (i = 0; (i < chn_num) && (i < send_arr_len) && (i < thread_arr_len); i++) {
        if (vdec_thread[i] != 0) {
            ret = pthread_join(vdec_thread[i], HI_NULL);
            if (ret == 0) {
                vdec_thread[i] = 0;
            }
        }
        vdec_thread[i] = 0;
        while (1) {
            ret = hi_mpi_vdec_query_status(vdec_send[i].chn_id, &status);
            if (ret != HI_SUCCESS) {
                printf("chn %d hi_mpi_vdec_query_status fail!!!\n", ret);
                return;
            }
            g_vdec_recv_frame_num[vdec_send[i].chn_id] = status.recv_stream_frames;
            if ((status.left_stream_bytes == 0) && (status.left_stream_frames == 0)) {
                sample_comm_vdec_print_chn_status(vdec_send[i].chn_id, status);
                break;
            }
            usleep(1000); /* 1000:Decoding wait time */
        }
    }
    printf("end!\n");
    return;
}

hi_void sample_comm_vdec_start_send_stream(hi_s32 chn_num, vdec_thread_param *vdec_send,
    pthread_t *vdec_thread, hi_u32 send_arr_len, hi_u32 thread_arr_len)
{
    hi_u32 i;
    if ((vdec_send == HI_NULL) || (vdec_thread == HI_NULL)) {
        sample_print("vdec_send or vdec_thread can't be NULL!\n");
        return;
    }
    for (i = 0; (i < (hi_u32)chn_num) && (i < send_arr_len) && (i < thread_arr_len); i++) {
        vdec_thread[i] = 0;
        pthread_create(&vdec_thread[i], 0, sample_comm_vdec_send_stream, (hi_void *)&vdec_send[i]);
    }
}

hi_void sample_comm_vdec_stop_send_stream(hi_s32 chn_num, vdec_thread_param *vdec_send,
    pthread_t *vdec_thread, hi_u32 send_arr_len, hi_u32 thread_arr_len)
{
    hi_u32 i;
    if ((vdec_send == HI_NULL) || (vdec_thread == HI_NULL)) {
        sample_print("vdec_send or vdec_thread can't be NULL!\n");
        return;
    }
    for (i = 0; (i < (hi_u32)chn_num) && (i < send_arr_len) && (i < thread_arr_len); i++) {
        vdec_send[i].e_thread_ctrl = THREAD_CTRL_STOP;
        if (vdec_thread[i] != 0) {
            pthread_join(vdec_thread[i], HI_NULL);
            vdec_thread[i] = 0;
        }
        hi_mpi_vdec_stop_recv_stream(i);
    }
}

hi_s32 sample_comm_vdec_stop(hi_s32 chn_num)
{
    hi_s32 i;

    for (i = 0; i < chn_num; i++) {
        check_chn_return(hi_mpi_vdec_stop_recv_stream(i), i, "vdec stop recv stream");
        check_chn_return(hi_mpi_vdec_destroy_chn(i), i, "vdec destroy chn");
    }

    return HI_SUCCESS;
}