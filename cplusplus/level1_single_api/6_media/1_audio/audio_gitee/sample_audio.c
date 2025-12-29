/**
* @File sample_audio.c
* @Description audio sample app
*
* Copyright (c) Huawei Technologies Co., Ltd. 2016-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include "sample_comm_audio.h"

#define sample_dbg(ret) \
    do { \
        printf("ret = %#x, fuc:%s, line:%d\n", ret, __FUNCTION__, __LINE__); \
    } while (0)

hi_s32 g_error_num = 0;
hi_s32 g_start = 0;
int g_point_num_per_frame = 960;
int g_bit_width = HI_AUDIO_BIT_WIDTH_16;
int g_snd_mode = HI_AUDIO_SOUND_MODE_MONO;
int g_dev_num = 2;
int g_chn_num = 0;

/* vb exit & MPI system exit */
hi_void sample_comm_sys_exit(hi_void)
{
    hi_mpi_sys_exit();
    return;
}

/* vb init & MPI system init */
hi_s32 sample_comm_sys_init(void)
{
    hi_s32 ret;
    ret = hi_mpi_sys_init();
    if (ret != HI_SUCCESS) {
        printf("hi_mpi_sys_init failed!\n");
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

static hi_void sample_audio_aio_init_param(hi_aio_attr *aio_attr)
{
    aio_attr->chn_cnt             = 1;
    aio_attr->point_num_per_frame = 960;
    aio_attr->expand_flag         = 0;
    aio_attr->frame_num           = 30;
    aio_attr->clk_share           = 1;
    aio_attr->snd_mode            = HI_AUDIO_SOUND_MODE_MONO;
    aio_attr->sample_rate         = HI_AUDIO_SAMPLE_RATE_48000;
    aio_attr->bit_width           = HI_AUDIO_BIT_WIDTH_16;
    aio_attr->work_mode           = HI_AIO_MODE_I2S_MASTER;
    aio_attr->i2s_type            = HI_AIO_I2STYPE_INNERCODEC;
}

hi_void handle(int signum)
{
    g_start = 0;
    return;
}

/* start ao */
hi_s32 sample_comm_audio_start_ao(hi_audio_dev ao_dev_id, hi_aio_attr *aio_attr)
{
    hi_s32 ret;

    ret = hi_mpi_ao_set_pub_attr(ao_dev_id, aio_attr);
    if (ret != HI_SUCCESS) {
        g_error_num = 1;
        printf("%s: hi_mpi_ao_set_pub_attr(%d) failed with %#x!\n", __FUNCTION__, ao_dev_id, ret);
        return HI_FAILURE;
    }

    ret = hi_mpi_ao_enable(ao_dev_id);
    if (ret != HI_SUCCESS) {
        g_error_num = 2;
        printf("%s: hi_mpi_ao_enable(%d) failed with %#x!\n", __FUNCTION__, ao_dev_id, ret);
        return HI_FAILURE;
    }

    ret = hi_mpi_ao_enable_chn(ao_dev_id, 0);
    if (ret != HI_SUCCESS) {
        g_error_num = 3;
        printf("%s: hi_mpi_ao_enable_chn(%d) failed with %#x!\n", __FUNCTION__, 0, ret);
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

/* start ai */
hi_s32 sample_comm_audio_start_ai(hi_audio_dev ai_dev_id, hi_aio_attr *aio_attr, hi_audio_dev ao_dev_id)
{
    hi_s32 ret;

    ret = hi_mpi_ai_set_pub_attr(ai_dev_id, aio_attr);
    if (ret) {
        printf("%s: hi_mpi_ai_set_pub_attr(%d) failed with %#x\n", __FUNCTION__, ai_dev_id, ret);
        return ret;
    }

    ret = hi_mpi_ai_enable(ai_dev_id);
    if (ret) {
        printf("%s: hi_mpi_ai_enable(%d) failed with %#x\n", __FUNCTION__, ai_dev_id, ret);
        return ret;
    }

    ret = hi_mpi_ai_enable_chn(ai_dev_id, 0);
    if (ret) {
        printf("%s: hi_mpi_ai_enable_chn(%d,%d) failed with %#x\n", __FUNCTION__, ai_dev_id, 0, ret);
        return ret;
    }

    return HI_SUCCESS;
}

/* stop ao */
hi_s32 sample_comm_audio_stop_ao(hi_audio_dev ao_dev_id)
{
    hi_s32 ret;
    ret = hi_mpi_ao_disable_chn(ao_dev_id, 0);
    if (ret != HI_SUCCESS) {
        printf("%s: hi_mpi_ao_disable_chn failed with %#x!\n", __FUNCTION__, ret);
        return ret;
    }

    ret = hi_mpi_ao_disable(ao_dev_id);
    if (ret != HI_SUCCESS) {
        printf("%s: hi_mpi_ao_disable failed with %#x!\n", __FUNCTION__, ret);
        return ret;
    }

    return HI_SUCCESS;
}

/* stop ai */
hi_s32 sample_comm_audio_stop_ai(hi_audio_dev ai_dev_id)
{
    hi_s32 ret;

    ret = hi_mpi_ai_disable_chn(ai_dev_id, 0);
    if (ret != HI_SUCCESS) {
        printf("[func]:%s [line]:%d [info]:%s\n", __FUNCTION__, __LINE__, "failed");
        return ret;
    }

    ret = hi_mpi_ai_disable(ai_dev_id);
    if (ret != HI_SUCCESS) {
        printf("[func]:%s [line]:%d [info]:%s\n", __FUNCTION__, __LINE__, "failed");
        return ret;
    }

    return HI_SUCCESS;
}

/* init ai volume */
hi_s32 sample_audio_ai_vol_init()
{
    hi_s32 fd;
    hi_s32 ret;

    fd = open("/dev/acodec", O_RDWR);
    hi_acodec_volume_ctrl vol_ctrl = {0};
    vol_ctrl.volume_ctrl = 15; // 15: default volume

    ret = ioctl(fd, HI_ACODEC_SET_ADCL_VOLUME, &vol_ctrl);
    if (ret != HI_SUCCESS) {
        printf("HI_ACODEC_SET_ADCL_VOLUME failed!\n");
        return HI_FAILURE;
    }

    close(fd);
    return HI_SUCCESS;
}

hi_s32 sample_audio_ao_init()
{
    hi_s32 ret;
    hi_aio_attr aio_attr = {0};

    sample_audio_aio_init_param(&aio_attr);

    ret = sample_comm_audio_start_ao(g_dev_num, &aio_attr);
    if (ret != HI_SUCCESS) {
        printf("sample_comm_audio_start_ao err, ret = %#x\n", ret);
        sample_dbg(ret);
    }

    return ret;
}

static hi_s32 sample_audio_ai_init()
{
    hi_s32 ret;
    hi_aio_attr aio_attr = {0};

    ret = sample_audio_ai_vol_init();
    if (ret != HI_SUCCESS) {
        printf("sample_audio_ai_vol_init err, ret = %#x\n", ret);
        sample_dbg(ret);
    }

    sample_audio_aio_init_param(&aio_attr);

    ret = sample_comm_audio_start_ai(g_dev_num, &aio_attr, -1);
    if (ret != HI_SUCCESS) {
       sample_dbg(ret);
       printf("sample_audio_ai err ret = %#x\n", ret);
       return ret;
    }

    return ret;
}

static hi_s32 audio_ai_get_frame_and_send(sample_ai *ai_ctl, FILE *capture_fd)
{
    hi_s32 ret;
    hi_audio_frame frame = {0};
    hi_aec_frame aec_frm = {0};

    /* get frame from ai chn */
    ret = hi_mpi_ai_get_frame(ai_ctl->ai_dev, ai_ctl->ai_chn, &frame, &aec_frm, -1);
    if (ret != HI_SUCCESS) {
        /* continue */
        return HI_SUCCESS;
    }

    fwrite(frame.virt_addr[0], 1, frame.len, capture_fd);

    /* finally you must release the stream */
    ret = hi_mpi_ai_release_frame(ai_ctl->ai_dev, ai_ctl->ai_chn, &frame, &aec_frm);
    if (ret != HI_SUCCESS) {
        printf("%s: hi_mpi_ai_release_frame(%d, %d), failed with %#x!\n", __FUNCTION__, ai_ctl->ai_dev, ai_ctl->ai_chn,
            ret);
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

static hi_void sample_audio_play_thread(void *parg)
{
    hi_s32 ret;
    hi_audio_frame audio_frame = {0};
    hi_char play_file_name[FILE_NAME_LEN] = {0};
    FILE *play_fd = NULL;
    hi_u32 readlen = 0;
    hi_u8 * audio_buf = NULL;
    hi_ao_chn ao_chn;
    hi_u32 trans_len;

    sample_ao *ao_ctl = (sample_ao *)parg;
    hi_audio_dev ao_dev = ao_ctl->ao_dev;

    ao_chn = ao_ctl->ao_chn;
    trans_len = g_point_num_per_frame * 16 / 8;

    audio_buf = (hi_u8 *)malloc(sizeof(hi_u8) * trans_len);
    if (audio_buf == NULL) {
        printf("%s: malloc buffer failed!\n", __FUNCTION__);
    }

    /* open file */
    ret = snprintf(play_file_name, FILE_NAME_LEN, ao_ctl->file_name);
    if (ret < 0) {
        printf("[func]:%s [line]:%d [info]:%s\n", __FUNCTION__, __LINE__, "get ao file name failed");
        return NULL;
    }

    play_fd = fopen(play_file_name, "r");
    if (play_fd == NULL) {
        printf("open play pcm1 file failed path = %s\n", play_file_name);
        free(audio_buf);
        return NULL;
    }

    do {
        readlen = fread(audio_buf, 1, trans_len, play_fd);
        if (readlen == 0) {
            printf("dev-%d play thread is end \n", ao_dev);
            break;
        }

        audio_frame.virt_addr[0] = audio_buf;

        audio_frame.len = trans_len;
        audio_frame.bit_width = g_bit_width;
        audio_frame.snd_mode = g_snd_mode;

        ret = hi_mpi_ao_send_frame(ao_dev, ao_chn, &audio_frame, -1);
        if (ret != HI_SUCCESS) {
            printf("%s: hi_mpi_ao_send_frame failed with %#x!\n", __FUNCTION__, ret);
            break;
        }

    } while (readlen > 0);

    free(audio_buf);
    audio_buf = NULL;
    fclose(play_fd);
    ao_ctl->start = HI_FALSE;
    pthread_kill(&ao_ctl->ao_pid, 0);
}

static hi_void sample_audio_capture_thread(void *parg)
{
    hi_s32 ret;
    sample_ai *ai_ctl = (sample_ai *)parg;
    hi_char capture_file_name[FILE_NAME_LEN] = {0};
    FILE *cap_fd;

    ret = snprintf(capture_file_name, FILE_NAME_LEN, ai_ctl->file_name);
    if (ret < 0) {
        printf("[func]:%s [line]:%d [info]:%s\n", __FUNCTION__, __LINE__, "get ai file name failed");
        return NULL;
    }

    cap_fd = fopen(capture_file_name, "w+");
    if (cap_fd == NULL) {
        printf("capture save file open failed!\n");
        return NULL;
    }

    while (g_start == 1) {
        ret = audio_ai_get_frame_and_send(ai_ctl, cap_fd);
        if (ret != HI_SUCCESS) {
            break;
        }
    }

    fclose(cap_fd);
    cap_fd = NULL;

    ai_ctl->start = HI_FALSE;

    pthread_kill(&ai_ctl->ai_pid, 0);
    return NULL;
}

static hi_void sample_audio_play(char *file_name)
{
    hi_s32 ret;
    sample_ao ao_ctl = {0};

    /* init ao */
    ret = sample_audio_ao_init();
    if (ret != HI_SUCCESS) {
        printf("ao init failed! ret = %#x\n", ret);
        if (g_error_num == 1 || g_error_num == 2) {
            return;
        } else if (g_error_num == 3) {
            ret = hi_mpi_ao_disable(g_dev_num);
            if (ret != HI_SUCCESS) {
                printf("%s: hi_mpi_ao_disable failed with %#x!\n", __FUNCTION__, ret);
                return ret;
            }
        }
    }

    ao_ctl.ao_dev = g_dev_num;
    strcpy(ao_ctl.file_name, file_name);
    ao_ctl.ao_chn = g_chn_num;
    ao_ctl.start = HI_TRUE;

    pthread_create(&ao_ctl.ao_pid, 0, sample_audio_play_thread, &ao_ctl);

    pthread_join(ao_ctl.ao_pid, NULL);

    hi_u32 delay = 0;
    do {
        hi_mpi_ao_get_chn_delay(g_dev_num, g_chn_num, &delay);
        usleep(1000);
    } while(delay != 0);

    ret = sample_comm_audio_stop_ao(g_dev_num);
    if (ret != HI_SUCCESS) {
        sample_dbg(ret);
    }
}

static hi_void sample_audio_capture(char *file_name)
{
    hi_s32 ret;
    const hi_ai_chn ai_chn = 0;
    sample_ai ai_ctl = {0};
    hi_s32 dev_id;
    /* init ai */
    ret = sample_audio_ai_init();
    if (ret != HI_SUCCESS) {
        printf("ai init failed! ret = %#x\n", ret);
        goto ai_init_err;
    }

    ai_ctl.ai_dev = g_dev_num;
    ai_ctl.ai_chn = g_chn_num;
    strcpy(ai_ctl.file_name, file_name);
    ai_ctl.start = HI_TRUE;

    g_start = 1;

    signal(SIGUSR2, &handle);

    pthread_create(&ai_ctl.ai_pid, 0, sample_audio_capture_thread, &ai_ctl);

    pthread_join(ai_ctl.ai_pid, NULL);
ai_init_err:
    ret = sample_comm_audio_stop_ai(g_dev_num);
    if (ret != HI_SUCCESS) {
        sample_dbg(ret);
    }
}

hi_s32 main(int argc, char *argv[])
{
    hi_s32 ret;
    char file_name[FILE_NAME_LEN];

    if ((!strcmp(argv[1], "play")) || (!strcmp(argv[1], "capture"))) {
        strcpy(file_name, argv[argc -1]);
    } else {
        printf("sample audio main() test_type fail\n");
        return HI_FAILURE;
    }

    ret = sample_comm_sys_init();
    if (ret != HI_SUCCESS) {
        printf("%s: system init failed with %#x!\n", __FUNCTION__, ret);
        return HI_FAILURE;
    }

    if (!strcmp(argv[1], "play")) {
        sample_audio_play(file_name);
    } else {
        sample_audio_capture(file_name);
    }

    sample_comm_sys_exit();
    printf("main finish!\n");
    return ret;
}