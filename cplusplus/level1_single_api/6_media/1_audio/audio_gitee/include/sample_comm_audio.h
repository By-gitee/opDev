/**
* @File sample_comm_audio.h
* @Description audio sample app
*
* Copyright (c) Huawei Technologies Co., Ltd. 2016-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "stdio.h"
#include "hi_mpi_audio.h"
#include "hi_acodec.h"

#define FILE_NAME_LEN 512

typedef struct {
    hi_bool start;
    hi_audio_dev ai_dev;
    hi_ai_chn ai_chn;
    hi_char file_name[FILE_NAME_LEN];
    pthread_t ai_pid;
} sample_ai;

typedef struct {
    hi_audio_dev ao_dev;
    hi_bool start;
    hi_char file_name[FILE_NAME_LEN];
    hi_ao_chn ao_chn;
    pthread_t ao_pid;
} sample_ao;