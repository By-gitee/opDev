/**
* @File sample_raw_back.c
* @Description sample app
*
* Copyright (c) Huawei Technologies Co., Ltd. 2016-2021. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>

#include "sample_comm.h"
#include "sample_comm_vi.h"
#include "sample_comm_isp.h"
#include "vi_with_sensor.h"

static volatile sig_atomic_t g_sig_flag = 0;

#define check_digit(x) ((x) >= '0' && (x) <= '9')

static hi_void sample_vio_usage(char *prg_name)
{
    printf("usage : %s <index> <camera num> <camera type>\n", prg_name);
    printf("index:\n");
    printf("    (1) run VI with sensor\n");
    printf("camera num:\n");
    printf("     num between [1, 2]\n");
    printf("camera type:\n");
    printf("    (1) imx-219\n");
    printf("    (2) imx-477\n");
}

static hi_void sample_vio_handle_sig(hi_s32 signo)
{
    if (signo == SIGINT || signo == SIGTERM) {
        g_sig_flag = 1;
    }
}

static hi_void sample_register_sig_handler(hi_void (*sig_handle)(hi_s32))
{
    struct sigaction sa;
    (hi_void)memset(&sa, 0, sizeof(struct sigaction));
    sa.sa_handler = sig_handle;
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, HI_NULL);
    sigaction(SIGTERM, &sa, HI_NULL);
}

static hi_s32 sample_vio_execute_case(hi_u32 case_index, case_info info)
{
    hi_s32 ret;

    switch (case_index) {
        case 1:
            ret = vi_with_sensor(0, info);
            break;
        case 2:
            ret = vi_with_sensor(1, info);
            break;
        default:
            ret = HI_FAILURE;
            printf("Error. This case is not implemented.\n");
            break;
    }

    return ret;
}

hi_s32 main(hi_s32 argc, hi_char *argv[])
{
    hi_s32 ret;
    hi_u32 index;
    case_info info;

    if (!strncmp(argv[1], "-h", 2)) { /* 2:arg num */
        sample_vio_usage(argv[0]);
        return HI_FAILURE;
    }

    if (!check_digit(argv[1][0]) || !check_digit(argv[2][0]) || !check_digit(argv[3][0])) { /* 2:arg len */
        printf("input is invalid\n", index);
        sample_vio_usage(argv[0]);
        return HI_FAILURE;
    }
    sample_register_sig_handler(sample_vio_handle_sig);

    index = atoi(argv[1]);
    if (index <= 0 || index > 2) {
        printf("index = %u is not supprot\n", index);
        return HI_FAILURE;
    }

    if (argc > 2) {
        info.sensor_num = atoi(argv[2]);
        if (info.sensor_num > 2 || info.sensor_num <= 0) {
            printf("sensor_num = %u is support 1 or 2", info.sensor_num);
            return HI_FAILURE;
        }
    }
    printf("set sensor_num=%u\n", info.sensor_num);
    if (argc > 3) {
        info.sensor_type= atoi(argv[3]); // 压缩方式
        if (info.sensor_type <= 0 && info.sensor_type > 2) {
            printf("sensor_type = %u is support 1 or 2\n", info.sensor_type);
            return HI_FAILURE;
        }
    }
    printf("set sensor_type = %u\n", info.sensor_type);

    ret = sample_vio_execute_case(index, info);
    if ((ret == HI_SUCCESS) && (g_sig_flag == 0)) {
        printf("\033[0;32mprogram exit normally!\033[0;39m\n");
    } else {
        printf("\033[0;31mprogram exit abnormally!\033[0;39m\n");
    }
    return ret;
}