/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Function of imx219 configure sensor
 * Author: Hisilicon multimedia software group
 * Create: 2023-03-06
 */

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include "linux/i2c-dev.h"
#include "imx219_cmos.h"
#include "imx219_cfgs.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

static hi_s32 g_fd[HI_ISP_MAX_PIPE_NUM] = {[0 ... (HI_ISP_MAX_PIPE_NUM - 1)] = -1};

#define I2C_DEV_FILE_NUM     16U
#define I2C_BUF_NUM          8U

static hi_s32 imx219_i2c_init(hi_vi_pipe viPipe)
{
    hi_s32 ret;
    char acDevFile[I2C_DEV_FILE_NUM];
    hi_s8 u8DevNum;
    SNS_CHECK_PIPE_RETURN(viPipe);
    if (g_fd[viPipe] >= 0) {
        return HI_SUCCESS;
    }

    u8DevNum = g_aunImx219BusInfo[viPipe].i2c_dev;
    ret = snprintf(acDevFile, sizeof(acDevFile), "/dev/i2c-%d", u8DevNum);
    if (ret < 0) {
        SNS_ERR_TRACE("snprintf err\n");
        return HI_FAILURE;
    }

    g_fd[viPipe] = open(acDevFile, O_RDWR, S_IRUSR | S_IWUSR);
    if (g_fd[viPipe] < 0) {
        SNS_ERR_TRACE("Open /dev/hi_i2c_drv-%u error! errno=%d.\n", u8DevNum, errno);
        return HI_FAILURE;
    }
    // Config sensor I2C device address (7bit)
    ret = ioctl(g_fd[viPipe], I2C_SLAVE_FORCE, (IMX219_I2C_ADDR >> 1));
    if (ret < 0) {
        SNS_ERR_TRACE("I2C_SLAVE_FORCE error!\n");
        close(g_fd[viPipe]);
        g_fd[viPipe] = -1;
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

static hi_s32 imx219_i2c_exit(hi_vi_pipe viPipe)
{
    SNS_CHECK_PIPE_RETURN(viPipe);
    if (g_fd[viPipe] >= 0) {
        close(g_fd[viPipe]);
        g_fd[viPipe] = -1;
        return HI_SUCCESS;
    }
    return HI_FAILURE;
}

hi_s32 imx219_read_register(hi_vi_pipe viPipe, hi_u32 addr)
{
    hi_s32 ret;
    hi_s32 idx = 0;
    char buf[I2C_BUF_NUM];
    SNS_CHECK_PIPE_RETURN(viPipe);
    if (g_fd[viPipe] < 0) {
        return HI_FAILURE;
    }
    if (IMX219_ADDR_BYTE == 2) { /* 2 byte */
        buf[idx] = (addr >> 8) & 0xff; /* shift 8 */
        idx++;
        buf[idx] = addr & 0xff;
        idx++;
    } else {
        buf[idx] = addr & 0xff;
        idx++;
    }
    ret = write(g_fd[viPipe], buf, IMX219_ADDR_BYTE);
    if (ret < HI_SUCCESS) {
        SNS_ERR_TRACE("I2C_WRITE DATA error!\n");
        return HI_FAILURE;
    }
    ret = read(g_fd[viPipe], buf, IMX219_DATA_BYTE);
    if (ret < HI_SUCCESS) {
        SNS_ERR_TRACE("I2C_READ DATA error!\n");
        return HI_FAILURE;
    }
    return buf[0];
}

hi_s32 imx219_write_register(hi_vi_pipe viPipe, hi_u32 addr, hi_u32 data)
{
    hi_s32 idx = 0;
    char buf[I2C_BUF_NUM];
    hi_s32 ret;
    SNS_CHECK_PIPE_RETURN(viPipe);

    if (g_fd[viPipe] < 0) {
        SNS_ERR_TRACE("fd err:%d\n", g_fd[viPipe]);
        return HI_FAILURE;
    }
    if (IMX219_ADDR_BYTE == 2) { /* 2 byte */
        buf[idx] = (addr >> 8) & 0xff; /* shift 8 */
        idx++;
        buf[idx] = addr & 0xff;
        idx++;
    } else {
        buf[idx] = addr & 0xff;
        idx++;
    }
    if (IMX219_DATA_BYTE == 2) { /* 2 byte */
        buf[idx] = (data >> 8) & 0xff; /* shift 8 */
        idx++;
        buf[idx] = data & 0xff;
        idx++;
    } else {
        buf[idx] = data & 0xff;
        idx++;
    }
    ret = write(g_fd[viPipe], buf, (IMX219_ADDR_BYTE + IMX219_DATA_BYTE));
    if (ret < HI_SUCCESS) {
        SNS_ERR_TRACE("I2C_WRITE DATA error!\n");
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

static hi_s32 imx219_readback_register(hi_vi_pipe viPipe, hi_u32 addr, hi_u8 data)
{
    hi_s32 read_back_data;
    hi_u32 addr_h;
    addr_h = (addr >> 8U) & 0xff; /* shift 8 */
    /* imx219 0x30xx 0x47xx 寄存器不可读 */
    /* 如果从地址的高位是 0x30 或�?0x47 , 不做回读直接返回成功 */
    if (addr_h == 0x30 || addr_h == 0x47) {
        return HI_SUCCESS;
    }
    read_back_data = imx219_read_register(viPipe, addr);
    if (data != (hi_u8)read_back_data) {
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

static void delay_ms(hi_u32 ms)
{
    usleep(ms * 1000); /* 1ms: 1000us */
    return;
}

void imx219_standby(hi_vi_pipe viPipe)
{
    SNS_CHECK_PIPE_VOID(viPipe);
    return;
}

void imx219_restart(hi_vi_pipe viPipe)
{
    SNS_CHECK_PIPE_VOID(viPipe);
    return;
}

void imx219_init(hi_vi_pipe viPipe)
{
    hi_u8 u8ImgMode;
    SNS_CHECK_PIPE_VOID(viPipe);
    hi_isp_sns_state *sensor_ctx = HI_NULL;
    sensor_ctx = imx219_get_ctx(viPipe);
    u8ImgMode = sensor_ctx->img_mode;
    hi_s32 ret = HI_SUCCESS;
    hi_u32 i;
    hi_u32 u32SeqEntries;
    hi_u16 i2caddr;
    hi_u8 i2cvalue;
    /* 2. sensor i2c init */
    imx219_i2c_init(viPipe);
    // Delay 100ms
    delay_ms(100);
    if (u8ImgMode == IMX219_24M_RAW10_MODE) {
        u32SeqEntries = sizeof(imx219_raw10_24M_20fps_mipi2lane) / sizeof(cis_cfg_imx219_t);
        for (i = 0 ; i < u32SeqEntries; i++) {
            i2caddr = imx219_raw10_24M_20fps_mipi2lane[i].address;
            i2cvalue = imx219_raw10_24M_20fps_mipi2lane[i].value;
            ret = imx219_write_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx219 vipipe:%d, i2caddr:%u,i2cvalue:%u, write register failed!\n",
                    viPipe, i2caddr, i2cvalue);
                return;
            }
            ret = imx219_readback_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx219 vipipe:%d, i2caddr:%u,i2cvalue:%u, read back failed!\n",
                    viPipe, i2caddr, i2cvalue);
            }
        }
    } else if (u8ImgMode == IMX219_24M_RAW8_MODE) {
        u32SeqEntries = sizeof(imx219_raw8_24M_20fps_mipi2lane) / sizeof(cis_cfg_imx219_t);
        for (i = 0 ; i < u32SeqEntries; i++) {
            i2caddr = imx219_raw8_24M_20fps_mipi2lane[i].address;
            i2cvalue = imx219_raw8_24M_20fps_mipi2lane[i].value;
            ret = imx219_write_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx219 vipipe:%d, i2caddr:%u,i2cvalue:%u, write register failed!\n",
                    viPipe, i2caddr, i2cvalue);
                return;
            }
            ret = imx219_readback_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx219 vipipe:%d, i2caddr:%u,i2cvalue:%u, read back failed!\n",
                    viPipe, i2caddr, i2cvalue);
            }
        }
    } else if (u8ImgMode == IMX219_1920x1080_RAW10_30FPS_MODE) {
        u32SeqEntries = sizeof(imx219_raw10_1920x1080_30fps_mipi2lane) / sizeof(cis_cfg_imx219_t);
        for (i = 0 ; i < u32SeqEntries; i++) {
            i2caddr = imx219_raw10_1920x1080_30fps_mipi2lane[i].address;
            i2cvalue = imx219_raw10_1920x1080_30fps_mipi2lane[i].value;
            ret = imx219_write_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx219 vipipe:%d, i2caddr:%u,i2cvalue:%u, write register failed!\n",
                    viPipe, i2caddr, i2cvalue);
                return;
            }
            ret = imx219_readback_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx219 vipipe:%d, i2caddr:%u,i2cvalue:%u, read back failed!\n",
                    viPipe, i2caddr, i2cvalue);
            }
        }
    } else {
            SNS_ERR_TRACE("imx219 Not support ImgMode: %d, should less than %d\n",
                sensor_ctx->img_mode, IMX219_MODE_BUTT);
        return;
    }

    SNS_INFO_TRACE("Camera sensor IMX219 (viPipe: %d, Operation mode: %s) init success!\n",
        viPipe, g_astImx219ModeTbl[u8ImgMode].pszModeName);
    sensor_ctx->init = HI_TRUE; // Initialization flag
    return ;
}

void imx219_exit(hi_vi_pipe viPipe)
{
    SNS_CHECK_PIPE_VOID(viPipe);
    imx219_i2c_exit(viPipe);
    return;
}
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /*  End of #ifdef __cplusplus  */
