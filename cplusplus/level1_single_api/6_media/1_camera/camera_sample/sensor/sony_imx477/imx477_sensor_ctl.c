/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Function of imx477 configure sensor
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
#include "imx477_cmos.h"
#include "imx477_config.h"

static hi_s32 g_fd[HI_ISP_MAX_PIPE_NUM] = {[0 ... (HI_ISP_MAX_PIPE_NUM - 1)] = -1};

#define I2C_DEV_FILE_NUM     16U
#define I2C_BUF_NUM          8U

static hi_s32 imx477_i2c_init(hi_vi_pipe viPipe)
{
    SNS_INFO_TRACE("Camera sensor IMX477 i2c init\n");

    hi_s32 ret;
    char acDevFile[I2C_DEV_FILE_NUM];
    hi_u8 u8DevNum;

    SNS_CHECK_PIPE_RETURN(viPipe);

    if (g_fd[viPipe] >= 0) {
        return HI_SUCCESS;
    }

    u8DevNum = (hi_u8)g_aunImx477BusInfo[viPipe].i2c_dev;
    SNS_INFO_TRACE("Camera sensor IMX477 vipipe :%d , i2c_dev:%u, i2c_addr:%x\n",
        viPipe, u8DevNum, IMX477_I2C_ADDR >> 1);
    ret = snprintf(acDevFile, sizeof(acDevFile),
        sizeof(acDevFile) - 1,  "/dev/i2c-%u", u8DevNum);
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
    ret = ioctl(g_fd[viPipe], I2C_SLAVE_FORCE, IMX477_I2C_ADDR >> 1);
    if (ret < 0) {
        SNS_ERR_TRACE("I2C_SLAVE_FORCE error!\n");
        close(g_fd[viPipe]);
        g_fd[viPipe] = -1;
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

static hi_s32 imx477_i2c_exit(hi_vi_pipe viPipe)
{
    SNS_CHECK_PIPE_RETURN(viPipe);

    if (g_fd[viPipe] >= 0) {
        close(g_fd[viPipe]);
        g_fd[viPipe] = -1;
        return HI_SUCCESS;
    }
    return HI_FAILURE;
}

hi_s32 imx477_read_register(hi_vi_pipe viPipe, hi_u32 addr)
{
    hi_s32 ret;
    hi_s32 idx = 0;
    char buf[I2C_BUF_NUM];

    SNS_CHECK_PIPE_RETURN(viPipe);

    if (g_fd[viPipe] < 0) {
        return HI_FAILURE;
    }

    if (IMX477_ADDR_BYTE == 2) { /* 2 byte */
        buf[idx] = (addr >> 8) & 0xff; /* shift 8 */
        idx++;
        buf[idx] = addr & 0xff;
        idx++;
    } else {
        buf[idx] = addr & 0xff;
        idx++;
    }

    ret = write(g_fd[viPipe], buf, IMX477_ADDR_BYTE);
    if (ret < HI_SUCCESS) {
        SNS_ERR_TRACE("I2C_WRITE DATA error!\n");
        return HI_FAILURE;
    }

    ret = read(g_fd[viPipe], buf, IMX477_DATA_BYTE);
    if (ret < HI_SUCCESS) {
        SNS_ERR_TRACE("I2C_READ DATA error!\n");
        return HI_FAILURE;
    }

    return buf[0];
}

hi_s32 imx477_write_register(hi_vi_pipe viPipe, hi_u32 addr, hi_u32 data)
{
    hi_s32 idx = 0;
    char buf[I2C_BUF_NUM];
    hi_s32 ret;

    SNS_CHECK_PIPE_RETURN(viPipe);
    SNS_INFO_TRACE("viPipe:%d, addr:%x, data:%x\n", viPipe, addr, data);
    if (g_fd[viPipe] < 0) {
        SNS_ERR_TRACE("fd err:%d\n", g_fd[viPipe]);
        return HI_FAILURE;
    }

    if (IMX477_ADDR_BYTE == 2) { /* 2 byte */
        buf[idx] = (addr >> 8) & 0xff; /* shift 8 */
        idx++;
        buf[idx] = addr & 0xff;
        idx++;
    } else {
        buf[idx] = addr & 0xff;
        idx++;
    }

    if (IMX477_DATA_BYTE == 2) { /* 2 byte */
        buf[idx] = (data >> 8) & 0xff; /* shift 8 */
        idx++;
        buf[idx] = data & 0xff;
        idx++;
    } else {
        buf[idx] = data & 0xff;
        idx++;
    }

    ret = write(g_fd[viPipe], buf, (IMX477_ADDR_BYTE + IMX477_DATA_BYTE));
    if (ret < HI_SUCCESS) {
        SNS_ERR_TRACE("I2C_WRITE DATA error!\n");
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

static hi_s32 imx477_readback_register(hi_vi_pipe viPipe, hi_s32 addr, hi_s32 data)
{
    SNS_CHECK_PIPE_RETURN(viPipe);
    hi_s32 read_back_data;
    read_back_data = imx477_read_register(viPipe, addr);
    if (data != (hi_u8)read_back_data) {
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

static hi_void delay_ms(hi_u32 ms)
{
    usleep(ms * 1000U); /* 1ms: 1000us */
    return;
}

hi_void imx477_standby(hi_vi_pipe viPipe)
{
    SNS_CHECK_PIPE_VOID(viPipe);
    hi_s32 ret;
    ret = imx477_write_register(viPipe, 0x100, 0x0);
    if (ret != HI_SUCCESS) {
        SNS_ERR_TRACE("imx477 vipipe:%d standby, write register failed!\n", viPipe);
    }
    return;
}

hi_void imx477_restart(hi_vi_pipe viPipe)
{
    SNS_CHECK_PIPE_VOID(viPipe);
    return;
}

hi_void imx477_init(hi_vi_pipe viPipe)
{
    hi_s32 ret = HI_SUCCESS;
    hi_u8 img_mode;
    SNS_INFO_TRACE("Camera sensor IMX477 viPipe:%d init\n", viPipe);
    SNS_CHECK_PIPE_VOID(viPipe);
    hi_isp_sns_state *sensor_ctx = HI_NULL;
    sensor_ctx = imx477_get_ctx(viPipe);
    img_mode = sensor_ctx->img_mode;

    hi_u32 i;
    hi_u32 u32SeqEntries;
    hi_u16 i2caddr;
    hi_u8 i2cvalue;
    /* 2. sensor i2c init */
    imx477_i2c_init(viPipe);
    // imx477 在接收到解复位信号后，由模组上的电路控制模组的供电，存在一定的时延
    // delay 500 ms
    delay_ms(500);
    /* When sensor first init, config all registers */
    if (img_mode == IMX477_4056x3040_RAW12_25FPS) {
        u32SeqEntries = sizeof(imx477_4056x3040_raw12_25fps) / sizeof(cis_cfg_imx477_t);
        for (i = 0 ; i < u32SeqEntries; i++) {
            i2caddr = imx477_4056x3040_raw12_25fps[i].address;
            i2cvalue = imx477_4056x3040_raw12_25fps[i].value;
            ret = imx477_write_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx477 vipipe:%d, i2caddr:%u,i2cvalue:%u, write register failed!\n",
                    viPipe, i2caddr, i2cvalue);
                return;
            }
            ret = imx477_readback_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx477 vipipe:%d, i2caddr:%u,i2cvalue:%u, read back failed!\n",
                    viPipe, i2caddr, i2cvalue);
            }
        }
    } else if (img_mode == IMX477_4056x3040_DOL2_RAW10_15fps) {
        u32SeqEntries = sizeof(imx477_4056x3040_dol2_raw10_15fps) / sizeof(cis_cfg_imx477_t);
        for (i = 0 ; i < u32SeqEntries; i++) {
            i2caddr = imx477_4056x3040_dol2_raw10_15fps[i].address;
            i2cvalue = imx477_4056x3040_dol2_raw10_15fps[i].value;
            ret = imx477_write_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx477 vipipe:%d, i2caddr:%u,i2cvalue:%u, write register failed!\n",
                    viPipe, i2caddr, i2cvalue);
                return;
            }
            ret = imx477_readback_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx477 vipipe:%d, i2caddr:%u,i2cvalue:%u, read back failed!\n",
                    viPipe, i2caddr, i2cvalue);
            }
        }
    } else if (img_mode == IMX477_4K_45FPS) {
        u32SeqEntries = sizeof(imx477_4k_raw10_45fps) / sizeof(cis_cfg_imx477_t);
        for (i = 0 ; i < u32SeqEntries; i++) {
            i2caddr = imx477_4k_raw10_45fps[i].address;
            i2cvalue = imx477_4k_raw10_45fps[i].value;
            ret = imx477_write_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx477 vipipe:%d, i2caddr:%u,i2cvalue:%u, write register failed!\n",
                    viPipe, i2caddr, i2cvalue);
                return;
            }
            ret = imx477_readback_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx477 vipipe:%d, i2caddr:%u,i2cvalue:%u, read back failed!\n",
                    viPipe, i2caddr, i2cvalue);
            }
        }
    } else if (img_mode == IMX477_1920x1080_RAW12) {
        u32SeqEntries = sizeof(imx477_1920x1080_raw12) / sizeof(cis_cfg_imx477_t);
        for (i = 0 ; i < u32SeqEntries; i++) {
            i2caddr = imx477_1920x1080_raw12[i].address;
            i2cvalue = imx477_1920x1080_raw12[i].value;
            ret = imx477_write_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx477 vipipe:%d, i2caddr:%u,i2cvalue:%u, write register failed!\n",
                    viPipe, i2caddr, i2cvalue);
                return;
            }
            ret = imx477_readback_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx477 vipipe:%d, i2caddr:%u,i2cvalue:%u, read back failed!\n",
                    viPipe, i2caddr, i2cvalue);
            }
        }
    } else if (img_mode == IMX477_1920x1080_DOL2_RAW10) {
        u32SeqEntries = sizeof(imx477_1920x1080_dol2_raw10) / sizeof(cis_cfg_imx477_t);
        for (i = 0 ; i < u32SeqEntries; i++) {
            i2caddr = imx477_1920x1080_dol2_raw10[i].address;
            i2cvalue = imx477_1920x1080_dol2_raw10[i].value;
            ret = imx477_write_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx477 vipipe:%d, i2caddr:%u,i2cvalue:%u, write register failed!\n",
                    viPipe, i2caddr, i2cvalue);
                return;
            }
            ret = imx477_readback_register(viPipe, i2caddr, i2cvalue);
            if (ret != HI_SUCCESS) {
                SNS_ERR_TRACE("imx477 vipipe:%d, i2caddr:%u,i2cvalue:%u, read back failed!\n",
                    viPipe, i2caddr, i2cvalue);
            }
        }
    } else {
        SNS_ERR_TRACE("imx477 Not support ImgMode: %d, should less than %d\n",
            sensor_ctx->img_mode, IMX477_MODE_BUTT);
        return;
    }
    SNS_INFO_TRACE("Camera sensor IMX477 (viPipe: %d, Operation mode: %s) init success!\n",
        viPipe, g_astImx477ModeTbl[img_mode].pszModeName);
    sensor_ctx->init = HI_TRUE; // Initialization flag
    return;
}

hi_void imx477_exit(hi_vi_pipe viPipe)
{
    SNS_CHECK_PIPE_VOID(viPipe);
    imx477_standby(viPipe);
    imx477_i2c_exit(viPipe);

    return;
}

/* Chip ID */
#define IMX477_CHIP_ID_REG      0x0016
#define IMX477_CHIP_ID          0x0477
#define CHIP_ID_LENGTH          2U
/* Verify chip ID */
hi_s32 imx477_identify_module(hi_vi_pipe viPipe)
{
    SNS_CHECK_PIPE_RETURN(viPipe);
    imx477_i2c_init(viPipe);
    delay_ms(20); // delay 20 ms

    if (HIGH_8BITS(IMX477_CHIP_ID) != imx477_read_register(viPipe, IMX477_CHIP_ID_REG) ||
        LOW_8BITS(IMX477_CHIP_ID) != imx477_read_register(viPipe, IMX477_CHIP_ID_REG + 1)) {
        SNS_INFO_TRACE("viPipe=%d, imx477 chip id recognition failed!\n", viPipe);
        imx477_i2c_exit(viPipe);
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}