/**
* @File vo_mipix.c
* @Description Provide util function for reading RAW & YUV data from the VI PIPE & CHN
*
* Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "func.h"
#include "hi_mipi_tx.h"
#include "linux/i2c-dev.h"

#define MIPI_TX_DEV_NAME "/dev/ot_mipi_tx"

static hi_s32 g_sample_comm_mipi_fd = -1;

const sample_mipi_tx_config mipi_tx_rpi_configs[1] = {
    {
            /* for combo dev config */
            .intf_sync = HI_MIPI_TX_OUT_USER,

            /* for user sync */
            .combo_dev_cfg = {
                .devno = 0,
                .lane_id = {0, -1, -1, -1},
                .out_mode = OUT_MODE_DSI_VIDEO,
                .out_format = OUT_FORMAT_RGB_24BIT,
                .video_mode =  BURST_MODE,
                .sync_info = {
                    .hsa_pixels = 20,
                    .hbp_pixels = 26,
                    .hact_pixels = 800,
                    .hfp_pixels = 70,
                    .vsa_lines = 2,
                    .vbp_lines = 21,
                    .vact_lines = 480,
                    .vfp_lines = 7,
                },
                .phy_data_rate = 720,
                .pixel_clk = 27448,
            },
    }
};

static hi_s32 sample_comm_mipi_tx_get_config(const combo_dev_cfg_t **mipi_tx_config)
{
    mipi_tx_intf_sync mipi_intf_sync;
    const sample_mipi_tx_config *tx_config = &mipi_tx_rpi_configs[0];

    mipi_intf_sync = tx_config->intf_sync;

    if (mipi_intf_sync == HI_MIPI_TX_OUT_USER) {
        *mipi_tx_config = &tx_config->combo_dev_cfg;
        if ((*mipi_tx_config)->phy_data_rate == 0) {
            printf("error: not set mipi tx user config\n");
            return HI_FAILURE;
        }
    }
    return HI_SUCCESS;
}

static void sample_comm_mipi_tx_do_close_fd(void)
{
    if (g_sample_comm_mipi_fd != -1) {
        close(g_sample_comm_mipi_fd);
        g_sample_comm_mipi_fd = -1;
    }
}

static hi_s32 g_fd = -1;

static hi_s32 rpi_i2c_init()
{
    char acDevFile[16] = {0};
    hi_u8 u8DevNum;
    hi_s32 ret;

    u8DevNum = 13;
    ret = snprintf(acDevFile, sizeof(acDevFile), "/dev/i2c-%u", u8DevNum);
    if (ret < 0) {
        return HI_FAILURE;
    }

    g_fd = open(acDevFile, O_RDWR, S_IRUSR | S_IWUSR);

    if (g_fd < 0) {
        printf("Open /dev/i2c-%u error!\n", u8DevNum);
        return HI_FAILURE;
    }
    ret = ioctl(g_fd, I2C_SLAVE_FORCE, (0x45));
    if (ret < 0) {
        printf("I2C_SLAVE_FORCE error!\n");
        close(g_fd);
        g_fd = -1;
        return ret;
    }

    return HI_SUCCESS;
}

static hi_s32 rpi_i2c_write_register(hi_u32 addr, hi_u32 data)
{
    if (g_fd < 0) {
        return HI_FAILURE;
    }

    hi_u32 idx = 0;
    hi_s32 ret;
    hi_u8 buf[8];

    buf[idx] = addr & 0xff;
    idx++;

    buf[idx] = data & 0xff;
    idx++;

    ret = write(g_fd, buf, 2);
    if (ret < 0) {
        printf("I2C_WRITE error!\n");
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

static const hi_u8 cmd1[] = {
    0x10U, 0x02U, 0x03U, 0x00, 0x00, 0x00
};

static const hi_u8 cmd2[] = {
    0x64U, 0x01U, 0x0cU, 0x00, 0x00, 0x00
};

static const hi_u8 cmd3[] = {
    0x68U, 0x01U, 0x0cU, 0x00, 0x00, 0x00
};

static const hi_u8 cmd4[] = {
    0x44U, 0x01U, 0x00, 0x00, 0x00, 0x00
};

static const hi_u8 cmd5[] = {
    0x48U, 0x01U, 0x00, 0x00, 0x00, 0x00
};

static const hi_u8 cmd6[] = {
    0x14U, 0x01U, 0x15U, 0x00, 0x00, 0x00
};

static const hi_u8 cmd7[] = {
    0x50U, 0x04U, 0x60U, 0x00, 0x00, 0x00
};

static const hi_u8 cmd8[] = {
    0x20U, 0x04U, 0x52U, 0x01U, 0x10U, 0x00
};

static const hi_u8 cmd9[] = {
    0x24U, 0x04U, 0x14U, 0x00, 0x24U, 0x00
};

static const hi_u8 cmd10[] = {
    0x28U, 0x04U, 0x20U, 0x03U, 0x90U, 0x00
};

static const hi_u8 cmd11[] = {
    0x2cU, 0x04U, 0x02U, 0x00, 0x15U, 0x00
};

static const hi_u8 cmd12[] = {
    0x30U, 0x04U, 0xe0U, 0x01U, 0x07U, 0x00
};

static const hi_u8 cmd13[] = {
    0x34U, 0x04U, 0x01U, 0x00, 0x00, 0x00
};

static const hi_u8 cmd14[] = {
    0x64U, 0x04U, 0x0fU, 0x04U, 0x00, 0x00
};

static const hi_u8 cmd15[] = {
    0x04U, 0x01U, 0x01U, 0x00, 0x00, 0x00
};

static const hi_u8 cmd16[] = {
    0x04U, 0x02U, 0x01U, 0x00, 0x00, 0x00
};

static mipi_tx_cmd_info g_cmd_info_video_mode[CMD_COUNT] = {
    /* LANE */
    {{0, 0, 1, 0x29, 0x0006, cmd1}, 50000U},
    /* D0S_CLRSIPOCOUNT */
    {{0, 0, 1, 0x29, 0x0006, cmd2}, 50000U},
    /* D1S_CLRSIPOCOUNT */
    {{0, 0, 1, 0x29, 0x0006, cmd3}, 50000U},
    /* D0S_ATMR */
    {{0, 0, 1, 0x29, 0x0006, cmd4}, 50000U},
    /* D1S_ATMR */
    {{0, 0, 1, 0x29, 0x0006, cmd5}, 50000U},
    /* LPTXTIMCNT */
    {{0, 0, 1, 0x29, 0x0006, cmd6}, 50000U},
    /* SPICMR/SPICTRL */
    {{0, 0, 1, 0x29, 0x0006, cmd7}, 50000U},
    /* PORT/LCDCTRL */
    {{0, 0, 1, 0x29, 0x0006, cmd8}, 50000U},
    /* HBPR/HSR */
    {{0, 0, 1, 0x29, 0x0006, cmd9}, 50000U},
    /* HFPR/HDISP */
    {{0, 0, 1, 0x29, 0x0006, cmd10}, 50000U},
    /* VBFR/VSR */
    {{0, 0, 1, 0x29, 0x0006, cmd11}, 50000U},
    /* VFPR/VDISP */
    {{0, 0, 1, 0x29, 0x0006, cmd12}, 50000U},
    /* VFUEN */
    {{0, 0, 1, 0x29, 0x0006, cmd13}, 50000U},
    /* SYSCTRL */
    {{0, 0, 1, 0x29, 0x0006, cmd14}, 50000U},
    /* STARTPPI */
    {{0, 0, 1, 0x29, 0x0006, cmd15}, 50000U},
    /* STARTDSI */
    {{0, 0, 1, 0x29, 0x0006, cmd16}, 50000U},
};

static hi_s32 vo_mst_mipi_tx_send_one_cmd(cmd_info_t *cmd_info)
{
    hi_s32 ret;
    ret = ioctl(g_sample_comm_mipi_fd, HI_MIPI_TX_SET_CMD, cmd_info);
    if (ret != HI_SUCCESS) {
        printf("MIPI_TX SET CMD failed\n");
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}

static hi_s32 mipi_tx_init_rpi_screen_v2()
{
    hi_s32 ret;
    cmd_info_t ci = {0};
    hi_u32 loop;
    hi_u32 loop_num;

    /* init i2c for power up */
    ret = rpi_i2c_init();
    ret |= rpi_i2c_write_register(0x85, 0x1);
    usleep(200000);
    ret |= rpi_i2c_write_register(0x81, 0x4);
    usleep(100000);
    ret |= rpi_i2c_write_register(0x86, 0xff);
    usleep(100000);

    if (ret != HI_SUCCESS) {
        printf("iic init failed!\n");
        return HI_FAILURE;
    }

    /* init screen */
    const sample_mipi_tx_config tx_config = {
        .cmd_count = CMD_COUNT,
        .cmd_info = &g_cmd_info_video_mode,
    };
    loop_num = tx_config.cmd_count;

    for (loop = 0; loop < loop_num; loop++) {
        (hi_void)memcpy(&ci, &(tx_config.cmd_info[loop].cmd_info), sizeof(cmd_info_t));
        ret = vo_mst_mipi_tx_send_one_cmd(&ci);
        if (ret != HI_SUCCESS) {
            printf("screen init failed!\n");
            return HI_FAILURE;
        }
        usleep(tx_config.cmd_info[loop].usleep_value);
    }

    return HI_SUCCESS;
}

hi_s32 start_mipi_tx()
{
    hi_s32 ret;
    const combo_dev_cfg_t *combo_config = HI_NULL;

    g_sample_comm_mipi_fd = open(MIPI_TX_DEV_NAME, O_RDONLY);
    if (g_sample_comm_mipi_fd < 0) {
        printf("open mipi dev file (%s) fail\n", MIPI_TX_DEV_NAME);
        return HI_FAILURE;
    }

    ret = sample_comm_mipi_tx_get_config(&combo_config);
    if (ret != HI_SUCCESS) {
        printf("get mipi tx config fail\n");
        sample_comm_mipi_tx_do_close_fd();
        return ret;
    }

    /* step1 */
    ret = ioctl(g_sample_comm_mipi_fd, HI_MIPI_TX_SET_DEV_CFG, combo_config);
    if (ret != HI_SUCCESS) {
        printf("ioctl mipi tx HI_MIPI_TX_SET_DEV_CFG fail at ret\n");
        sample_comm_mipi_tx_do_close_fd();
        return ret;
    }

    /* step2 */
    ret = mipi_tx_init_rpi_screen_v2();

    if (ret != HI_SUCCESS) {
        printf("init screen failed\n");
        sample_comm_mipi_tx_do_close_fd();
        return ret;
    }

    ret = ioctl(g_sample_comm_mipi_fd, HI_MIPI_TX_ENABLE, NULL);
    if (ret != HI_SUCCESS) {
        printf("ioctl mipi tx HI_MIPI_TX_ENABLE fail at ret\n");
        sample_comm_mipi_tx_do_close_fd();
        return ret;
    }

    return HI_SUCCESS;
}

hi_s32 close_mipi_tx()
{
    hi_s32 ret;
    ret = ioctl(g_sample_comm_mipi_fd, HI_MIPI_TX_DISABLE, NULL);
    if (ret != HI_SUCCESS) {
        printf("ioctl mipi tx HI_MIPI_TX_DISABLE fail at ret\n");
        sample_comm_mipi_tx_do_close_fd();
        return ret;
    }
    sample_comm_mipi_tx_do_close_fd();

    return HI_SUCCESS;
}