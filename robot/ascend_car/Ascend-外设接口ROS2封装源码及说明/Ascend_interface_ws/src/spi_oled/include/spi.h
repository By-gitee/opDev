#ifndef __SPI_H
#define __SPI_H

#include <stdint.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <linux/spi/spidev.h>
#include <linux/types.h>

#define OLED_CMD  0	//cmd
#define OLED_DATA 1	//data
/*
* SPIDataRW:
*    Write and Read a block of data over the SPI bus.
*    Note the data ia being read into the transmit buffer, so will
*    overwrite it!
*    This is also a full-duplex operation.
*********************************************************************************
*********************************************************************************/
int SPIDataRW (unsigned char *tx_data, unsigned char *rx_data, int len);
int SPISetupMode (int speed, int mode, int bpw);

void spi_write_cmd(uint8_t dat,uint8_t cmd);

int gpio_set_value(int num, const char* val);
int gpio_init(const char* num);
int gpio_config(int num, const char *attr, const char *val);

#endif
