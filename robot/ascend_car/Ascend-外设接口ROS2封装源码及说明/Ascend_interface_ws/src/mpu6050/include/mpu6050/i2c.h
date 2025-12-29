#ifndef __I2C_H
#define __I2C_H

#include <stdio.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>

#define MPU6050_DEVICE  "/dev/i2c-7"
#define MPU6050_ADDR    0x68

int IIC_Init(void);
unsigned char IICreadByte(unsigned char dev, unsigned char reg, unsigned char *read_value);
unsigned char IICwriteByte(unsigned char dev, unsigned char reg, unsigned char data);
unsigned char IICwriteBits(unsigned char dev,unsigned char reg,unsigned char bitStart,unsigned char length,unsigned char data);
unsigned char IICwriteBit(unsigned char dev, unsigned char reg, unsigned char bitNum, unsigned char data);

#endif
