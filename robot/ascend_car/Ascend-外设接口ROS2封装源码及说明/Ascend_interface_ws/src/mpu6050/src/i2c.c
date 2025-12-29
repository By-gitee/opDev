#include "i2c.h"

int fd_mpu;

int IIC_Init(void)
{
	fd_mpu = open(MPU6050_DEVICE, O_RDWR);
	if(fd_mpu <= 0)
	{
		printf("open /dev/i2c-7 failed!\n");
		return -1;
	}
	printf("open device successed!\n");

	if (ioctl(fd_mpu, I2C_SLAVE_FORCE, MPU6050_ADDR) < 0)
	{
		printf("set slave address failed \n");
		return -1;
	}
	printf("set slave address successed!\n");

	return fd_mpu;
}

unsigned char IICreadByte(unsigned char dev, unsigned char reg, unsigned char *read_value)
{
	write(fd_mpu, &reg, 1);
	read(fd_mpu, read_value, 1);
	return *read_value;
}

unsigned char IICreadBytes(unsigned char dev, unsigned char reg, unsigned char *read_value, int num)
{
	write(fd_mpu, &reg, 1);
	read(fd_mpu, read_value, num);
	return 0;
}

unsigned char IICwriteByte(unsigned char dev, unsigned char reg, unsigned char data)
{
	int buf[2], count;
	buf[0] = reg;
	buf[1] = data;
	count = write(fd_mpu, buf, 2);
	return count;
}

unsigned char IICwriteBits(unsigned char dev,unsigned char reg,unsigned char bitStart,unsigned char length,unsigned char data)
{
	uint8_t b, ret, rb;
	if (IICreadByte(dev, reg, &b) || 1) {
		unsigned char mask = (0xFF << (bitStart + 1)) | 0xFF >> ((8 - bitStart) + length - 1);
		data <<= (8 - length);
		data >>= (7 - bitStart);
		b &= mask;
		b |= data;
		ret = IICwriteByte(dev, reg, b);
		IICreadByte(dev, reg, &rb);
		if(b != rb || ret != 2)
		{
			//printf("failed: wrtie %d, but read %d, ret is %d\n", b, rb, ret);
			return -1;
		}
	} else {
		printf("write bits error!\n");
		return 0;
	}
}

unsigned char IICwriteBit(unsigned char dev, unsigned char reg, unsigned char bitNum, unsigned char data)
{
	uint8_t b, ret, rb;
	IICreadByte(dev, reg, &b);
	b = (data != 0) ? (b | (1 << bitNum)) : (b & ~(1 << bitNum));
	ret = IICwriteByte(dev, reg, b);
	IICreadByte(dev, reg, &rb);
	if(b != rb || ret != 2)
		{
			printf("write failed: wrtie %d, but read %d, ret is %d\n", b, rb, ret);
			return -1;
		}
		return 0;
	
}

