#include "spi.h"

const char   *spiDev0  = "/dev/spidev0.0" ;
static uint8_t     spiBPW   = 8 ;
static uint16_t    spiDelay = 200 ;
static uint32_t     spiSpeeds ;
static int          spiFds ;

/*
 * SPIDataRW:
 *    Write and Read a block of data over the SPI bus.
 *    Note the data ia being read into the transmit buffer, so will
 *    overwrite it!
 *    This is also a full-duplex operation.
 *********************************************************************************
 *********************************************************************************/
int SPIDataRW (unsigned char *tx_data, unsigned char *rx_data, int len)
{
	int i = 0;
	struct spi_ioc_transfer spi ;

	memset (&spi, 0, sizeof (spi)) ;

	spi.tx_buf        = (unsigned long)tx_data ;
	spi.rx_buf        = (unsigned long)rx_data ;
	spi.len           = len ;
	spi.delay_usecs   = spiDelay ;
	spi.speed_hz      = spiSpeeds ;
	spi.bits_per_word = spiBPW ;

	return ioctl (spiFds, SPI_IOC_MESSAGE(1), &spi) ; //SPI_IOC_MESSAGE(1)
}


/*
 * SPISetupMode:
 *    Open the SPI device, and set it up, with the mode, etc.
 *********************************************************************************
 *********************************************************************************/

int SPISetupMode (int speed, int mode, int bpw)
{
	int speed_r, bits_r=0;
	uint8_t lsb=0, mode_r=0;
	int fd;

	if ((fd = open (spiDev0, O_RDWR)) < 0)
	{
		printf("Unable to open SPI device: %s\n", strerror (errno)) ;
		return -1;
	}

	spiSpeeds = speed ;
	spiFds = fd ;

	/*
	 *  Mode 0 CPOL=0, CPHA=0
	 *  Mode 1 CPOL=0, CPHA=1
	 *  Mode 2 CPOL=1, CPHA=0
	 *  Mode 3 CPOL=1, CPHA=1
	 *********************************************************************************
	 */
	if (ioctl (fd, SPI_IOC_WR_MODE, &mode) < 0)                     
	{                                                               
		printf("Can't set spi mode: %s\n", strerror (errno)) ;         
		return -1;                                                    
	}                                                               

	if (ioctl (fd, SPI_IOC_RD_MODE, &mode_r) < 0)                     
	{                                                               
		printf("Can't get spi mode: %s\n", strerror (errno)) ;        
		return -1;                                                 
	}    

	/*
	 * set bits_per_word
	 * 
	 *********************************************************************************
	 */
	if (ioctl (fd, SPI_IOC_WR_BITS_PER_WORD, &spiBPW) < 0)          
	{                                                               
		printf("Can't set bits per word: %s\n", strerror (errno))  ;  
		return -1;                                                    
	}                                                              

	if (ioctl (fd, SPI_IOC_RD_BITS_PER_WORD, &bits_r) < 0)          
	{                                                               
		printf("Can't get bits per word: %s\n", strerror (errno))  ;  
		return -1;                                                   
	}   

	/*
	 * set speed
	 *********************************************************************************
	 */
	if (ioctl (fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0)
	{
		printf("Can't set max speed hz: %s\n", strerror (errno));
		return -1;
	}

	if (ioctl (fd, SPI_IOC_RD_MAX_SPEED_HZ, &speed_r) < 0)
	{
		printf("Can't get max speed hz: %s\n", strerror (errno));
		return -1;
	}

	if (ioctl(fd, SPI_IOC_RD_LSB_FIRST, &lsb) < 0) {
		perror("SPI wr_lsb_fist");
		return -1;
	}
	/*设置LSB模式*/
	if (ioctl(fd, SPI_IOC_WR_LSB_FIRST, &lsb) < 0) {
		perror("SPI rd_lsb_fist");
		return -1;
	}

	printf("get mode: %d, speed: %d, bits: %d, lsb: %d\n", mode_r, speed_r, bits_r, lsb);

	return fd ;
}

char gpio_path[100];

int gpio_config(int num, const char *attr, const char *val)
{
	char file_path[100];
	int len;
	int fd;
	sprintf(file_path, "/sys/class/gpio/gpio%d/%s", num, attr);
	if (0 > (fd = open(file_path, O_WRONLY))) {
		perror("open error");
		return fd;
	}
	len = strlen(val);
	if (len != write(fd, val, len)) {
		perror("write error");
		close(fd);
		return -1;
	}
	close(fd);
	return 0;
}

int gpio_set_value(int num, const char* val)
{
	return gpio_config(num, "value", val);
}

int gpio_init(const char* num)
{
	int fd,len;

	sprintf(gpio_path, "/sys/class/gpio/gpio%s", num);

	if (access(gpio_path, F_OK)) {
		if (0 > (fd = open("/sys/class/gpio/export", O_WRONLY))) {
			perror("open error");
			return -1;
		}
		len=strlen(num);
		if (len != write(fd, num, len)) {
			perror("write error");
			close(fd);
			return -1;
		}
		close(fd); 
	}

	int gpio_num = atoi(num);
	if (gpio_config(gpio_num, "direction", "out"))
		return -1;

	if (gpio_config(gpio_num, "active_low", "0"))
		return -1;

	return 0;
}
//write a byte to led.
//dat:data/cmd
//cmd:0, command. 1, data
//gpio17:res, gpio27:dc
void spi_write_cmd(uint8_t dat,uint8_t cmd)
{
	int ret=0;
	uint8_t dat_r=0, i;
	if(cmd)
	{
		gpio_set_value(38, "1");	// dc set
	}
	else
		gpio_set_value(38, "0");	// dc clr
#if 1
	ret = SPIDataRW(&dat, &dat_r, 1);
	if(ret < 1)
		perror("can't send spi message");

#else
	for(i = 0; i < 8; i++)
	{
		gpio_set_value(79, "0");	// scl clr
		if(dat & 0x80)
		{
			gpio_set_value(80, "1");	// sda set
		}
		else
			gpio_set_value(80, "0");	// sda clr
		gpio_set_value(79, "1");	// scl set
		dat<<=1;
	}
#endif
	gpio_set_value(38, "1");		// dc set
	//usleep(100*1000);

	return;
}
