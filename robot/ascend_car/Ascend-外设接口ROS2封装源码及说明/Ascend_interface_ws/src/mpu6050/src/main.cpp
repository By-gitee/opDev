#include "rclcpp/rclcpp.hpp"
#include <iostream>
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/accel.hpp"
#include <cstdio>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>

extern "C" {
#include "mpu6050.h"
#include "i2c.h"
}

using namespace std;

using acc_msg = geometry_msgs::msg::Accel;

class mpu6050: public rclcpp::Node{
	public:
		mpu6050(const string & node_name): Node(node_name)
		{
			publisher_acc = this->create_publisher<acc_msg>("imu", 10);
		};
		virtual ~mpu6050(){};

		void init()
		{
			IIC_Init();
			MPU6050_initialize();
			read_value = MPU6050_getDeviceID();
			printf("test read ID value is %d\n", read_value);
		};

		void msg_pub(acc_msg &msg)
		{
			publisher_acc->publish(msg);
		};



	private:
		rclcpp::Publisher<acc_msg>::SharedPtr publisher_acc;
		int read_value=0;
};


int main(int argc, char ** argv)
{
  acc_msg acc;
  rclcpp::init(argc, argv);
  string name = "mpu6050";
  /*
	printf("init value vertify :read value is %d, %d, %d, %d, %d\n", 
			IICreadByte(MPU6050_DEFAULT_ADDRESS, MPU6050_RA_PWR_MGMT_1,&read_value),
			IICreadByte(MPU6050_DEFAULT_ADDRESS, MPU6050_RA_SMPLRT_DIV,&read_value),
			IICreadByte(MPU6050_DEFAULT_ADDRESS, MPU6050_RA_GYRO_CONFIG,&read_value),
			IICreadByte(MPU6050_DEFAULT_ADDRESS, MPU6050_RA_ACCEL_CONFIG,&read_value),
			IICreadByte(MPU6050_DEFAULT_ADDRESS, MPU6050_RA_CONFIG,&read_value));
      */
	auto node_ = make_shared<mpu6050>(name);
  node_->init();

	while(1)
	{
    uint16_t temp_imu[3]; 
		printf("MPU 6050 temp is %d\n", Read_Temperature());
	
		Read_Acc(temp_imu);
    acc.linear.x = temp_imu[0];
    acc.linear.y = temp_imu[1];
    acc.linear.z = temp_imu[2];
		//printf("加速度：%8d%8d%8d\n",acc.linear[0],acc.linear[1],acc.linear[2]);
	
		Read_Gyro(temp_imu);
		//printf("陀螺仪%8d%8d%8d\n",acc.angular[0],acc.angular[1],acc.angular[2]);
    acc.angular.x = temp_imu[0];
    acc.angular.y = temp_imu[1];
    acc.angular.z = temp_imu[2];
    
    node_->msg_pub(acc);
   
		sleep(1);
	}

	return 0;
}
