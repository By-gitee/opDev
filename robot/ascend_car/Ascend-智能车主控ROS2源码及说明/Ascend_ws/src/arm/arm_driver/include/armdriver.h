#ifndef __ArmDriver_H_
#define __ArmDriver_H_

#include "rclcpp/rclcpp.hpp"
#include <iostream>
#include <string.h>
#include <string> 
#include <iostream>
#include <math.h>
#include <stdlib.h>    
#include <unistd.h>      
#include <sys/types.h>
#include <sys/stat.h>
#include <serial/serial.h>
#include <pthread.h>
#include <fcntl.h>          
#include <stdbool.h>
#include <thread>
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include <serial/serial.h>
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/int8.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/quaternion.hpp>
#include <cstdio>
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include "sensor_msgs/msg/joint_state.hpp"


using namespace std;
//using std::placeholders::_1;


//Macro definition
//宏定义
#define POSE_FRAME_HEADER      0X55 
#define FRAME_TAIL        0X7D       //Frame tail //帧尾
#define RECEIVE_DATA_SIZE 24         //The length of the data sent by the lower computer //下位机发送过来的数据的长度
#define SEND_DATA_SIZE    11         //The length of data sent by ROS to the lower machine //ROS向下位机发送的数据的长度
#define SEND_POSE_DATA_SIZE    18
#define PI 				  3.1415926f //PI //圆周率

//Data structure for speed and position
//速度、位置数据结构体
typedef struct __Vel_Pos_Data_
{
	float X;
	float Y;
	float Z;
}Vel_Pos_Data;

union FourBytesToFloat {
    unsigned int intValue;
    float floatValue;
};

//The structure of the ROS to send data to the down machine
//ROS向下位机发送数据的结构体
typedef struct _SEND_DATA_  
{
	    uint8_t tx[SEND_DATA_SIZE];
		float X_speed;	       
		float Y_speed;           
		float Z_speed;         
		unsigned char Frame_Tail; 
}SEND_DATA;

typedef struct _POSE_SEND_DATA_  
{
	    uint8_t tx[SEND_POSE_DATA_SIZE]; 
}POSE_SEND_DATA;


class ArmDriver : public rclcpp::Node
{
public:
    // 构造函数,有一个参数为节点名称
    ArmDriver(std::string name);

    ~ArmDriver();
    void UpDate();

private:

    // 声名定时器指针
    rclcpp::TimerBase::SharedPtr timer_;
    // 声明话题发布者订阅者指针
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr Cmd_Vel_Sub;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr Pose_Sub;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr endpose_publisher;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_publisher;
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr Cmd_Sub;
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr Gripper_Sub;

    serial::Serial Stm32_Serial; 
    geometry_msgs::msg::Pose  end_pose;
    sensor_msgs::msg::JointState joint_state;

    void timer_callback();
    void Gripper_Callback(const std_msgs::msg::Int8 &direct); 
    void Pose_Callback(const geometry_msgs::msg::Pose &pose);
    void cmd_Callback(const std_msgs::msg::Int8 &cmd);         
    int f;
    string usart_port_name;
    int serial_baud_rate;      //Serial communication baud rate //串口通信波特率
    POSE_SEND_DATA Send_Pose_Data;

};



#endif
