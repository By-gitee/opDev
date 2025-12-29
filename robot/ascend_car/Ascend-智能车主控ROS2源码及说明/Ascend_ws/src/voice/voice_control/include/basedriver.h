#ifndef __BASEDRIVER_H_
#define __BASEDRIVER_H_

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
#include <fcntl.h>          
#include <stdbool.h>
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include <serial/serial.h>
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/float32.hpp"
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

#include <user_interface.h>
#include <string>
#include <locale>
#include <codecvt>
#include <ctime>
#include <joint.h>
#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/int32.hpp>
#include <sys/stat.h>





class BaseDriver : public rclcpp::Node
{
public:
    // 构造函数,有一个参数为节点名称
    BaseDriver(std::string name);

    ~BaseDriver();

    void Control();

private:
    // 声明话题发布者指针
    //rclcpp::Publisher<std_msgs::msg::String>::SharedPtr subscribe_and_publish_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher;

};



#endif
