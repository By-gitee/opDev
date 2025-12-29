#ifndef __ArmScene_H_
#define __ArmScene_H_

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
#include <pthread.h>
#include <fcntl.h>          
#include <stdbool.h>
#include <thread>
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
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
#include "std_msgs/msg/int8.hpp"


using namespace std;
//using std::placeholders::_1;


class ArmScene : public rclcpp::Node
{
public:
    // 构造函数,有一个参数为节点名称
    ArmScene(std::string name);

    ~ArmScene();
    void Control(float* data);
    void UpDate();

private:
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_publisher;
    rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr cmd_publisher;
    rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr input_publisher;
    rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr gripper_publisher;
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr input_Sub;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr goal_Sub;

    geometry_msgs::msg::Pose  arm_pose_;
    std_msgs::msg::Int8 cmd_data_;
    std_msgs::msg::Int8 gripper_data_;

    void Input_Callback(const std_msgs::msg::Int8 &in);     
    void goal_Callback(const geometry_msgs::msg::Pose &goal); 
    void robotPickPose(); 
    void openGripper();
    void closeGripper(); 
    void robotPlace();
    void robotMove2PrePose(float* data);
    void robotMove2Pose(float* data);

    float x_scale, y_scale, z_scale,x_pre,y_pre,z_pre;       //correction parameters //修正参数
    bool visionIsConnected,inputResponding;
    float goalData[3]={0,0,0};

};



#endif
