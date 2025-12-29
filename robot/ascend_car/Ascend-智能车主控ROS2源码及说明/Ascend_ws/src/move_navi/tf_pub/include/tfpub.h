#ifndef __TfPub_H_
#define __TfPub_H_

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
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/quaternion.hpp>
#include <cstdio>
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include "std_msgs/msg/int8.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"



using namespace std;
//using std::placeholders::_1;

struct pose_type {
	double x;  // Ä¬ÈÏµ¥Î»Îªm£¨»òm/s£©
	double y;  // Ä¬ÈÏµ¥Î»Îªm£¨»òm/s£©
	double theta; // Ä¬ÈÏµ¥Î»Îªrad£¨»òrad/s£©
};


class TfPub : public rclcpp::Node
{
public:
    // 构造函数,有一个参数为节点名称
    TfPub(std::string name);

    ~TfPub();
    void Control();

private:

    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr laser_sub;
    rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr pub_basepose;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
    geometry_msgs::msg::Pose2D base_pose_;
    rclcpp::TimerBase::SharedPtr timer_;
    pose_type laser_pose_;
    pose_type last_laser_pose_;
 
    void workCallback(const geometry_msgs::msg::PoseWithCovarianceStamped& p);
    void timer_callback();

    float x, y,z,theta;       //correction parameters //修正参数
    std::shared_timed_mutex mutex_;
    int loop_rate_;
    int flag_init;
    geometry_msgs::msg::PoseWithCovarianceStamped last_published_pose;
  
    tf2_ros::TransformBroadcaster odom_broadcaster;

};



#endif
