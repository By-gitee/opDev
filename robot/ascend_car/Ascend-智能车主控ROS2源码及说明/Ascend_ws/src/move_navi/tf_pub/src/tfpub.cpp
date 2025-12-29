#include "tfpub.h"
#include <signal.h>


sensor_msgs::msg::Imu Mpu6050;
bool need_exit = false;

void ExitHandler(int sig)
{
    (void)sig;
    need_exit = true;
}

TfPub::TfPub(std::string name) : Node(name), odom_broadcaster(this)
{

  this->declare_parameter<float>("x",0.0);
  this->declare_parameter<float>("y",0.0);
  this->declare_parameter<float>("z",0.0);
  this->declare_parameter<float>("theta",0.0);

  //correction parameters
  this->get_parameter_or<float>("x",    x,    0.0); 
  this->get_parameter_or<float>("y",    y,    0.0); 
  this->get_parameter_or<float>("z",    z,    0.0); 
  this->get_parameter_or<float>("theta",    theta,    0.0);  

  loop_rate_ = 50;
  flag_init = 0;

  odom_pub = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
  pub_basepose = this->create_publisher<geometry_msgs::msg::Pose2D>("basepose", 10);
  laser_sub = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("laserpose", 10,std::bind(&TfPub::workCallback, this , std::placeholders::_1));

  timer_ = this->create_wall_timer(1000ms, std::bind(&TfPub::timer_callback, this));

}

void TfPub::workCallback(const geometry_msgs::msg::PoseWithCovarianceStamped& p_pose)
{
      std::unique_lock<std::shared_timed_mutex> pose_lock(mutex_);
      tf2::Quaternion tf_quaternion(p_pose.pose.pose.orientation.x,p_pose.pose.pose.orientation.y,p_pose.pose.pose.orientation.z,p_pose.pose.pose.orientation.w);
      double yaw = tf_quaternion.getAngle();

      flag_init = 1;
      geometry_msgs::msg::Quaternion odom_quat;
      laser_pose_.x = p_pose.pose.pose.position.x;
      laser_pose_.y = p_pose.pose.pose.position.y;

      laser_pose_.theta = yaw;
      geometry_msgs::msg::PoseStamped odom_to_map;
      
      geometry_msgs::msg::PoseWithCovarianceStamped p;
      p.pose.pose.position.x = p_pose.pose.pose.position.x - cos(yaw+theta)*x + sin(yaw+theta)*y;
      p.pose.pose.position.y = p_pose.pose.pose.position.y - sin(yaw+theta)*x - cos(yaw+theta)*y;


      base_pose_.x = p.pose.pose.position.x;
      base_pose_.y = p.pose.pose.position.y;
      base_pose_.theta = yaw+theta;
      base_pose_.theta = atan2(sin(base_pose_.theta), cos(base_pose_.theta));
      pub_basepose->publish(base_pose_);

      nav_msgs::msg::Odometry odom;
      odom.header.stamp = p_pose.header.stamp;
      odom.header.frame_id = "odom";

      //set the position
      odom.pose.pose.position.x = p.pose.pose.position.x;
      odom.pose.pose.position.y = p.pose.pose.position.y;
      odom.pose.pose.position.z = 0.0;
      //odom_quat = tf::createQuaternionMsgFromYaw(base_pose_.theta);
      tf2::Quaternion quaternion;
      quaternion.setRPY(0.0, 0.0, base_pose_.theta);

      odom_quat.x = quaternion.x();
      odom_quat.y = quaternion.y();
      odom_quat.z = quaternion.z();
      odom_quat.w = quaternion.w();
      odom.pose.pose.orientation = odom_quat;

      //set the velocity
      odom.child_frame_id = "base_footprint";
      odom.twist.twist.linear.x = 0;
      odom.twist.twist.linear.y = 0;
      odom.twist.twist.angular.z = 0;

      //publish the message
      odom_pub->publish(odom);

      geometry_msgs::msg::TransformStamped odom_trans;
      odom_trans.header.stamp = p_pose.header.stamp;
      odom_trans.header.frame_id = "odom";
      odom_trans.child_frame_id = "base_footprint";
      odom_trans.transform.translation.x = p.pose.pose.position.x;
      odom_trans.transform.translation.y = p.pose.pose.position.y;
      odom_trans.transform.translation.z = 0.0;
      odom_trans.transform.rotation = odom_quat;

      //send the transform
      odom_broadcaster.sendTransform(odom_trans);
      pose_lock.unlock();
}

void TfPub::timer_callback()
{
  if(flag_init){
      if(last_laser_pose_.x == laser_pose_.x && last_laser_pose_.y == laser_pose_.y && last_laser_pose_.theta == laser_pose_.theta)
      {
        RCLCPP_INFO(this->get_logger(),"laser_pose NO.");
        //ros::param::set("/laser_pose",0);
      }else if(last_laser_pose_.x != laser_pose_.x || last_laser_pose_.y != laser_pose_.y || last_laser_pose_.theta != laser_pose_.theta)
      {
        RCLCPP_INFO(this->get_logger(),"laser_pose YES");
        //ros::param::set("/laser_pose",1);
      }
    }
    last_laser_pose_ = laser_pose_;

}

TfPub::~TfPub()
{
  
}

          

void TfPub::Control()
{
  rclcpp::Rate rate(loop_rate_);  // 创建一个频率为loop_rate_Hz的Rate对象

  while (rclcpp::ok()) {
    // 在循环内部执行需要以10Hz运行的操作

    rate.sleep();  // 控制循环的速率，使其按照指定的频率运行
  }
  
}

void myThreadF1(std::shared_ptr<TfPub> node)
{
  node->Control();
}

void myThreadF2(std::shared_ptr<TfPub> node)
{
  rclcpp::spin(node);
}



int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    /*产生一个的节点*/
    auto node = std::make_shared<TfPub>("TfPub");
    RCLCPP_INFO(node->get_logger(),"TfPub is started");
    thread myThread1(myThreadF1,node);
    thread myThread2(myThreadF2,node);
	  //阻塞主线程，当子线程执行完毕再开始执行
	  myThread1.join();
    myThread2.join();
    rclcpp::shutdown();
    return 0;
}
