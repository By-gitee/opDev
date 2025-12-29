#include "armscene.h"
#include <signal.h>
#include <math.h>


sensor_msgs::msg::Imu Mpu6050;
bool need_exit = false;

void ExitHandler(int sig)
{
    (void)sig;
    need_exit = true;
}

ArmScene::ArmScene(std::string name) : Node(name)
{

  visionIsConnected = false;
  inputResponding = false;

  this->declare_parameter<float>("x_scale",0.0);
  this->declare_parameter<float>("y_scale",0.0);
  this->declare_parameter<float>("z_scale",0.0);
  this->declare_parameter<float>("x_pre",0.0);
  this->declare_parameter<float>("y_pre",0.0);
  this->declare_parameter<float>("z_pre",0.0);

  //correction parameters
  this->get_parameter_or<float>("x_pre",    x_pre,    0.0); 
  this->get_parameter_or<float>("y_pre",    y_pre,    0.0); 
  this->get_parameter_or<float>("z_pre",    z_pre,    0.0); 
  this->get_parameter_or<float>("x_scale",    x_scale,    0.0); 
  this->get_parameter_or<float>("y_scale",    y_scale,    0.0); 
  this->get_parameter_or<float>("z_scale",    z_scale,    0.0); 

  this->declare_parameter<int>("usr_task",0);
 

  pose_publisher = this->create_publisher<geometry_msgs::msg::Pose>("arm_pose", 10);
  cmd_publisher = this->create_publisher<std_msgs::msg::Int8>("cmd", 10);
  gripper_publisher = this->create_publisher<std_msgs::msg::Int8>("gripper", 10);
  input_publisher = this->create_publisher<std_msgs::msg::Int8>("input", 10);
  input_Sub = this->create_subscription<std_msgs::msg::Int8>("input", 10, std::bind(&ArmScene::Input_Callback, this , std::placeholders::_1));
  goal_Sub = this->create_subscription<geometry_msgs::msg::Pose>("vision_target", 10, std::bind(&ArmScene::goal_Callback, this , std::placeholders::_1));

}

void ArmScene::robotPickPose()
{
  std_msgs::msg::Int8 p;
  p.data = 0;
  cmd_publisher->publish(p);
}

void ArmScene::openGripper()//夹爪张开
{
  std_msgs::msg::Int8 p;
  p.data = 1;
  gripper_publisher->publish(p);
  RCLCPP_INFO(this->get_logger(),"openGripper");
}

void ArmScene::closeGripper()//夹爪闭合
{
  std_msgs::msg::Int8 p;
  p.data = 0;
  gripper_publisher->publish(p);
  RCLCPP_INFO(this->get_logger(),"closeGripper");
}

void ArmScene::Input_Callback(const std_msgs::msg::Int8 &msg)
{
  if(msg.data == 0)
  {
    std_msgs::msg::Int8 p;
    p.data = 0;
    cmd_publisher->publish(p);
  }
  
  if(!inputResponding)
  {
    inputResponding = true;
    if(msg.data == 1)
    {
      if(visionIsConnected)
      {
        if(goalData[0] == x_scale && goalData[1] == y_scale && goalData[2] == z_scale)
        {
          RCLCPP_WARN(this->get_logger(),"The target object is not detected !");
        }else{
          Control(goalData);
        }
      }else{
        RCLCPP_WARN(this->get_logger(),"Need to connect the VisionServer first, please try to restart VisionServer !");
      }

    }
    inputResponding = false;
  }else{
    RCLCPP_WARN(this->get_logger(),"Input is responding, please wait until the response is over !");
  }

}

void ArmScene::goal_Callback(const geometry_msgs::msg::Pose &goal)
{
  visionIsConnected = true;
  goalData[0]=goal.position.x * 1000.0 + x_scale;
  goalData[1]=goal.position.y * 1000.0 + y_scale;
  goalData[2]=goal.position.z * 1000.0 + z_scale;
  goalData[0] = round(goalData[0]);// / 10;
  goalData[1] = round(goalData[1]);
  goalData[2] = round(goalData[2]);
RCLCPP_INFO(this->get_logger(),"goal_pose: %f %f %f",goal.position.x,goal.position.y,goal.position.z);
RCLCPP_INFO(this->get_logger(),"goal_pose: %f %f %f",goalData[0],goalData[1],goalData[2]);
}

ArmScene::~ArmScene()
{
  
}

void ArmScene::robotPlace()
{
  RCLCPP_INFO(this->get_logger(),"robotPlace");
  std_msgs::msg::Int8 p;
  p.data = 0;
  cmd_publisher->publish(p);
}

void ArmScene::robotMove2PrePose(float* data)
{
  geometry_msgs::msg::Pose p;
  p.position.x = -156;//data[0] + x_pre;
  p.position.y = -4.8;//data[1] + y_pre;
  p.position.z = 167;//data[2] + z_pre;
  pose_publisher->publish(p);
  RCLCPP_INFO(this->get_logger(),"prepose: %f %f %f",p.position.x,p.position.y,p.position.z);
}

void ArmScene::robotMove2Pose(float* data)
{
  geometry_msgs::msg::Pose p;
  p.position.x = data[0];
  p.position.y = data[1];
  p.position.z = data[2];
  pose_publisher->publish(p);
  RCLCPP_INFO(this->get_logger(),"pose: %f %f %f",p.position.x,p.position.y,p.position.z);
}
          

void ArmScene::Control(float* data)
{
  openGripper();
  rclcpp::sleep_for(std::chrono::seconds(5));
  robotMove2PrePose(data);
  rclcpp::sleep_for(std::chrono::seconds(5));
  robotMove2Pose(data);
  rclcpp::sleep_for(std::chrono::seconds(5));
  closeGripper();
  rclcpp::sleep_for(std::chrono::seconds(5));
  robotPlace();
}

void ArmScene::UpDate()
{
  int tmp = 0;
  std_msgs::msg::Int8 p;
  p.data = 1;
  while (rclcpp::ok()) {
    this->get_parameter("usr_task", tmp);
    if(tmp == 1){
      input_publisher->publish(p);
      this->set_parameter(rclcpp::Parameter("usr_task", 0));
    }
    rclcpp::sleep_for(std::chrono::nanoseconds(100000000));//0.1s
    //RCLCPP_INFO(this->get_logger(),"UpDate usr_task");
  }
}

void myThreadF1(std::shared_ptr<ArmScene> node)
{
  node->UpDate();
}

void myThreadF2(std::shared_ptr<ArmScene> node)
{
  rclcpp::spin(node);
}



int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    /*产生一个的节点*/
    auto node = std::make_shared<ArmScene>("ArmScene");
    RCLCPP_INFO(node->get_logger(),"ArmScene is started");
    thread myThread1(myThreadF1,node);
    thread myThread2(myThreadF2,node);
	  //阻塞主线程，当子线程执行完毕再开始执行
	  myThread1.join();
    myThread2.join();
    rclcpp::shutdown();
    return 0;
}
