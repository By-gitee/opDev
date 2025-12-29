#include "armdriver.h"
#include <signal.h>
#include <cmath>
#include <string>

sensor_msgs::msg::Imu Mpu6050;
bool need_exit = false;

void ExitHandler(int sig)
{
    (void)sig;
    need_exit = true;
}

ArmDriver::ArmDriver(std::string name) : Node(name)
{
  
  memset(&Send_Pose_Data, 0, sizeof(Send_Pose_Data));
  joint_state.position = { 0, 0 , 0, 0, 0, 0};
  f = 0;
  this->declare_parameter<std::string>("usart_port_name","serial");
  this->declare_parameter<int>("serial_baud_rate",115200);

  this->get_parameter_or<std::string>("usart_port_name",  usart_port_name,  "/dev/serial"); //Fixed serial port number //固定串口号
  this->get_parameter_or<int>        ("serial_baud_rate", serial_baud_rate, 115200); //Communicate baud rate 115200 to the lower machine //和下位机通信波特率115200

  joint_publisher = this->create_publisher<sensor_msgs::msg::JointState>("/robot_driver/joint_states", 10);
  endpose_publisher = this->create_publisher<geometry_msgs::msg::Pose>("/robot_driver/tool_point", 10);
  Pose_Sub = this->create_subscription<geometry_msgs::msg::Pose>("arm_pose", 10, std::bind(&ArmDriver::Pose_Callback, this , std::placeholders::_1));
  Cmd_Sub = this->create_subscription<std_msgs::msg::Int8>("cmd", 10, std::bind(&ArmDriver::cmd_Callback, this , std::placeholders::_1));
  Gripper_Sub = this->create_subscription<std_msgs::msg::Int8>("gripper", 10, std::bind(&ArmDriver::Gripper_Callback, this , std::placeholders::_1));

  timer_ = this->create_wall_timer(1000ms, std::bind(&ArmDriver::timer_callback, this));

  try
  { 
    //Attempts to initialize and open the serial port //尝试初始化与开启串口
    Stm32_Serial.setPort(usart_port_name); //Select the serial port number to enable //选择要开启的串口号
    Stm32_Serial.setBaudrate(serial_baud_rate); //Set the baud rate //设置波特率
    serial::Timeout _time = serial::Timeout::simpleTimeout(2000); //Timeout //超时等待
    Stm32_Serial.setTimeout(_time);
    Stm32_Serial.open(); //Open the serial port //开启串口
  }
  catch (serial::IOException& e)
  {
    RCLCPP_ERROR(this->get_logger(),"robot can not open serial port,Please check the serial port cable! "); //If opening the serial port fails, an error message is printed //如果开启串口失败，打印错误信息
  }
  if(Stm32_Serial.isOpen())
  {
    RCLCPP_INFO(this->get_logger(),"robot serial port opened"); //Serial port opened successfully //串口开启成功提示
  }


}

void ArmDriver::timer_callback()
{
  std::string cmd;
  if(f == 0){
  cmd.push_back(0x55);
  cmd.push_back(0x55);
  cmd.push_back(0x01);
  cmd.push_back(0x00);
  f = 1;
  }else{
    cmd.push_back(0x55);
  cmd.push_back(0x55);
  cmd.push_back(0x01);
  cmd.push_back(0x01);
  f = 0;
  }
  //cmd.push_back(0x00);
  //cmd.push_back(0x04); 
  //RCLCPP_INFO(this->get_logger(),"timer_callback");
  try
  {
    Stm32_Serial.write(cmd); //Sends data to the downloader via serial port //通过串口向下位机发送数据 
  }
  catch (serial::IOException& e)   
  {
    RCLCPP_ERROR(this->get_logger(),"Unable to send data through serial port"); //If sending data fails, an error message is printed //如果发送数据失败，打印错误信息
  }

}

ArmDriver::~ArmDriver()
{
  Stm32_Serial.close(); //Close the serial port //关闭串口  
  RCLCPP_INFO(this->get_logger(),"Shutting down"); //Prompt message //提示信息
}

void ArmDriver::Gripper_Callback(const std_msgs::msg::Int8 &direct)
{
  if(direct.data == 0)
  {
    RCLCPP_INFO(this->get_logger(),"direct 0 闭合");
    std::string cmd;
    cmd.push_back(0x55);
    cmd.push_back(0x55);
    cmd.push_back(0x04);
    cmd.push_back(0x04);
    cmd.push_back(0x00);
    cmd.push_back(0x00);
    cmd.push_back(0x08); 
    try
    {
      Stm32_Serial.write(cmd); //Sends data to the downloader via serial port //通过串口向下位机发送数据 
    }
    catch (serial::IOException& e)   
    {
      RCLCPP_ERROR(this->get_logger(),"Unable to send data through serial port"); //If sending data fails, an error message is printed //如果发送数据失败，打印错误信息
    }
  }else if (direct.data == 1)
  {
    RCLCPP_INFO(this->get_logger(),"direct 1 张开");
    std::string cmd;
    cmd.push_back(0x55);
    cmd.push_back(0x55);
    cmd.push_back(0x04);
    cmd.push_back(0x04);
    cmd.push_back(0x01);
    cmd.push_back(0x00);
    cmd.push_back(0x09); 
    try
    {
      Stm32_Serial.write(cmd); //Sends data to the downloader via serial port //通过串口向下位机发送数据 
    }
    catch (serial::IOException& e)   
    {
      RCLCPP_ERROR(this->get_logger(),"Unable to send data through serial port"); //If sending data fails, an error message is printed //如果发送数据失败，打印错误信息
    }
  }
}

void ArmDriver::cmd_Callback(const std_msgs::msg::Int8 &cmd)
{
  if(cmd.data == 0)//回零
  {
    RCLCPP_INFO(this->get_logger(),"cmd 0");
    std::string cmd_;
    cmd_.push_back(0x55);
    cmd_.push_back(0x55);
    cmd_.push_back(0x03);
    cmd_.push_back(0x05);
    cmd_.push_back(0x00);
    cmd_.push_back(0x08); 
    //RCLCPP_INFO(this->get_logger(),"timer_callback");
    try
    {
      Stm32_Serial.write(cmd_); //Sends data to the downloader via serial port //通过串口向下位机发送数据 
    }
    catch (serial::IOException& e)   
    {
      RCLCPP_ERROR(this->get_logger(),"Unable to send data through serial port"); //If sending data fails, an error message is printed //如果发送数据失败，打印错误信息
    }
  }else if (cmd.data == 1)//抓取
  {
    RCLCPP_INFO(this->get_logger(),"cmd 1");
    std::string cmd_;
    cmd_.push_back(0x55);
    cmd_.push_back(0x55);
    cmd_.push_back(0x03);
    cmd_.push_back(0x06);
    cmd_.push_back(0x00);
    cmd_.push_back(0x09); 
    //RCLCPP_INFO(this->get_logger(),"timer_callback");
    try
    {
      Stm32_Serial.write(cmd_); //Sends data to the downloader via serial port //通过串口向下位机发送数据 
    }
    catch (serial::IOException& e)   
    {
      RCLCPP_ERROR(this->get_logger(),"Unable to send data through serial port"); //If sending data fails, an error message is printed //如果发送数据失败，打印错误信息
    }
  }
  

}

void ArmDriver::Pose_Callback(const geometry_msgs::msg::Pose &pose)
{
  //RCLCPP_INFO(this->get_logger(),"Pose_Callback");

  //short  transition;
  Send_Pose_Data.tx[0] = POSE_FRAME_HEADER;
  Send_Pose_Data.tx[1] = POSE_FRAME_HEADER;
  Send_Pose_Data.tx[2] = 0x0E;
  Send_Pose_Data.tx[3] = 0x02;
  /*
  transition = pose.position.x;
  Send_Pose_Data.tx[4] = transition;
  Send_Pose_Data.tx[5] = transition>>8;
  Send_Pose_Data.tx[6] = transition>>16;
  Send_Pose_Data.tx[7] = transition>>24;
  transition = pose.position.y;
  Send_Pose_Data.tx[8] = transition;
  Send_Pose_Data.tx[9] = transition>>8;
  Send_Pose_Data.tx[10] = transition>>16;
  Send_Pose_Data.tx[11] = transition>>24;
  transition = pose.position.z;
  Send_Pose_Data.tx[12] = transition;
  Send_Pose_Data.tx[13] = transition>>8;
  Send_Pose_Data.tx[14] = transition>>16;
  Send_Pose_Data.tx[15] = transition>>24;

  transition = pose.position.x;
  Send_Pose_Data.tx[7] = transition;
  Send_Pose_Data.tx[6] = transition>>8;
  Send_Pose_Data.tx[5] = transition>>16;
  Send_Pose_Data.tx[4] = transition>>24;
  transition = pose.position.y;
  Send_Pose_Data.tx[11] = transition;
  Send_Pose_Data.tx[10] = transition>>8;
  Send_Pose_Data.tx[9] = transition>>16;
  Send_Pose_Data.tx[8] = transition>>24;
  transition = pose.position.z;
  Send_Pose_Data.tx[15] = transition;
  Send_Pose_Data.tx[14] = transition>>8;
  Send_Pose_Data.tx[13] = transition>>16;
  Send_Pose_Data.tx[12] = transition>>24;
*/

  FourBytesToFloat converter1,converter2,converter3;

  converter1.floatValue = pose.position.x;
  converter2.floatValue = pose.position.y;
  converter3.floatValue = pose.position.z;
  Send_Pose_Data.tx[4] = converter1.intValue;
  Send_Pose_Data.tx[5] = converter1.intValue>>8;
  Send_Pose_Data.tx[6] = converter1.intValue>>16;
  Send_Pose_Data.tx[7] = converter1.intValue>>24;
  Send_Pose_Data.tx[8] = converter2.intValue;
  Send_Pose_Data.tx[9] = converter2.intValue>>8;
  Send_Pose_Data.tx[10] = converter2.intValue>>16;
  Send_Pose_Data.tx[11] = converter2.intValue>>24;
  Send_Pose_Data.tx[12] = converter3.intValue;
  Send_Pose_Data.tx[13] = converter3.intValue>>8;
  Send_Pose_Data.tx[14] = converter3.intValue>>16;
  Send_Pose_Data.tx[15] = converter3.intValue>>24;
  int sum = 0;
  for(int i = 2;i<16;i++)
  {
    sum += Send_Pose_Data.tx[i];
  }

  Send_Pose_Data.tx[16] = sum;
  Send_Pose_Data.tx[17] = sum>>8;
  
  RCLCPP_INFO(this->get_logger(),"%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x",
        Send_Pose_Data.tx[0],Send_Pose_Data.tx[1],Send_Pose_Data.tx[2],Send_Pose_Data.tx[3],Send_Pose_Data.tx[4],Send_Pose_Data.tx[5],Send_Pose_Data.tx[6],Send_Pose_Data.tx[7],
        Send_Pose_Data.tx[8],Send_Pose_Data.tx[9],Send_Pose_Data.tx[10],Send_Pose_Data.tx[11],Send_Pose_Data.tx[12],Send_Pose_Data.tx[13],Send_Pose_Data.tx[14],Send_Pose_Data.tx[15],
        Send_Pose_Data.tx[16],Send_Pose_Data.tx[17]);

  try
  {
    Stm32_Serial.write(Send_Pose_Data.tx,sizeof (Send_Pose_Data.tx)); //Sends data to the downloader via serial port //通过串口向下位机发送数据 
  }
  catch (serial::IOException& e)   
  {
    RCLCPP_ERROR(this->get_logger(),"Unable to send data through serial port"); //If sending data fails, an error message is printed //如果发送数据失败，打印错误信息
  }
}           

void ArmDriver::UpDate()
{

  std::vector<unsigned char> tempBuffer;
  bool bfind1 = false,bfind2 = false;
  int num = 0,cont = 0;
  while(rclcpp::ok())
  {

    try
		{
			tempBuffer.clear();
			num = Stm32_Serial.read(tempBuffer);
			if(num == 0)
			{
				RCLCPP_INFO(this->get_logger(),"bms receive buffer 0");
				continue;
			}
		}
		catch (const serial::SerialException &e)
		{
			continue;
		}
		catch (const serial::IOException &e)
		{
			continue;
		}

    if(!bfind1)
		{
			if(tempBuffer[0] == 0x55)
			{
				bfind1 = true;
				//RCLCPP_INFO(this->get_logger(),"0X55");
			}
		}else if(!bfind2)
    {
      if(tempBuffer[0] == 0x55)
			{
				bfind2 = true;
				//RCLCPP_INFO(this->get_logger(),"0X55");
			}
    }else{
      if(bfind1 && bfind2)
			{
				cont = tempBuffer[0];
        try
					{
						tempBuffer.clear();
						num = Stm32_Serial.read(tempBuffer,cont);
						if(num == 0)
						{
							RCLCPP_INFO(this->get_logger(),"bms receive buffer 0");
              bfind1 = false;
              bfind2 = false;
							continue;
						}
					}
					catch (const serial::SerialException &e)
					{
						RCLCPP_INFO(this->get_logger(),"serial receive buffer err");
						continue;
					}
					catch (const serial::IOException &e)
					{
						RCLCPP_INFO(this->get_logger(),"io receive buffer err");
						continue;
					}

          int cs = cont;
          int css = 0;
          for(int i = 0;i<cont-2;i++)
          {
            cs +=tempBuffer[i];
          }
          css = 0x0000 | (tempBuffer[cont - 1] << 8) | tempBuffer[cont - 2];
          if(cs != css){
            RCLCPP_INFO(this->get_logger(),"校验和错误..%x or %x",cs,css);
            bfind1 = false;
            bfind2 = false;
            continue;
          }
          FourBytesToFloat converter1,converter2,converter3,converter4,converter5,converter6;


          
          converter1.intValue = 0x00000000 | (tempBuffer[1] << 24) | (tempBuffer[2] << 16) | (tempBuffer[3] << 8) | tempBuffer[4];
          converter2.intValue = 0x00000000 | (tempBuffer[8] << 24) | (tempBuffer[7] << 16) | (tempBuffer[6] << 8) | tempBuffer[5];
          converter3.intValue = 0x00000000 | (tempBuffer[12] << 24) | (tempBuffer[11] << 16) | (tempBuffer[10] << 8) | tempBuffer[9];
          converter4.intValue = 0x00000000 | (tempBuffer[16] << 24) | (tempBuffer[15] << 16) | (tempBuffer[14] << 8) | tempBuffer[13];
          converter5.intValue = 0x00000000 | (tempBuffer[20] << 24) | (tempBuffer[19] << 16) | (tempBuffer[18] << 8) | tempBuffer[17];
          converter6.intValue = 0x00000000 | (tempBuffer[24] << 24) | (tempBuffer[23] << 16) | (tempBuffer[22] << 8) | tempBuffer[21];

          float d1 = converter1.floatValue;
          float d2 = converter2.floatValue;
          float d3 = converter3.floatValue;
          float d4 = converter4.floatValue;
          float d5 = converter5.floatValue;
          float d6 = converter6.floatValue;

          if(tempBuffer[0] == 0x00)
          {
            end_pose.position.x = d1;
            end_pose.position.y = d2;
            end_pose.position.z = d3;
            tf2::Quaternion quaternion;
	          quaternion.setRPY(d4, d5, d6);
            end_pose.orientation.x = quaternion.x();
            end_pose.orientation.y = quaternion.y();
            end_pose.orientation.z = quaternion.z();
            end_pose.orientation.w = quaternion.w();
            endpose_publisher->publish(end_pose);
          }else if (tempBuffer[0] == 0x01)
          {
            joint_state.position[0] = d1;
            joint_state.position[1] = d2;
            joint_state.position[2] = d3;
            joint_state.position[3] = d4;
            joint_state.position[4] = d5;
            joint_state.position[5] = d6;
            joint_publisher->publish(joint_state);
          }

          // for(int i = 0;i<cont;i++)
          // {
          //   RCLCPP_INFO(this->get_logger()," %x",tempBuffer[i]);
          // }

          RCLCPP_INFO(this->get_logger(),"data: 55-55-1b-%x-%x%x%x%x-%x%x%x%x-%x%x%x%x-%x%x%x%x-%x%x%x%x-%x%x%x%x-%x-%x",
          tempBuffer[0],tempBuffer[1],tempBuffer[2],tempBuffer[3],tempBuffer[4],tempBuffer[5],tempBuffer[6],tempBuffer[7],
          tempBuffer[8],tempBuffer[9],tempBuffer[10],tempBuffer[11],tempBuffer[12],tempBuffer[13],tempBuffer[14],tempBuffer[15],
          tempBuffer[16],tempBuffer[17],tempBuffer[18],tempBuffer[19],tempBuffer[20],tempBuffer[21],tempBuffer[22],tempBuffer[23],tempBuffer[24],tempBuffer[25],tempBuffer[26]);
          


          bfind1 = false;
          bfind2 = false;
      }
    }
  }
}



void myThreadF1(std::shared_ptr<ArmDriver> node)
{
  node->UpDate();
}

void myThreadF2(std::shared_ptr<ArmDriver> node)
{
  rclcpp::spin(node);
}


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    /*产生一个的节点*/
    //auto node = std::make_shared<ArmDriver>("ArmDriver");
    std::shared_ptr<ArmDriver> node = std::make_shared<ArmDriver>("ArmDriver");
    thread myThread1(myThreadF1,node);
    thread myThread2(myThreadF2,node);
	  //阻塞主线程，当子线程执行完毕再开始执行
	  myThread1.join();
    myThread2.join();
    //node->Control();
    rclcpp::shutdown();
    return 0;
}
