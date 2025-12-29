#include "udpscene.h"
#include <signal.h>


sensor_msgs::msg::Imu Mpu6050;
bool need_exit = false;

void ExitHandler(int sig)
{
    (void)sig;
    need_exit = true;
}

UdpScene::UdpScene(std::string name) : Node(name)
{

  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Failed to create socket." << std::endl;
        return ;
    }

  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(10000);
  serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1"); 

  input_publisher = this->create_publisher<std_msgs::msg::Int8>("input", 10);
  cmdvel_publisher = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
  vision_target_publisher = this->create_publisher<geometry_msgs::msg::Pose>("vision_target", 10);
  input_Sub = this->create_subscription<std_msgs::msg::Int8>("arm_start", 10, std::bind(&UdpScene::Input_Callback, this , std::placeholders::_1));

}

void UdpScene::Input_Callback(const std_msgs::msg::Int8 &msg)
{
  if(msg.data == 1)
  {
    const char* message = "start";
    ssize_t bytesSent = sendto(sockfd, message, strlen(message), 0, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    std::cout << "Send data start" << std::endl;
    if (bytesSent < 0) {
        std::cerr << "Failed to send data." << std::endl;
        close(sockfd);
        return ;
    }
  }else if(msg.data == 0)
  {
    const char* message = "stop";
    ssize_t bytesSent = sendto(sockfd, message, strlen(message), 0, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    std::cout << "Send data stop" << std::endl;
    if (bytesSent < 0) {
        std::cerr << "Failed to send data." << std::endl;
        close(sockfd);
        return ;
    }
  }

}


UdpScene::~UdpScene()
{
  
}


void UdpScene::UpDate()
{
  char buffer[1024];
    //struct sockaddr_in serverAddr;
    socklen_t addrLen = sizeof(serverAddr);
    RCLCPP_INFO(this->get_logger(), "start receive data.");
    while (rclcpp::ok()) {
      memset(buffer, 0, 1024);
        ssize_t bytesRead = recvfrom(sockfd, buffer, sizeof(buffer), 0, (struct sockaddr*) &serverAddr, &addrLen);
        if (bytesRead < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to receive data.");
            break;
        }

        std::stringstream ss(buffer);
        std::vector<std::string> words;
        std::string word;

        while (ss >> word) {
            words.push_back(word);
        }

        // for (const auto& w : words) {
        //     std::cout << w << std::endl;
        // }

        if(words[0] == "P")
        {
          geometry_msgs::msg::Pose d;
          d.position.x = std::stof(words[1]);
          d.position.y = std::stof(words[2]);
          d.position.z = std::stof(words[3]);
          vision_target_publisher->publish(d);
          std::cout << "P"  << std::endl;

        }else if(words[0] == "R")
        {
          std::cout << "R" << std::endl;
          geometry_msgs::msg::Twist d;
          d.angular.z = std::stof(words[1]);
          cmdvel_publisher->publish(d);
        }else if(words[0] == "L")
        {
          std::cout << "L" << std::endl;
          geometry_msgs::msg::Twist d;
          d.linear.x = std::stof(words[1]);
          cmdvel_publisher->publish(d);
        }

        RCLCPP_INFO(this->get_logger(), "Received message from server: %s", buffer);
    }
}

void myThreadF1(std::shared_ptr<UdpScene> node)
{
  node->UpDate();
}

void myThreadF2(std::shared_ptr<UdpScene> node)
{
  rclcpp::spin(node);
}



int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    /*产生一个的节点*/
    auto node = std::make_shared<UdpScene>("UdpScene");
    RCLCPP_INFO(node->get_logger(),"UdpScene is started");
    thread myThread1(myThreadF1,node);
    thread myThread2(myThreadF2,node);
	  //阻塞主线程，当子线程执行完毕再开始执行
	  myThread1.join();
    myThread2.join();
    rclcpp::shutdown();
    return 0;
}
