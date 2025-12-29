#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdint.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>
#include <linux/gpio.h>
#include <cstdio>

#include "rclcpp/rclcpp.hpp"
#include <iostream>
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/accel.hpp"
#include "self_msg/msg/oled.hpp"     // CHANGE

using std::placeholders::_1;

extern "C" {
#include "spi.h"
#include "oled.h"
}

class oled_demo : public rclcpp::Node {
	public:
		oled_demo();
		void oled_init();
	private:
		rclcpp::Subscription<self_msg::msg::Oled>::SharedPtr oledSub_;
		void oled_callback(const self_msg::msg::Oled::SharedPtr msg) const; 
};
