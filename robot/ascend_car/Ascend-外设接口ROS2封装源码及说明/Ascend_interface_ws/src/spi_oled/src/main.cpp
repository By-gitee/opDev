#include "main.hpp"


oled_demo::oled_demo()
:Node("oled")
{
	oledSub_ = this->create_subscription<self_msg::msg::Oled>("oled_show", 10, std::bind(&oled_demo::oled_callback, this, _1));
}

void oled_demo::oled_init()
{
	const char *str = "VSTC china";
	gpio_init("82");	//res
	gpio_init("38");	//dc
	gpio_init("79");	//scl
	gpio_init("80");	//sda

	gpio_set_value(82, "0");	//reset oled
	usleep(100*1000);
	gpio_set_value(82, "1");

	// spi init with: 100Khz, mode 2
	SPISetupMode(500000, 0, 8);
	
	OLED_Init();
	
	OLED_DrawPoint(0, 0, 1);
	OLED_ShowString(20,20,str);
	OLED_ShowNumber(30,50,88,6,12);
	OLED_Refresh_Gram();
	return;
}

void oled_demo::oled_callback(const self_msg::msg::Oled::SharedPtr msg) const
{
	auto data = msg;
	const char *show = msg->show.c_str();
	if(msg->type == 0)	//show string type
	{
		OLED_ShowString(data->x, data->y, show);
	}
	else if(msg->type == 1)	//show point
	{
		OLED_DrawPoint(data->x, data->y, 1);
		
	}
	else if(msg->type == 2)	//show num
	{
		OLED_ShowNumber(30,50,data->num,6,12);
	}
	else			//show all message in fix pos
	{
		OLED_DrawPoint(0, 0, 1);
		OLED_ShowString(20, 20, show);
		OLED_ShowNumber(30, 50, data->num, 6, 12);
	}

	OLED_Refresh_Gram();
	return;
}

int main(int argc, char **argv)
{
	rclcpp::init(argc, argv);
	auto node_ = std::make_shared<oled_demo>();
	node_->oled_init();
	rclcpp::spin(node_);
	rclcpp::shutdown();
}

