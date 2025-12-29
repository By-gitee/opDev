
#include "include.h"



int main(void) {
    
    SystemInit();              //系统时钟初始化为72M  SYSCLK_FREQ_72MHz
    InitDelay(72);             //延时初始化
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);    //设置NVIC中断分组2:2位抢占优先级，2位响应优先级

    Init_Vision_Comm();        //视觉通讯模块初始化
	Init_Motor_Control();      //舵机控制-PWM初始化
    Init_Timer2();             //用于产生100us的定时中断

    while (1) {
        TaskRun_Vision_Comm();  
    }
}



