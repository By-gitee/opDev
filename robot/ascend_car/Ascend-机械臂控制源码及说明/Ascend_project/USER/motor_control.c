#include "include.h"


uint16 ServoPwmDuty[8] = {1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500};     //PWM脉冲宽度
uint16 ServoPwmDutySet[8] = {1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500};  //PWM脉冲宽度
float ServoPwmDutyInc[8];  //为了速度控制，当PWM脉宽发生变化时，每2.5ms或20ms递增的PWM脉宽

bool ServoPwmDutyHaveChange = FALSE; //脉宽有变化标志位

uint16 ServoTime = 2000; //舵机从当前角度运动到指定角度的时间，也就是控制速度


static void _Angle2Pluse(float angle[4], uint16 pulse[6]){  //将角度转换成脉冲值	
    
    uint8 i;
    for (i = 0; i < 4; ++i) {
        if(i == 1){

            angle[i] = -angle[i];
        }
        if(angle[i] >= 270 && angle[i] <= 360){
        
            angle[i] =  angle[i] - 360;
        }
        else if(angle[i] >= -360 && angle[i] <= -270){
        
            angle[i] = 360 + angle[i];
        }
        else if(angle[i] < -360){
            
            angle[i] += 360;
        }
        else if(angle[i] > 360){

            angle[i] -= 360;
        }
        else if(angle[i] > 630 && angle[i] < 720)
        {
            angle[i] = -(90-(angle[i] - 630));
        }
        else if(angle[i] > -720 && angle[i] < -630)
        {
            angle[i]= 90+(angle[i] + 630);
        }
        pulse[i] = ((angle[i]*1500) / 135) + 1500;        
    }
}

void _Pluse2Angle(float angle[4]){ //脉冲转换成角度	
   
    uint8 i;
//	angle[4] = ((ServoPwmDuty[6-i])*135.0)/1500-135;
//	angle[5] = ((ServoPwmDuty[6-i])*135.0)/1500-135;
    
    for (i = 0; i < 4; ++i){
        angle[i] = ((ServoPwmDuty[6-i])*135.0)/1500-135;
        if(i == 1)
        {
            angle[i] = -angle[i];
        }
    }
}

void ServoSetPluseAndTime_Motor_Control(uint8 id, uint16 p, uint16 time){
    
    if(id != 4)
    {
        if (id > 7 || p < 500 || p > 2500){
            return;
        }
    }
    if (time < 20){
        time = 20;
    }    
    if (time > 30000){
        time = 30000; 
    }				
    ServoPwmDutySet[id] = p;
    ServoTime = time;
    ServoPwmDutyHaveChange = TRUE;
}

void ServoPwmDutyCompare_Motor_Control(void){  //脉宽变化比较及速度控制
    
    uint8 i;		
    static uint16 ServoPwmDutyIncTimes;  //需要递增的次数
    static bool ServoRunning = FALSE;    //舵机正在以指定速度运动到指定的脉宽对应的位置
    
    if (ServoPwmDutyHaveChange){         //停止运动并且脉宽发生变化时才进行计算  
        ServoPwmDutyHaveChange = FALSE;
        ServoPwmDutyIncTimes = ServoTime/20; //当每20ms调用一次函数时用此句
        
        for (i = 0; i < 8; i++){
            if (ServoPwmDutySet[i] > ServoPwmDuty[i]){
                ServoPwmDutyInc[i] = ServoPwmDutySet[i] - ServoPwmDuty[i];
                ServoPwmDutyInc[i] = -ServoPwmDutyInc[i];
            } else{
                ServoPwmDutyInc[i] = ServoPwmDuty[i] - ServoPwmDutySet[i];
            }
            ServoPwmDutyInc[i] /= ServoPwmDutyIncTimes;  //每次递增的脉宽——总增长的脉宽除以递增的次数
        }
        ServoRunning = TRUE; //舵机开始动作
    }
    
    if (ServoRunning){			
        ServoPwmDutyIncTimes--;
        for (i = 0; i < 8; i++){
            if (ServoPwmDutyIncTimes == 0){ //最后一次递增就直接将设定值赋给当前值           	  
                ServoPwmDuty[i] = ServoPwmDutySet[i];
                ServoRunning = FALSE;       //到达设定位置，舵机停止运动
            } else{
                ServoPwmDuty[i] = ServoPwmDutySet[i] +
                                  (signed short int) (ServoPwmDutyInc[i] * ServoPwmDutyIncTimes);							
            }
        }
    }
}


void InitTimer3_Motor_Control(void){
    
    NVIC_InitTypeDef NVIC_InitStructure;

    RCC->APB1ENR |= 1 << 1; //TIM3时钟使能
    TIM3->ARR = 10000 - 1;  //设定计数器自动重装值,刚好1ms
    TIM3->PSC = 72 - 1;     //预分频器72,得到1Mhz的计数时钟
    TIM3->DIER |= 1 << 0;   //允许更新中断
//	TIM3->DIER|=1<<6;       //允许触发中断	   
    TIM3->CR1 |= 0x01;      //使能定时器3

    NVIC_InitStructure.NVIC_IRQChannel = TIM3_IRQn;            //TIM3中断
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;  //先占优先级3级
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;         //从优先级3级
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;            //IRQ通道被使能
    NVIC_Init(&NVIC_InitStructure);  //根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器

}


void Init_Motor_Control(void){

    GPIO_InitTypeDef GPIO_InitStructure;

    InitTimer3_Motor_Control();

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_5 | GPIO_Pin_8;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;      //推挽输出
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOB, &GPIO_InitStructure);

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC, ENABLE);
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10 | GPIO_Pin_11 | GPIO_Pin_12;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;      //推挽输出
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOC, &GPIO_InitStructure);

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOD, ENABLE);
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;      //推挽输出
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOD, &GPIO_InitStructure);

}


//将PWM脉宽转化成自动装载寄存器的值
void Timer3_ARRValue_Motor_Control(uint16 pwm){

    TIM3->ARR = pwm + 1;
}


//定时器3中断服务程序	 
void TIM3_IRQHandler(void){

    static uint16 i = 1;
    if (TIM3->SR & 0x0001){ //溢出中断  
        switch (i){
            case 1:  //由于机械臂关节由1~6所以此处忽略
// 				SERVO0 = 1;	//PWM控制脚高电平
                //给定时器0赋值，计数Pwm0Duty个脉冲后产生中断，下次中断会进入下一个case语句
                Timer3_ARRValue_Motor_Control(ServoPwmDuty[0]);
                break;
            case 2:
// 				SERVO0 = 0;	   //PWM控制脚低电平
                //此计数器赋值产生的中断表示下一个单元要进行任务的开始
                Timer3_ARRValue_Motor_Control(2500 - ServoPwmDuty[0]);
                break;
            case 3:
                SERVO1 = 1;
                Timer3_ARRValue_Motor_Control(ServoPwmDuty[1]);
                break;
            case 4:
                SERVO1 = 0;    //PWM控制脚低电平
                Timer3_ARRValue_Motor_Control(2500 - ServoPwmDuty[1]);
                break;
            case 5:
                SERVO2 = 1;
                Timer3_ARRValue_Motor_Control(ServoPwmDuty[2]);
                break;
            case 6:
                SERVO2 = 0;    //PWM控制脚低电平
                Timer3_ARRValue_Motor_Control(2500 - ServoPwmDuty[2]);
                break;
            case 7:
                SERVO3 = 1;
                Timer3_ARRValue_Motor_Control(ServoPwmDuty[3]);
                break;
            case 8:
                SERVO3 = 0;    //PWM控制脚低电平
                Timer3_ARRValue_Motor_Control(2500 - ServoPwmDuty[3]);
                break;
            case 9:
                SERVO4 = 1;
                Timer3_ARRValue_Motor_Control(ServoPwmDuty[4]);
                break;
            case 10:
                SERVO4 = 0;    //PWM控制脚低电平
                Timer3_ARRValue_Motor_Control(2500 - ServoPwmDuty[4]);
                break;
            case 11:
                SERVO5 = 1;
                Timer3_ARRValue_Motor_Control(ServoPwmDuty[5]);
                break;
            case 12:
                SERVO5 = 0;    //PWM控制脚低电平
                Timer3_ARRValue_Motor_Control(2500 - ServoPwmDuty[5]);
                break;
            case 13:
                SERVO6 = 1;
                Timer3_ARRValue_Motor_Control(ServoPwmDuty[6]);
                break;
            case 14:
                SERVO6 = 0;    //PWM控制脚低电平
                Timer3_ARRValue_Motor_Control(2500 - ServoPwmDuty[6]);
                break;
            case 15:           //忽略第七关节
// 				SERVO7 = 1;	
                Timer3_ARRValue_Motor_Control(ServoPwmDuty[7]);
                break;
            case 16:
// 				SERVO7 = 0;	   //PWM控制脚低电平
                Timer3_ARRValue_Motor_Control(2500 - ServoPwmDuty[7]);
                i = 0;
                break;
        }
        i++;
    }
    TIM3->SR &= ~(1 << 0); //清除中断标志位
}


void Goto_TargetPose_Motor_Control(float pose[6]){ //位姿转换
    
    uint16 pulse[4];
    float angle[4] = {0};
    float Pose[6] = {0};
	uint8 i;
    Pose[0] = pose[0];
    Pose[1] = pose[1];
    Pose[2] = pose[2];
    Pose[3] = pose[3];
    Pose[4] = pose[4];
    Pose[5] = pose[5];
    AlgoInverseKinematics(Pose,angle);  //逆解
    _Angle2Pluse(angle, pulse); 
//    for(i = 6;i > 2;i--)
//    {
//        if(pulse[i] > 2600 || pulse[i] < 350)
//        {
//            return;
//        }
//    }
    for (i = 6; i > 2; i--) {
        if(i == 4)
        {
            pulse[6-i] -= 85;
        }
        ServoSetPluseAndTime_Motor_Control(i, pulse[6-i],2000);
    }
}

void Calculate_TargetPosition_Motor_Control(float pose[6]){  //发送位姿信息
    
    float Angle[4];
    float T04[16] = {0};
    _Pluse2Angle(Angle); 
    AlgoPositiveSolution(Angle,T04,pose);
}

void Ctrl_Joint_Motor_Control(U8 id, U8 dir){  //机械臂关节控制
	
    int pwm_idx = 0;
	if (id < 1 || id > 6){
		return;
	}

	pwm_idx = 7 - id;
    if(id != 3)
    {
        if (dir == 0x01){
            if(ServoPwmDutySet[pwm_idx] < 500){  //限位最小500脉冲值
                ServoPwmDutySet[pwm_idx] = 500;
                return;
            }
            ServoSetPluseAndTime_Motor_Control( pwm_idx, ServoPwmDutySet[pwm_idx] - 20, 20);
        } else if(dir == 0x00){        
            if(ServoPwmDutySet[pwm_idx] > 2500){ //限位最大2500脉冲值
                ServoPwmDutySet[pwm_idx] = 2500;
                return;
            }
            ServoSetPluseAndTime_Motor_Control( pwm_idx, ServoPwmDutySet[pwm_idx] + 20, 20);
        }		
    }
    if(id == 3)
    {
        if (dir == 0x01){
            if(ServoPwmDutySet[pwm_idx] < 350){  //限位最小350脉冲值
                ServoPwmDutySet[pwm_idx] = 350;
                return;
            }
            ServoSetPluseAndTime_Motor_Control( pwm_idx, ServoPwmDutySet[pwm_idx] - 20, 20);
        } else if(dir == 0x00){
            if(ServoPwmDutySet[pwm_idx] > 2600){ //限位最大3000脉冲值
                ServoPwmDutySet[pwm_idx] = 2600;
                return;
            }
            ServoSetPluseAndTime_Motor_Control( pwm_idx, ServoPwmDutySet[pwm_idx] + 20, 20);
        }		
    }
}

void Ctrl_Gripper_Motor_Control(U8 dir){ //机械臂夹爪控制
    
    if (dir == 0x01){        
		ServoSetPluseAndTime_Motor_Control( 1, 600,1500);        
	} else if(dir == 0x00){
		ServoSetPluseAndTime_Motor_Control( 1, 1500, 1500);
    }
}

void Ctrl_Zeroing_Motor_Control(void){   //控制机械臂回零
    
    ServoSetPluseAndTime_Motor_Control( 1, 1500, 1500);
    ServoSetPluseAndTime_Motor_Control( 2, 1500, 1500);
    ServoSetPluseAndTime_Motor_Control( 3, 1500, 1500);
    ServoSetPluseAndTime_Motor_Control( 4, 1500, 1500);
    ServoSetPluseAndTime_Motor_Control( 5, 1500, 1500);
    ServoSetPluseAndTime_Motor_Control( 6, 1500, 1500);
}

void Grap_Preparation_Motor_Control(void){ //抓取准备
    
    ServoSetPluseAndTime_Motor_Control( 1, 600, 1500);
    ServoSetPluseAndTime_Motor_Control( 2, 1500, 1500);
    ServoSetPluseAndTime_Motor_Control( 3, 2060, 1500);
    ServoSetPluseAndTime_Motor_Control( 4, 2667, 1500);
    ServoSetPluseAndTime_Motor_Control( 5, 2212, 1500);
    ServoSetPluseAndTime_Motor_Control( 6, 1500, 1500);
}

