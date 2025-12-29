#ifndef _MOTOR_CONTROL_H_
#define _MOTOR_CONTROL_H_


#include "include.h"

#define SERVO1 PCout(10)            //控制爪子的开合
#define SERVO2 PCout(11)            //上下爪子的舵机
#define SERVO3 PCout(12)
#define SERVO4 PDout(2)
#define SERVO5 PBout(5)
#define SERVO6 PBout(8)


extern uint16 ServoPwmDutySet[];//脉冲宽度
extern uint16 ServoPwmDuty[];

extern bool ServoPwmDutyHaveChange;//脉宽变化标志位

void ServoSetPluseAndTime_Motor_Control(uint8 id, uint16 p, uint16 time);//时间控制
void ServoPwmDutyCompare_Motor_Control(void);//脉宽变化比较及速度控制
void Init_Motor_Control(void);//初始化PWM
void InitTimer3_Motor_Control(void);
void Calculate_TargetPosition_Motor_Control(float pose[6]);
void Goto_TargetPose_Motor_Control(float pose[3]);
void _Pluse2Angle(float angle[6]);//脉冲转换成角度
void Ctrl_Joint_Motor_Control(U8 id, U8 dir); //控制舵机
void Ctrl_Gripper_Motor_Control(U8 dir); //控制夹爪
void Ctrl_Zeroing_Motor_Control(void);   //控制机械臂回零
void Grap_Preparation_Motor_Control(void); //抓取准备


#endif

