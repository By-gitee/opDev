//
// Created by lenovo on 2023/7/5.
//

#ifndef _VISIONCOMMUNICATION_H
#define _VISIONCOMMUNICATION_H

#include "include.h"


void Init_Vision_Comm(void);        //视觉串口通信初始化
void MsgHandle_Vision_Comm(void);  //视觉消息处理

void Send_TargetPosition_Vision_Comm(float pos[6]);  //向视觉发送数据

void Send_Angle_Vision_Comm(float joint[6]);  //发送机械臂角度信息
void TaskRun_Vision_Comm(void);

void UART1SendDataPacket(uint8 dat[], uint8 count);


#endif 
