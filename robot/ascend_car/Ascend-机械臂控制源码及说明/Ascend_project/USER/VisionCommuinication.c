//
// Created by lenovo on 2023/7/5.
//
#include "include.h"

#define LED        PBout(9)
#define LED_ON 0

#define CMD_GET_TARGET_POSE            0x00    //请求获取末端位位姿
#define CMD_GET_ALL_JOINT              0x01    //请求获取关节角度
#define CMD_SET_TARGET_POSE            0x02    //发送目标物体位姿
#define CMD_CONTROL_ROBOT              0x03    //请求控制机械臂
#define CMD_CONTROL_GRIPPER            0x04    //请求控制夹爪
#define CMD_CONTROL_ZEROING            0x05    //控制机械臂回零
#define CMD_GRAB_PREPARATION           0x06    //抓取准备
 

static bool fUartRxComplete = FALSE;
static uint8 UartRxBuffer[260];
uint8 Uart1RxBuffer[260];

uint8 frameIndexSumSum[256];


static void Init_LED(void) {
    
    GPIO_InitTypeDef GPIO_InitStructure;

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE); //使能PA端口时钟
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;  //LED0-->PC.2 端口配置
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP; //推挽输出
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOB, &GPIO_InitStructure);

}


void Init_Vision_Comm(void){  //视觉串口通信初始化——USART1
    
    NVIC_InitTypeDef NVIC_InitStructure;

    GPIO_InitTypeDef GPIO_InitStructure;
    USART_InitTypeDef USART_InitStructure;
    
    //NVIC_InitTypeDef NVIC_InitStructure;
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1 | RCC_APB2Periph_GPIOA | RCC_APB2Periph_AFIO, ENABLE);
    
    //USART1_TX   PA.9
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    //USART1_RX	  PA.10
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    //USART 初始化设置
    USART_InitStructure.USART_BaudRate = 9600;  //一般设置为9600;
    USART_InitStructure.USART_WordLength = USART_WordLength_8b;
    USART_InitStructure.USART_StopBits = USART_StopBits_1;
    USART_InitStructure.USART_Parity = USART_Parity_No;
    USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;

    USART_Init(USART1, &USART_InitStructure);

    USART_ITConfig(USART1, USART_IT_RXNE, ENABLE); //开启中断

    USART_Cmd(USART1, ENABLE); //使能串口


    NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;        
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;  //IRQ通道使能
    NVIC_Init(&NVIC_InitStructure);  //根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器USART1
		
    Init_LED();
    LED = LED_ON;		
}

void Uart1SendData(BYTE dat){ //通过发送数据
    
    while ((USART1->SR & 0X40) == 0);//循环发送,直到发送完毕
    USART1->DR = (u8) dat;
    while ((USART1->SR & 0X40) == 0);//循环发送,直到发送完毕
}

void UART1SendDataPacket(uint8 dat[], uint8 count){ //发送数据包  
    
    uint32 i;
    for (i = 0; i < count; i++) {
        while ((USART1->SR & 0X40) == 0);//循环发送,直到发送完毕
        USART1->DR = dat[i];
        while ((USART1->SR & 0X40) == 0);//循环发送,直到发送完毕
    }
}

void USART1_IRQHandler(void){ //中断处理函数
    
    uint8 i;
    uint8 rxBuf;
    static uint8 startCodeSum = 0;
    static bool fFrameStart = FALSE;
    static uint8 messageLength = 0;
    static uint8 messageLengthSum = 1;
																			
    if (USART_GetITStatus(USART1, USART_IT_RXNE) == RESET) {
        return;
    }

    rxBuf = USART_ReceiveData(USART1); //读取接收到的数据
    if (!fFrameStart) {
        if (rxBuf == 0x55){ //读取帧头0x55
            startCodeSum++;
            if (startCodeSum == 2){
                startCodeSum = 0;
                fFrameStart = TRUE;
                messageLength = 1;				
            }
        } else {
            fFrameStart = FALSE;
            messageLength = 0;
            startCodeSum = 0;
        }
    }
    
    if (fFrameStart){       
        Uart1RxBuffer[messageLength] = rxBuf;
        if (messageLength == 2) {
            messageLengthSum = Uart1RxBuffer[messageLength];
            if (messageLengthSum < 1){                 
                messageLengthSum = 1;
                fFrameStart = FALSE;
            }
        }
        messageLength++;

        if (messageLength == messageLengthSum + 3){  //读取完成      		  
            if (fUartRxComplete == FALSE) {			
                for (i = 0; i < messageLength; i++){                   
                    UartRxBuffer[i] = Uart1RxBuffer[i];   //将收到数据存入数组
                    fUartRxComplete = TRUE;  
                }               
            }
            fFrameStart = FALSE;
        }
    }
}

void Send_TargetPosition_Vision_Comm(float pos[6]){   //发送位姿信息    
    
    U8 buf[30];
	uint8 i;
	uint16 checksum = 0;  //数据校验
    buf[0] = 0x55;
    buf[1] = 0x55;
    buf[2] = 27;   //数据长度
    buf[3] = CMD_GET_TARGET_POSE;  //控制命令

    for (i = 0; i < 6; i++){
        unsigned int tmp = *((unsigned int *)(&pos[i]));
        buf[i*4+4] = GET_U32_BYTE(tmp, 0);  //将位姿信息进行字节拆分
        buf[i*4+5] = GET_U32_BYTE(tmp, 1); 
        buf[i*4+6] = GET_U32_BYTE(tmp, 2); 
        buf[i*4+7] = GET_U32_BYTE(tmp, 3);
    }
    for (i = 0; i < 26; i++){
        checksum += buf[2 + i];
    }

    buf[28] = GET_U16_BYTE(checksum, 0);
    buf[29] = GET_U16_BYTE(checksum, 1);

    UART1SendDataPacket(buf, buf[2] + 3); //调用发送数据包函数，将数据发送通过串口发送

}

void Send_Angle_Vision_Comm(float joint[6]){ //发送角度信息   
    
    U8 buf[30];
	uint8 i;
	uint16 checksum = 0; //数据校验
    buf[0] = 0x55;
    buf[1] = 0x55;
    buf[2] = 27; //数据长度
    buf[3] = CMD_GET_ALL_JOINT; //控制命令
    
    for (i = 0; i < 6; i++){
        unsigned int tmp = *((unsigned int *)(&joint[i]));
        buf[i*4+4] = GET_U32_BYTE(tmp, 0);   //将关节角度信息进行字节拆分
        buf[i*4+5] = GET_U32_BYTE(tmp, 1);
        buf[i*4+6] = GET_U32_BYTE(tmp, 2);
        buf[i*4+7] = GET_U32_BYTE(tmp, 3);
    }

    for (i = 0; i < 26; i++){
        checksum += buf[2 + i];
    }

    buf[28] = GET_U16_BYTE(checksum, 0);
    buf[29] = GET_U16_BYTE(checksum, 1);

    UART1SendDataPacket(buf, buf[2] + 3); //调用发送数据包函数，将数据发送通过串口发送
}

void MsgHandle_Vision_Comm(void){  //视觉消息处理

    uint8 cmd;
    float Angle[6];
    float pose[6];
    if (!fUartRxComplete){
        return;
    }
    
    cmd = UartRxBuffer[3];  //控制命令
    switch (cmd){
        case CMD_GET_TARGET_POSE: //请求获取末端位姿
        {
                     
            Calculate_TargetPosition_Motor_Control(pose);
            Send_TargetPosition_Vision_Comm(pose);
            break;
        }
        
        case CMD_GET_ALL_JOINT:  //请求获取机械臂角度
        {    
            
            _Pluse2Angle(Angle);            //脉冲转换成角度
            Send_Angle_Vision_Comm(Angle);  //接收机械臂关节角度
            break;
        }      

        case CMD_SET_TARGET_POSE:  //发送目标物体位姿
        {
            uint32 tmp = 0;
            
            tmp = SET_U32_BYTE(tmp, 0, UartRxBuffer[4]);
            tmp = SET_U32_BYTE(tmp, 1, UartRxBuffer[5]);
            tmp = SET_U32_BYTE(tmp, 2, UartRxBuffer[6]);
            tmp = SET_U32_BYTE(tmp, 3, UartRxBuffer[7]);
            pose[0] = *((float *)(&tmp));
            
            tmp = 0;
            tmp = SET_U32_BYTE(tmp, 0, UartRxBuffer[8]);
            tmp = SET_U32_BYTE(tmp, 1, UartRxBuffer[9]);
            tmp = SET_U32_BYTE(tmp, 2, UartRxBuffer[10]);
            tmp = SET_U32_BYTE(tmp, 3, UartRxBuffer[11]);
            pose[1] = *((float *)(&tmp));
            
            tmp = 0;
            tmp = SET_U32_BYTE(tmp, 0, UartRxBuffer[12]);
            tmp = SET_U32_BYTE(tmp, 1, UartRxBuffer[13]);
            tmp = SET_U32_BYTE(tmp, 2, UartRxBuffer[14]);
            tmp = SET_U32_BYTE(tmp, 3, UartRxBuffer[15]);
            pose[2] = *((float *)(&tmp));
            
//            pose[3] = -150.0000F;
//            pose[4] = 20.0000F;
//            pose[5] = 90.0000F;
            pose[3] =  -178.65F;
            pose[4] =  0.0000F;
            pose[5] =  90.0000F;
            Goto_TargetPose_Motor_Control(pose);//舵机控制
            break;
        } 

        case CMD_CONTROL_ROBOT:   //请求控制舵机
        {
            Ctrl_Joint_Motor_Control(UartRxBuffer[5], UartRxBuffer[4]);
            break;
        }

        case CMD_CONTROL_GRIPPER: //请求控制夹爪
        {
                
            Ctrl_Gripper_Motor_Control(UartRxBuffer[4]);
            break;
        }       
        
        case CMD_CONTROL_ZEROING: //控制机械臂回零
        { 
             
            Ctrl_Zeroing_Motor_Control();
            break;
        }
        
        case CMD_GRAB_PREPARATION: //抓取准备
        {
             
            Grap_Preparation_Motor_Control();
            break;
        }
        default:
            break;
    }
    fUartRxComplete = FALSE;
}


void TaskRun_Vision_Comm(void){
    
    if (IsTimeStep20msDone()){
        ServoPwmDutyCompare_Motor_Control();  //调用速度控制函数
    }

    if (IsTimeStep1000msDone()){
        LED = ~LED; //LED灯闪灭
    }

    MsgHandle_Vision_Comm();  //消息处理
}

