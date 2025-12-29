//
// Created by lenovo on 2023/7/11.
//

#include "include.h"


u32 gSystemTickCount = 0;

uint16 BatteryVoltage;
static u8  fac_us=0;  //us延时倍乘数
static u16 fac_ms=0;  //ms延时倍乘数

//初始化延迟函数
//SYSTICK的时钟固定为HCLK时钟的1/8
//SYSCLK:系统时钟
void InitDelay(u8 SYSCLK){
    
    SysTick_CLKSourceConfig(SysTick_CLKSource_HCLK_Div8);	//选择外部时钟  HCLK/8
    fac_us=SYSCLK/8;
    fac_ms=(u16)fac_us*1000;
}


//延时nms
//注意nms的范围
//SysTick->LOAD为24位寄存器,所以,最大延时为:
//nms<=0xffffff*8*1000/SYSCLK
//SYSCLK单位为Hz,nms单位为ms
//对72M条件下,nms<=1864
void DelayMs(u16 nms){
    
    u32 temp;
    SysTick->LOAD=(u32)nms*fac_ms;//时间加载(SysTick->LOAD为24bit)
    SysTick->VAL =0x00;   //清空计数器
    SysTick->CTRL=0x01 ;  //开始倒数
    do{
        temp=SysTick->CTRL;
    }
    while(temp&0x01&&!(temp&(1<<16)));//等待时间到达
    SysTick->CTRL=0x00;  //关闭计数器
    SysTick->VAL =0X00;  //清空计数器
}


//延时nus
//nus为要延时的us数.
void DelayUs(u32 nus){
    
    u32 temp;
    SysTick->LOAD=nus*fac_us; //时间加载
    SysTick->VAL=0x00;   //清空计数器
    SysTick->CTRL=0x01 ; //开始倒数
    do{
        temp=SysTick->CTRL;
    }
    while(temp&0x01&&!(temp&(1 << 16)));//等待时间到达
    SysTick->CTRL = 0x00; //关闭计数器
    SysTick->VAL = 0X00;  //清空计数器
}


void Init_Timer2(void){

    NVIC_InitTypeDef NVIC_InitStructure;
    
    TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2, ENABLE); //时钟使能
    TIM_TimeBaseStructure.TIM_Period = (10 - 1);         //设置在下一个更新事件装入活动的自动重装载寄存器周期的值
    TIM_TimeBaseStructure.TIM_Prescaler = (720 - 1);     //设置用来作为TIMx时钟频率除数的预分频值
    TIM_TimeBaseStructure.TIM_ClockDivision = 0;         //设置时钟分割:TDTS = Tck_tim
    TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;  //TIM向上计数模式
    TIM_TimeBaseInit(TIM2, &TIM_TimeBaseStructure);      //根据TIM_TimeBaseInitStruct中指定的参数初始化TIMx的时间基数单位
    TIM_ITConfig(  //使能或者失能指定的TIM中断
            TIM2,  //TIM2
            TIM_IT_Update  |  //TIM 中断源
            TIM_IT_Trigger,   //TIM 触发中断源
            ENABLE //使能
    );
    TIM_Cmd(TIM2, ENABLE);  //使能TIMx外设

    NVIC_InitStructure.NVIC_IRQChannel = TIM2_IRQn;     //TIM2中断
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 2;  //先占优先级0级
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;	//从优先级3级
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;     //IRQ通道被使能
    NVIC_Init(&NVIC_InitStructure);  //根据NVIC_InitStruct中指定的参数初始化外设NVIC寄存器
}


//定时器2中断  100us
void TIM2_IRQHandler(void){   //TIM2中断
    
    static uint32 time = 0;
    if(TIM_GetITStatus(TIM2, TIM_IT_Update) != RESET){  //检查指定的TIM中断发生与否:TIM 中断源
        TIM_ClearITPendingBit(TIM2, TIM_IT_Update);     //清除TIMx的中断待处理位:TIM 中断源
        if(++time >= 10){     //100us延迟10倍就是1ms gSystemTickCount自加一次   
            time = 0;
            gSystemTickCount++;//时间计时        
        }
    }
}


bool IsTimeStep20msDone(void){     //设置20ms中断
    
    static uint32 time = 1;
    static uint32 times = 0;
    if(gSystemTickCount > time){   //time自加10，由于gSystemTickCount 1ms自加一次，所以times 10ms自加一次
        time += 10;
        times++;
        if (times % 2 == 0){
           return TRUE;
        }
    }
    return FALSE;	
}


bool IsTimeStep1000msDone(void){   //设置1000ms中断
    
    static uint32 time = 1;
    static uint32 times = 0;
    if(gSystemTickCount > time){  //time自加10，由于gSystemTickCount 1ms自加一次，所以times 10ms自加一次
        time += 10;
        times++;
        if (times % 100 == 0){    
           return TRUE;
        }
    }
    return FALSE;	
}


