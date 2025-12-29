//
// Created by lenovo on 2023/7/12.
//

#ifndef HAUWEI_SHENGTENG_PROJECT_TIMER_INTERRUPT_H
#define HAUWEI_SHENGTENG_PROJECT_TIMER_INTERRUPT_H

#include "include.h"


extern u32 gSystemTickCount;
void InitDelay(u8 SYSCLK);
void DelayMs(u16 nms);
void DelayUs(u32 nus);
void Init_Timer2(void);
bool IsTimeStep20msDone(void);
bool IsTimeStep1000msDone(void);


#endif //HAUWEI_SHENGTENG_PROJECT_TIMER_INTERRUPT_H
