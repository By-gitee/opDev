首先确保设备下载了keil5软件（如果没下载可以去官网下载，链接： https://www2.keil.com/mdk5），进入工程目录双击绿色图标“Project”文件打开工程程序

工程文件夹“CORE”存放STM32启动文件，主要用于初始化CPU和启动应用程序；
工程文件夹“FWLib”存放STM32库函数，使用时直接调用即可；
工程文件夹“USER”存放用户编写程序，其中：
“algo.c”文件是调用算法文件，主要调用了正解、逆解；
“motor_control.c”文件存放舵机控制程序，通过PWM对舵机输出高低电平实现机械臂舵机控制；
“Time_interrup.c”文件存放时钟定时器程序，其中包括设置延时、设置中断等；
“VisionCommuinication.c”文件存放通讯接口程序，定义机械臂各个控制命令；
“four_Forward_standardDH.c”文件存放正解算法程序，将角度值转换成位姿坐标；
“SDH_IK_4axis.c”文件存放逆解算法程序，将位姿坐标转换成角度值；
