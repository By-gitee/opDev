一、ROS源码目录如下

arm:						机械臂驱控程序包	
mobile:					移动机器人驱控程序包
move_navi:					tf树构建和发布程序包
sensor:					传感器驱控程序包	
scene:						场景任务样例程序包
params:					导航相关参数
voice:						语音控制驱控程序包
google_ws:               cartographer功能包


二、相关启动命令如下：
硬件设备驱动程序：
1、底盘驱控程序启动：
ros2 launch base_driver basedriver_launch.py
2、机械臂驱控程序启动：
ros2 launch arm_driver armdriver_launch.py
3、激光雷达启动：
ros2 launch  sllidar_ros2 sllidar_a3_launch.py
4、语音控制驱控程序启动：
ros2 launch  voice_control voice_control_launch.py

场景控制程序：
1、机械臂抓取场景：
①启动机械臂
ros2 launch arm_driver armdriver_launch.py
② 启动抓取程序
ros2 launch arm_scene armscene_launch.py
③ 启动与视觉通信程序
ros2 launch  udp_scene udpscene_launch.py
④ 按先后顺序发送以下相关命令：
机械臂准备位
ros2 topic pub --once /cmd std_msgs/msg/Int8 "data: 1"
视觉识别调整
ros2 topic pub --once /arm_start std_msgs/msg/Int8 "data: 1"
开始抓取
ros2 topic pub --once /input std_msgs/msg/Int8 "data: 1"
⑤ 其他单独控制指令：
夹爪闭合
ros2 topic pub --once /gripper std_msgs/msg/Int8 "data: 0"
夹爪张开
ros2 topic pub --once /gripper std_msgs/msg/Int8 "data: 1"
机械臂回零
ros2 topic pub --once /cmd std_msgs/msg/Int8 "data: 0"

2、建图场景：
①启动底盘控制程序
ros2 launch base_driver basedriver_launch.py
②启动雷达驱动程序
ros2 launch  sllidar_ros2 sllidar_a3_launch.py
③启动建图程序
ros2 launch  cartographer_ros backpack_2d.launch.py
④键盘控制程序
ros2 run teleop_twist_keyboard teleop_twist_keyboard
⑤ 打开rviz2 ,即可建图。


3、导航场景：
⑤启动底盘控制程序
ros2 launch base_driver basedriver_launch.py
⑥启动雷达驱动程序
ros2 launch  sllidar_ros2 sllidar_a3_launch.py
③ 启动定位程序
ros2 launch cartographer_ros backpack_2d_localization.launch.py
⑦启动导航程序
ros2 launch  nav2_bringup navigation_launch.py
⑧启动rviz2
即可在rviz2上通过点击地图，确定目标点，实现导航。

4、语音控制场景：
① 启动底盘控制程序
ros2 launch base_driver basedriver_launch.py
② 语音控制驱控程序启动：
ros2 launch  voice_control voice_control_launch.py
②语音交互功能启动：
ros2 launch  wheeltec_mic_ros2  base.launch.py
此时，通过唤醒词“你好，小薇”唤醒语音，唤醒后即可说出相关控制指令，目前支持“小车向前”，“小车左转”，“小车右转”，“小车后退”，“小车停”五个指令。

