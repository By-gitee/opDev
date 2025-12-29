from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    call_recognition = Node(
        package="wheeltec_mic_ros2",
        executable="call_recognition",
        #output='screen',
        parameters=[{"confidence_threshold": 20, 	#离线命令词识别置信度
        	"time_per_order": 10}]					#最大识别录音持续时长(0~10s)
    )
    command_recognition = Node(
        package="wheeltec_mic_ros2",
        executable="command_recognition",
        output='screen'
    )
    node_feedback = Node(
        package="wheeltec_mic_ros2",
        executable="node_feedback",
        output='screen'
    )

    ld = LaunchDescription()

    # ld.add_action(voice_control)
    ld.add_action(call_recognition)
    ld.add_action(command_recognition)
    ld.add_action(node_feedback)
    
    return ld