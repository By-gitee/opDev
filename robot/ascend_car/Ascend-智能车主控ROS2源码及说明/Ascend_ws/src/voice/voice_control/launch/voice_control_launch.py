#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():
    package_path = get_package_share_directory('voice_control')

    return LaunchDescription([
        # DeclareLaunchArgument(
        #     'package_path',
        #     default_value=package_path,
        #     description='Path to voice_control package'
        # ),
        # ExecuteProcess(
        #     cmd=['ros2', 'param', 'load', 'find', 'voice_control',os.path.join(package_path, 'config/appid_params.yaml')],
        #     output='screen',
        # ),
        Node(
            package='voice_control',
            executable='voicecontrol',
            name='voicecontrol',
            output='screen',
            parameters=[{'source_path': package_path}]
        )
    ])