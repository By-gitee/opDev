#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    x =  LaunchConfiguration('x', default='0.0')
    y =  LaunchConfiguration('y', default='0.0')
    z =  LaunchConfiguration('z', default='0.0')
    theta =  LaunchConfiguration('theta', default='0.0')

    return LaunchDescription([

        DeclareLaunchArgument(
            'x',
            default_value=x,
            description='x'),

        DeclareLaunchArgument(
            'y',
            default_value=y,
            description='y'),

        DeclareLaunchArgument(
            'z',
            default_value=z,
            description='z'),

        DeclareLaunchArgument(
            'theta',
            default_value=theta,
            description='theta'),

        Node(
            package='tf_pub',
            executable='tfpub',
            name='tfpub',
            parameters=[{'x': x, 
                         'y': y,
                         'z': z,
                         'theta': theta}],
            output='screen'),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments = ['1', '0', '0', '3.14', '0', '0', 'base_link', 'laser_link']
            ),
            
    ])



