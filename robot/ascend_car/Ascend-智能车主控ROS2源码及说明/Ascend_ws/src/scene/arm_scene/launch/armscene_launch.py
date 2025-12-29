#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    x_scale =  LaunchConfiguration('x_scale', default='0.0')
    y_scale =  LaunchConfiguration('y_scale', default='0.0')
    z_scale =  LaunchConfiguration('z_scale', default='0.0')
    x_pre =  LaunchConfiguration('x_pre', default='0.0')
    y_pre =  LaunchConfiguration('y_pre', default='0.0')
    z_pre =  LaunchConfiguration('z_pre', default='0.0')

    return LaunchDescription([

        DeclareLaunchArgument(
            'x_scale',
            default_value=x_scale,
            description='x_scale'),

        DeclareLaunchArgument(
            'y_scale',
            default_value=y_scale,
            description='y_scale'),

        DeclareLaunchArgument(
            'z_scale',
            default_value=z_scale,
            description='z_scale'),

        DeclareLaunchArgument(
            'x_pre',
            default_value=x_pre,
            description='x_pre'),

        DeclareLaunchArgument(
            'y_pre',
            default_value=y_pre,
            description='y_pre'),

        DeclareLaunchArgument(
            'z_pre',
            default_value=z_pre,
            description='z_pre'),

        Node(
            package='arm_scene',
            executable='armscene',
            name='armscene',
            parameters=[{'x_scale': x_scale, 
                         'y_scale': y_scale,
                         'z_scale': z_scale,
                         'x_pre': x_pre,
                         'y_pre': y_pre,
                         'z_pre': z_pre}],
            output='screen'),
            
    ])



