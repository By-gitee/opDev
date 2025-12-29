#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    usart_port_name =  LaunchConfiguration('usart_port_name', default='/dev/arm')
    serial_baud_rate =  LaunchConfiguration('serial_baud_rate', default='9600')
    odom_frame_id =  LaunchConfiguration('odom_frame_id', default='odom_combined')
    robot_frame_id =  LaunchConfiguration('robot_frame_id', default='base_footprint')
    gyro_frame_id =  LaunchConfiguration('gyro_frame_id', default='gyro_link')
    odom_x_scale =  LaunchConfiguration('odom_x_scale', default='1.0')
    odom_y_scale =  LaunchConfiguration('odom_y_scale', default='1.0')
    odom_z_scale_positive =  LaunchConfiguration('odom_z_scale_positive', default='1.0')
    odom_z_scale_negative =  LaunchConfiguration('odom_z_scale_negative', default='1.0')

    return LaunchDescription([

        DeclareLaunchArgument(
            'usart_port_name',
            default_value=usart_port_name,
            description='Specifying channel type of lidar'),

        DeclareLaunchArgument(
            'serial_baud_rate',
            default_value=serial_baud_rate,
            description='Specifying usb port to connected lidar'),

        DeclareLaunchArgument(
            'odom_frame_id',
            default_value=odom_frame_id,
            description='Specifying usb port baudrate to connected lidar'),
        
        DeclareLaunchArgument(
            'robot_frame_id',
            default_value=robot_frame_id,
            description='Specifying frame_id of lidar'),

        DeclareLaunchArgument(
            'gyro_frame_id',
            default_value=gyro_frame_id,
            description='Specifying whether or not to invert scan data'),

        DeclareLaunchArgument(
            'odom_x_scale',
            default_value=odom_x_scale,
            description='Specifying whether or not to enable angle_compensate of scan data'),

        DeclareLaunchArgument(
            'odom_y_scale',
            default_value=odom_y_scale,
            description='Specifying scan mode of lidar'),

        DeclareLaunchArgument(
            'odom_z_scale_positive',
            default_value=odom_z_scale_positive,
            description='Specifying scan mode of lidar'),

        DeclareLaunchArgument(
            'odom_z_scale_negative',
            default_value=odom_z_scale_negative,
            description='Specifying scan mode of lidar'),

        Node(
            package='arm_driver',
            executable='armdriver',
            name='armdriver',
            parameters=[{'usart_port_name':usart_port_name,
                         'serial_baud_rate': serial_baud_rate, 
                         'odom_frame_id': odom_frame_id, 
                         'robot_frame_id': robot_frame_id,
                         'gyro_frame_id': gyro_frame_id, 
                         'odom_x_scale': odom_x_scale, 
                         'odom_y_scale': odom_y_scale,
                         'odom_z_scale_positive': odom_z_scale_positive,
                         'odom_z_scale_negative': odom_z_scale_negative}],
            output='screen'),
            
    ])



