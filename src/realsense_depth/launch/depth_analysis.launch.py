"""Bring up the RealSense driver plus both analysis nodes.

The driver arguments matter:
  * 848x480 is the D435's native depth resolution. Other sizes are rescaled
    internally and lose accuracy.
  * align_depth puts depth on the colour pixel grid, which is what makes
    "depth at pixel (u, v) of the colour image" a meaningful question.
  * pointcloud gives cloud_pipeline its input.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('realsense_depth')
    params = os.path.join(share, 'config', 'params.yaml')

    launch_camera = LaunchConfiguration('launch_camera')
    viewer = LaunchConfiguration('viewer')
    yolo = LaunchConfiguration('yolo')

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('realsense2_camera'),
            '/launch/rs_launch.py']),
        condition=IfCondition(launch_camera),
        launch_arguments={
            'depth_module.depth_profile': '848x480x30',
            'rgb_camera.color_profile': '848x480x30',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'true',
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('launch_camera', default_value='true'),
        DeclareLaunchArgument('viewer', default_value='false',
                              description='open the OpenCV probe window'),
        DeclareLaunchArgument('yolo', default_value='false',
                              description='run yolo_measure (needs ultralytics installed)'),
        camera,
        Node(package='realsense_depth', executable='depth_measure',
             name='depth_measure', output='screen',
             parameters=[params, {'viewer': viewer}]),
        Node(package='realsense_depth', executable='cloud_pipeline',
             name='cloud_pipeline', output='screen',
             parameters=[params]),
        Node(package='realsense_depth', executable='yolo_measure',
             name='yolo_measure', output='screen',
             parameters=[params, {'viewer': viewer}],
             condition=IfCondition(yolo)),
    ])
