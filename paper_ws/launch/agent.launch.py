import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):

    # Floor
    tree=Node(package='paper_ws',executable='execute_tree_1.py',output='screen',parameters=[{"use_sim_time": True},])
    floor_1 = Node(package='paper_ws',executable='floor_1_action_servers.py',output='screen',parameters=[{"use_sim_time": True},])
    ceiling_1 = Node(package='paper_ws',executable='ceiling_action_servers.py',output='screen',parameters=[{"use_sim_time": True},])
    # floor_2 = Node(package='sia_gazebo',executable='floor_2.py',output='screen',parameters=[{"use_sim_time": True},])


    nodes_to_start = [
        # tree,
        # floor_1,
        ceiling_1
    ]

    return nodes_to_start


def generate_launch_description():
    declared_arguments = []

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
