from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_tb3_worlds = get_package_share_directory("paper_ws")

    return LaunchDescription(
        [

            # Main autonomy node
            Node(
                package="paper_ws",
                executable="move.py",
                name="autonomy_node_python",
                output="screen",
                emulate_tty=True,
            ),
            # Behavior tree visualization
            ExecuteProcess(cmd=["py-trees-tree-viewer", "--no-sandbox"]),
        ]
    )
