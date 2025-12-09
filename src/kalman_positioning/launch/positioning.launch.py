from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='kalman_positioning',
            executable='positioning_node',
            name='kalman_positioning_node',
            output='screen'
        )
    ])