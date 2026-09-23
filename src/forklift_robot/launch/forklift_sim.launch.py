"""Bring up Gazebo with the forklift and a pallet, ready for training.

One launch file; the variants that would otherwise be separate copies are
launch arguments.

    ros2 launch forklift_robot forklift_sim.launch.py            # headless
    ros2 launch forklift_robot forklift_sim.launch.py gui:=true rviz:=true
"""

from __future__ import annotations

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ROBOT_PKG = "forklift_robot"
ENV_PKG = "forklift_gym_env"


def generate_launch_description() -> LaunchDescription:
    robot_share = get_package_share_directory(ROBOT_PKG)
    env_share = get_package_share_directory(ENV_PKG)

    # Gazebo needs to find the pallet mesh. The collision plugin is discovered
    # through the <gazebo_ros plugin_path=...> export in ros_gazebo_plugins'
    # package.xml; setting it here too means the launch still works when that
    # hook is not picked up.
    _prepend_env("GAZEBO_MODEL_PATH", os.path.join(env_share, "models"))
    _prepend_env(
        "GAZEBO_PLUGIN_PATH",
        os.path.join(get_package_share_directory("ros_gazebo_plugins"), "..", "..", "lib"),
    )

    args = [
        DeclareLaunchArgument(
            "gui",
            default_value="false",
            description="run gzclient (costs frame time; leave off to train)",
        ),
        DeclareLaunchArgument("rviz", default_value="false", description="also open RViz"),
        DeclareLaunchArgument(
            "world",
            default_value="collision_detection.world",
            description="world file name inside forklift_gym_env/worlds",
        ),
        DeclareLaunchArgument(
            "verbose", default_value="false", description="gazebo verbose output"
        ),
        DeclareLaunchArgument("pallet_x", default_value="6.0"),
        DeclareLaunchArgument("pallet_y", default_value="2.0"),
    ]

    robot_description = xacro.process_file(
        os.path.join(robot_share, "urdf", "forklift.urdf.xacro")
    ).toxml()

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("gazebo_ros"), "/launch", "/gazebo.launch.py"]
        ),
        launch_arguments={
            "world": PathJoinSubstitution([env_share, "worlds", LaunchConfiguration("world")]),
            "gui": LaunchConfiguration("gui"),
            "verbose": LaunchConfiguration("verbose"),
        }.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_forklift",
        output="screen",
        arguments=["-topic", "robot_description", "-entity", "forklift_bot", "-z", "0.30"],
    )

    # The pallet is spawned once here and then *teleported* on every reset,
    # rather than deleted and respawned (which cost ~6 s of sleep per episode).
    spawn_pallet = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_pallet",
        output="screen",
        arguments=[
            "-file",
            os.path.join(env_share, "models", "pallet", "model.sdf"),
            "-entity",
            "pallet",
            "-x",
            LaunchConfiguration("pallet_x"),
            "-y",
            LaunchConfiguration("pallet_y"),
            "-z",
            "0.0",
        ],
    )

    def controller(name: str) -> ExecuteProcess:
        return ExecuteProcess(
            cmd=["ros2", "control", "load_controller", "--set-state", "active", name],
            output="screen",
        )

    joint_broad = controller("joint_broad")
    fork_joint = controller("fork_joint_controller")
    diff_cont = controller("diff_cont")

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
        arguments=["-d", os.path.join(robot_share, "rviz", "forklift_with_sensors.rviz")],
        parameters=[{"use_sim_time": True}],
    )

    # Controllers are chained off process exits so they load in order; loading
    # them concurrently races the controller_manager and fails intermittently.
    return LaunchDescription(
        [
            *args,
            gazebo,
            robot_state_publisher,
            spawn_robot,
            spawn_pallet,
            RegisterEventHandler(OnProcessExit(target_action=spawn_robot, on_exit=[joint_broad])),
            RegisterEventHandler(OnProcessExit(target_action=joint_broad, on_exit=[fork_joint])),
            RegisterEventHandler(OnProcessExit(target_action=fork_joint, on_exit=[diff_cont])),
            rviz,
        ]
    )


def _prepend_env(name: str, value: str) -> None:
    existing = os.environ.get(name, "")
    if value not in existing.split(os.pathsep):
        os.environ[name] = f"{value}{os.pathsep}{existing}" if existing else value
