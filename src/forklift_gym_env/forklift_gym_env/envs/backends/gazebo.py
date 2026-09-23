"""ROS 2 / Gazebo backend.

Four things here exist to keep the per-step cost down, because in Gazebo the
simulator is the bottleneck and everything else is noise:

* **One node for the whole run.** Every service client and publisher is cached
  on it and a single executor spins it in a background thread. Creating a node
  per service call means a fresh DDS participant, and the discovery traffic that
  implies, twice per ``step()`` and eight or more times per ``reset()``.
* **Resets teleport, they do not respawn.** ``/gazebo/set_entity_state`` moves
  the entities and zeroes their twist. Nothing is deleted, so the ros2_control
  controllers stay loaded and there is nothing to sleep for.
* **The world runs uncapped** (``real_time_update_rate: 0``) and this backend
  paces itself off ``/clock``, so real time is not the limit.
* **Every wait has a timeout** and raises :class:`GazeboTimeoutError` naming
  what it was waiting for, rather than spinning forever against a stalled
  ``gzserver``.
"""

from __future__ import annotations

import math
import threading
from typing import Any

from forklift_gym_env.config import GazeboConfig, RobotConfig
from forklift_gym_env.envs.backends.base import DriveCommand, SimulationBackend, WorldState
from forklift_gym_env.geometry import Pose2D, quaternion_from_yaw, yaw_from_quaternion


class GazeboUnavailableError(RuntimeError):
    """Raised when a required Gazebo service never shows up."""


class GazeboTimeoutError(RuntimeError):
    """Raised when the simulator stops responding within the configured budget."""


def _require_ros() -> tuple[Any, ...]:
    """Import the ROS stack, with an actionable message when it is missing."""
    try:
        import rclpy
        from gazebo_msgs.msg import ContactsState
        from gazebo_msgs.srv import GetEntityState, SetEntityState
        from geometry_msgs.msg import Twist
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from std_msgs.msg import Float64MultiArray
        from std_srvs.srv import Empty
    except ImportError as exc:  # pragma: no cover - only hit without ROS
        raise ImportError(
            "The Gazebo backend needs a sourced ROS 2 Humble install "
            "(rclpy, gazebo_msgs, std_srvs). Either `source /opt/ros/humble/setup.bash` "
            "or run with `env.backend: kinematic`, which needs no ROS at all."
        ) from exc
    return (
        rclpy,
        Node,
        SingleThreadedExecutor,
        QoSProfile,
        ReliabilityPolicy,
        Empty,
        GetEntityState,
        SetEntityState,
        Twist,
        Float64MultiArray,
        Clock,
        ContactsState,
    )


class GazeboBackend(SimulationBackend):
    """Drives a forklift in Gazebo through a single long-lived ROS 2 node."""

    name = "gazebo"

    #: Link whose pose is treated as the robot's pose.
    REFERENCE_LINK = "chassis_bottom_link"

    def __init__(self, robot: RobotConfig, config: GazeboConfig) -> None:
        (
            rclpy,
            Node,
            SingleThreadedExecutor,
            QoSProfile,
            ReliabilityPolicy,
            Empty,
            GetEntityState,
            SetEntityState,
            Twist,
            Float64MultiArray,
            Clock,
            ContactsState,
        ) = _require_ros()

        self.robot_cfg = robot
        self.cfg = config
        self._rclpy = rclpy
        self._Twist = Twist
        self._Float64MultiArray = Float64MultiArray
        self._SetEntityState = SetEntityState
        self._GetEntityState = GetEntityState
        self._Empty = Empty

        if not rclpy.ok():
            rclpy.init()
        self._owns_context = True

        self._node = Node("forklift_gym_env")
        self._node.set_parameters(
            [rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)]
        )

        # -- cached clients: created once, reused for the whole run --------
        self._pause = self._client(Empty, "/pause_physics")
        self._unpause = self._client(Empty, "/unpause_physics")
        self._set_state = self._client(SetEntityState, "/gazebo/set_entity_state")
        self._get_state = self._client(GetEntityState, "/gazebo/get_entity_state")
        self._empty_req = Empty.Request()

        # diff_drive_controller subscribes on ~/cmd_vel_unstamped, which resolves
        # to /diff_cont/cmd_vel_unstamped -- not the bare /cmd_vel the old
        # gazebo_ros_diff_drive plugin used.
        self._cmd_vel = self._node.create_publisher(Twist, "/diff_cont/cmd_vel_unstamped", 10)
        self._fork_cmd = self._node.create_publisher(
            Float64MultiArray, "/fork_joint_controller/commands", 10
        )

        # -- sim clock ------------------------------------------------------
        self._sim_time = 0.0
        best_effort = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._node.create_subscription(Clock, "/clock", self._on_clock, best_effort)

        # -- contact sensors -------------------------------------------------
        self._contacts: set[str] = set()
        for link in self.cfg.collision_links:
            self._node.create_subscription(
                ContactsState,
                f"/collision_detection/{link}",
                self._make_contact_cb(link),
                best_effort,
            )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._stop = threading.Event()
        self._spin_thread.start()

        self._wait_for_services()
        self._spawned = False
        self._goal = Pose2D()
        self._closed = False

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #
    def _client(self, srv_type: Any, name: str) -> Any:
        return self._node.create_client(srv_type, name)

    def _spin(self) -> None:
        while not self._stop.is_set():
            self._executor.spin_once(timeout_sec=0.05)

    def _on_clock(self, msg: Any) -> None:
        self._sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _make_contact_cb(self, link: str):
        def _cb(msg: Any) -> None:
            # Ground contact is expected and is not a crash.
            for state in msg.states:
                other = state.collision2_name or ""
                if "ground_plane" in other or "ground_plane" in state.collision1_name:
                    continue
                self._contacts.add(f"{link}:{other}")

        return _cb

    def _wait_for_services(self) -> None:
        deadline = self.cfg.service_timeout_s
        for client, name in (
            (self._pause, "/pause_physics"),
            (self._unpause, "/unpause_physics"),
            (self._set_state, "/gazebo/set_entity_state"),
            (self._get_state, "/gazebo/get_entity_state"),
        ):
            if not client.wait_for_service(timeout_sec=deadline):
                raise GazeboUnavailableError(
                    f"service {name} did not appear within {deadline:g}s. "
                    "Is gazebo running, and does the world load libgazebo_ros_state.so?"
                )

    def _call(self, client: Any, request: Any) -> Any:
        """Synchronous service call with a hard timeout.

        ``rclpy.spin_until_future_complete`` is deliberately avoided: the node is
        already being spun by our executor thread, and spinning it from a second
        place is how you deadlock.
        """
        future = client.call_async(request)
        if not _wait_future(future, self.cfg.service_timeout_s):
            raise GazeboTimeoutError(
                f"{client.srv_name} did not respond within {self.cfg.service_timeout_s:g}s"
            )
        return future.result()

    # ------------------------------------------------------------------ #
    # SimulationBackend
    # ------------------------------------------------------------------ #
    def reset(self, robot: Pose2D, goal: Pose2D) -> WorldState:
        self._goal = goal
        self._call(self._unpause, self._empty_req)

        # Stop the wheels before moving anything, otherwise the residual command
        # is still being tracked when the robot lands at its new pose.
        self._publish(DriveCommand())
        self._teleport(self.cfg.robot_entity, robot, self.cfg.spawn_z)
        self._teleport(self.cfg.pallet_entity, goal, 0.0)
        self._contacts.clear()

        # Let the physics settle for a couple of control periods so the first
        # observation is not taken mid-drop.
        self._advance_sim_time(self.cfg.settle_steps * 0.05)
        state = self._read_state()
        self._call(self._pause, self._empty_req)
        return state

    def step(self, command: DriveCommand, dt: float) -> WorldState:
        self._call(self._unpause, self._empty_req)
        self._publish(command)
        self._advance_sim_time(dt)
        state = self._read_state()
        self._call(self._pause, self._empty_req)
        return state

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._spin_thread.join(timeout=2.0)
        self._executor.remove_node(self._node)
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _publish(self, command: DriveCommand) -> None:
        twist = self._Twist()
        twist.linear.x = float(command.linear)
        twist.angular.z = float(command.angular)
        self._cmd_vel.publish(twist)

        fork = self._Float64MultiArray()
        fork.data = [float(command.fork)]
        self._fork_cmd.publish(fork)

    def _teleport(self, entity: str, pose: Pose2D, z: float) -> None:
        req = self._SetEntityState.Request()
        state = req.state
        state.name = entity
        state.pose.position.x = float(pose.x)
        state.pose.position.y = float(pose.y)
        state.pose.position.z = float(z)
        qx, qy, qz, qw = quaternion_from_yaw(pose.yaw)
        state.pose.orientation.x, state.pose.orientation.y = qx, qy
        state.pose.orientation.z, state.pose.orientation.w = qz, qw
        # Zeroing the twist matters: without it the entity keeps the velocity it
        # had before the teleport and drifts away from where it was placed.
        for field in (state.twist.linear, state.twist.angular):
            field.x = field.y = field.z = 0.0
        state.reference_frame = "world"

        result = self._call(self._set_state, req)
        if result is None or not result.success:
            raise GazeboUnavailableError(
                f"could not place entity {entity!r}. Is it spawned in the world?"
            )

    def _advance_sim_time(self, seconds: float) -> None:
        """Block until the simulator's clock has advanced by ``seconds``.

        Bounded by a wall-clock timeout so a stalled ``gzserver`` surfaces as an
        error instead of an infinite loop -- which is exactly what the previous
        ``while True`` observation loops did.
        """
        import time as _time

        target = self._sim_time + seconds
        deadline = _time.monotonic() + self.cfg.service_timeout_s
        while self._sim_time < target:
            if _time.monotonic() > deadline:
                raise GazeboTimeoutError(
                    f"simulated clock advanced only {self._sim_time - target + seconds:.3f}s "
                    f"of the requested {seconds:.3f}s before timing out. Is /clock being "
                    "published and physics unpaused?"
                )
            _time.sleep(0.001)

    def _read_state(self) -> WorldState:
        req = self._GetEntityState.Request()
        req.name = f"{self.cfg.robot_entity}::{self.REFERENCE_LINK}"
        req.reference_frame = "world"
        result = self._call(self._get_state, req)
        if result is None or not result.success:
            # Fall back to the model root, whose name is stable even when the
            # lumped collision link name changes between URDF revisions.
            req.name = self.cfg.robot_entity
            result = self._call(self._get_state, req)
            if result is None or not result.success:
                raise GazeboUnavailableError(
                    f"/gazebo/get_entity_state has no state for {self.cfg.robot_entity!r}"
                )

        pose, twist = result.state.pose, result.state.twist
        yaw = yaw_from_quaternion(
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        )
        robot = Pose2D(pose.position.x, pose.position.y, yaw)
        linear = math.copysign(
            math.hypot(twist.linear.x, twist.linear.y),
            twist.linear.x * math.cos(yaw) + twist.linear.y * math.sin(yaw) or 1.0,
        )
        contacts = tuple(sorted(self._contacts))
        self._contacts.clear()
        return WorldState(
            robot=robot,
            goal=self._goal,
            linear_velocity=float(linear),
            angular_velocity=float(twist.angular.z),
            in_collision=bool(contacts),
            sim_time=self._sim_time,
            contacts=contacts,
        )


def _wait_future(future: Any, timeout_s: float) -> bool:
    """Poll a future to completion without re-spinning its node."""
    import time as _time

    deadline = _time.monotonic() + timeout_s
    while not future.done():
        if _time.monotonic() > deadline:
            return False
        _time.sleep(0.001)
    return True


__all__ = ["GazeboBackend", "GazeboTimeoutError", "GazeboUnavailableError"]
