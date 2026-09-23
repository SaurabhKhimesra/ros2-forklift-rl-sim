"""Keyboard teleoperation for the forklift.

Replaces the Tk GUI controller, which needed a display, hard-coded its topic
names and duplicated the publisher logic already in the environment. This runs
in any terminal, including over ssh into the Docker container.

    ros2 run forklift_robot teleop
"""

from __future__ import annotations

import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

HELP = """
forklift teleop
---------------
  w / s    drive forward / back
  a / d    turn left / right
  r / f    raise / lower the forks
  space    stop
  + / -    change speed step
  q        quit
"""

LINEAR_STEP = 0.1
ANGULAR_STEP = 0.2
FORK_STEP = 0.05


class Teleop(Node):
    def __init__(self) -> None:
        super().__init__("forklift_teleop")
        self.cmd_vel = self.create_publisher(Twist, "/diff_cont/cmd_vel_unstamped", 10)
        self.fork = self.create_publisher(Float64MultiArray, "/fork_joint_controller/commands", 10)
        self.linear = 0.0
        self.angular = 0.0
        self.fork_position = 0.0
        self.scale = 1.0

    def apply(self, key: str) -> bool:
        if key == "q":
            return False
        if key == "w":
            self.linear += LINEAR_STEP * self.scale
        elif key == "s":
            self.linear -= LINEAR_STEP * self.scale
        elif key == "a":
            self.angular += ANGULAR_STEP * self.scale
        elif key == "d":
            self.angular -= ANGULAR_STEP * self.scale
        elif key == "r":
            self.fork_position += FORK_STEP
        elif key == "f":
            self.fork_position -= FORK_STEP
        elif key == " ":
            self.linear = self.angular = 0.0
        elif key == "+":
            self.scale = min(4.0, self.scale * 1.5)
        elif key == "-":
            self.scale = max(0.1, self.scale / 1.5)
        self.publish()
        return True

    def publish(self) -> None:
        twist = Twist()
        twist.linear.x = self.linear
        twist.angular.z = self.angular
        self.cmd_vel.publish(twist)
        self.fork.publish(Float64MultiArray(data=[self.fork_position]))
        print(
            f"\rlinear {self.linear:+.2f} m/s  angular {self.angular:+.2f} rad/s  "
            f"fork {self.fork_position:+.2f}  step x{self.scale:.2f}   ",
            end="",
            flush=True,
        )


def _read_key() -> str:
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def main(argv: list[str] | None = None) -> int:
    rclpy.init(args=argv)
    node = Teleop()
    print(HELP)
    try:
        while _read_key_and_apply(node):
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.linear = node.angular = 0.0
        node.publish()
        print()
        node.destroy_node()
        rclpy.shutdown()
    return 0


def _read_key_and_apply(node: Teleop) -> bool:
    key = _read_key()
    if key == "\x03":  # Ctrl-C in raw mode
        return False
    return node.apply(key)


if __name__ == "__main__":
    sys.exit(main())
