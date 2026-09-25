#!/usr/bin/env python3
"""Small deterministic ROS Noetic gait driver for deterministic perception scenario s15.

The repository's historical walker is Python-2-only. This isolated benchmark
node drives the same Gazebo joint-controller topics and is activated only by
the frozen /darwin/cmd_vel scenario command. It is not production code.
"""

import math
import threading

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import rospy


class DeterministicWalker:
    def __init__(self):
        self.lock = threading.Lock()
        self.enabled = False
        self.joints_seen = set()
        self.phase_start = None
        self.publishers = {}
        rospy.Subscriber("/darwin/joint_states", JointState, self.on_joints, queue_size=1)
        rospy.Subscriber("/darwin/cmd_vel", Twist, self.on_cmd, queue_size=1)

    def on_joints(self, msg):
        with self.lock:
            self.joints_seen.update(msg.name)
            for joint in msg.name:
                if joint not in self.publishers:
                    topic = f"/darwin/{joint}PositionController/command"
                    self.publishers[joint] = rospy.Publisher(topic, Float64, queue_size=1)

    def on_cmd(self, msg):
        with self.lock:
            requested = abs(msg.linear.x) + abs(msg.linear.y) + abs(msg.angular.z) > 1e-6
            if requested and not self.enabled:
                self.phase_start = rospy.Time.now().to_sec()
            self.enabled = requested

    def publish(self, joint, value):
        publisher = self.publishers.get(joint)
        if publisher is not None:
            publisher.publish(Float64(float(value)))

    def step(self):
        with self.lock:
            enabled = self.enabled
            phase_start = self.phase_start
        if not enabled or phase_start is None:
            for joint in (
                "j_thigh1_l", "j_thigh2_l", "j_tibia_l", "j_ankle1_l", "j_ankle2_l",
                "j_thigh1_r", "j_thigh2_r", "j_tibia_r", "j_ankle1_r", "j_ankle2_r",
                "j_shoulder_l", "j_shoulder_r", "j_high_arm_l", "j_high_arm_r",
                "j_low_arm_l", "j_low_arm_r",
            ):
                self.publish(joint, 0.0)
            return
        phase = 2.0 * math.pi * 1.35 * (rospy.Time.now().to_sec() - phase_start)
        left = math.sin(phase)
        right = math.sin(phase + math.pi)
        for suffix, wave in (("l", left), ("r", right)):
            self.publish(f"j_thigh2_{suffix}", 0.55 + 0.22 * wave)
            self.publish(f"j_tibia_{suffix}", -1.10 - 0.40 * max(0.0, wave))
            self.publish(f"j_ankle1_{suffix}", -0.55 + 0.18 * wave)
            self.publish(f"j_thigh1_{suffix}", 0.035 * wave)
            self.publish(f"j_ankle2_{suffix}", -0.035 * wave)
        self.publish("j_shoulder_l", 0.16 * right)
        self.publish("j_shoulder_r", -0.16 * left)


def main():
    rospy.init_node("perception_deterministic_walker", anonymous=False)
    walker = DeterministicWalker()
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        walker.step()
        rate.sleep()


if __name__ == "__main__":
    main()
