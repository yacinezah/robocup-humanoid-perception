#!/usr/bin/env python3
"""ROS1 capture node for the frozen deterministic perception Gazebo suite.

Run only in the repository's prepared Gazebo container. It renders once and
records immutable PNG/state records; detector runs must replay those files.
Unsupported opponent/walking behaviors are recorded but never promoted to GT.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading
import time

import cv2
from cv_bridge import CvBridge
from gazebo_msgs.msg import LinkStates, ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState
from std_msgs.msg import Float64
from std_srvs.srv import Empty
import rospy


import os
ROOT = Path(os.environ.get('PERCEPTION_REPLAY_ROOT', '.')).resolve()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def lerp_keyframes(keyframes, elapsed):
    points = sorted((float(t), value) for value, t in keyframes)
    if elapsed <= points[0][0]:
        return points[0][1]
    if elapsed >= points[-1][0]:
        return points[-1][1]
    for (ta, va), (tb, vb) in zip(points, points[1:]):
        if ta <= elapsed <= tb:
            alpha = (elapsed - ta) / (tb - ta)
            if isinstance(va, list):
                return [a + alpha * (b - a) for a, b in zip(va, vb)]
            return va + alpha * (vb - va)
    raise AssertionError("unreachable")


class Capture:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Condition()
        self.model_states = None
        self.link_states = None
        self.joint_state = None
        self.camera_info = None
        self.imu = None
        self.odometry = None
        self.clock = None
        self.active = None
        self.target_frames = None
        self.frame_index = 0
        self.image_counter = 0
        self.frames = []
        self.runtime_notes = []
        self.state_stream = None
        self.frame_root = ROOT / "replay" / "frames"
        self.frame_root.mkdir(parents=True, exist_ok=True)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.on_models, queue_size=1)
        rospy.Subscriber("/gazebo/link_states", LinkStates, self.on_links, queue_size=1)
        rospy.Subscriber("/darwin/joint_states", JointState, self.on_joints, queue_size=1)
        rospy.Subscriber("/darwin/camera/camera_info", CameraInfo, self.on_camera_info, queue_size=1)
        rospy.Subscriber("/darwin/imu", Imu, self.on_imu, queue_size=1)
        rospy.Subscriber("/darwin/odom", Odometry, self.on_odometry, queue_size=1)
        rospy.Subscriber("/clock", Clock, self.on_clock, queue_size=1)
        rospy.Subscriber("/darwin/camera/image_raw", Image, self.on_image, queue_size=1, buff_size=2 ** 24)

    def on_models(self, msg):
        with self.lock:
            self.model_states = msg

    def on_links(self, msg):
        with self.lock:
            self.link_states = msg

    def on_joints(self, msg):
        with self.lock:
            self.joint_state = msg

    def on_camera_info(self, msg):
        with self.lock:
            self.camera_info = msg

    def on_imu(self, msg):
        with self.lock:
            self.imu = msg

    def on_odometry(self, msg):
        with self.lock:
            self.odometry = msg

    def on_clock(self, msg):
        with self.lock:
            self.clock = msg

    @staticmethod
    def pose_record(pose):
        return {
            "position": [pose.position.x, pose.position.y, pose.position.z],
            "orientation_xyzw": [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        }

    @staticmethod
    def twist_record(twist):
        return {
            "linear": [twist.linear.x, twist.linear.y, twist.linear.z],
            "angular": [twist.angular.x, twist.angular.y, twist.angular.z],
        }

    def snapshot(self):
        models = {}
        model_twists = {}
        if self.model_states:
            models = {name: self.pose_record(pose) for name, pose in zip(self.model_states.name, self.model_states.pose)}
            model_twists = {
                name: self.twist_record(twist)
                for name, twist in zip(self.model_states.name, self.model_states.twist)
            }
        links = {}
        link_twists = {}
        if self.link_states:
            links = {name: self.pose_record(pose) for name, pose in zip(self.link_states.name, self.link_states.pose)}
            link_twists = {
                name: self.twist_record(twist)
                for name, twist in zip(self.link_states.name, self.link_states.twist)
            }
        joints = {}
        if self.joint_state:
            joints = {name: value for name, value in zip(self.joint_state.name, self.joint_state.position)}
        camera = None
        if self.camera_info:
            camera = {
                "width": self.camera_info.width,
                "height": self.camera_info.height,
                "distortion_model": self.camera_info.distortion_model,
                "D": list(self.camera_info.D),
                "K": list(self.camera_info.K),
                "R": list(self.camera_info.R),
                "P": list(self.camera_info.P),
            }
        imu = None
        if self.imu:
            imu = {
                "stamp_ns": int(self.imu.header.stamp.to_nsec()),
                "orientation_xyzw": [
                    self.imu.orientation.x, self.imu.orientation.y,
                    self.imu.orientation.z, self.imu.orientation.w,
                ],
                "angular_velocity": [
                    self.imu.angular_velocity.x, self.imu.angular_velocity.y,
                    self.imu.angular_velocity.z,
                ],
                "linear_acceleration": [
                    self.imu.linear_acceleration.x, self.imu.linear_acceleration.y,
                    self.imu.linear_acceleration.z,
                ],
            }
        odometry = None
        if self.odometry:
            odometry = {
                "stamp_ns": int(self.odometry.header.stamp.to_nsec()),
                "pose": self.pose_record(self.odometry.pose.pose),
                "twist": self.twist_record(self.odometry.twist.twist),
            }
        camera_head_link = links.get("darwin::head")
        return {
            "models": models,
            "model_twists": model_twists,
            "links": links,
            "link_twists": link_twists,
            "joints": joints,
            "camera_info": camera,
            "camera_pose_source": {
                "head_link_world_pose": camera_head_link,
                "sensor_pose_in_head_xyz_rpy": [0.0144, 0.0, 0.06302, 0.0, 0.0, 0.0],
                "sensor_frame": "camera_link",
            },
            "imu": imu,
            "odometry": odometry,
            "clock_ns": int(self.clock.clock.to_nsec()) if self.clock else None,
        }

    def on_image(self, msg):
        with self.lock:
            self.image_counter += 1
            self.lock.notify_all()
            if self.active is None or self.state_stream is None:
                return
            if self.target_frames is not None and self.frame_index >= self.target_frames:
                return
            scenario_id = self.active
            index = self.frame_index
            state = self.snapshot()
            self.frame_index += 1
            self.lock.notify_all()
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError("PNG encode failed")
        data = encoded.tobytes()
        scenario_dir = self.frame_root / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        path = scenario_dir / f"{index:06d}.png"
        path.write_bytes(data)
        state_record = {
            "scenario_id": scenario_id,
            "frame_index": index,
            "ros_stamp_ns": int(msg.header.stamp.to_nsec()),
            "sim_time_ns": int(rospy.Time.now().to_nsec()),
            "state": state,
        }
        state_hash = canonical_hash(state_record)
        self.state_stream.write(json.dumps(state_record, separators=(",", ":")) + "\n")
        self.state_stream.flush()
        self.frames.append({
            "scenario_id": scenario_id,
            "frame_index": index,
            "ros_stamp_ns": state_record["ros_stamp_ns"],
            "sim_time_ns": state_record["sim_time_ns"],
            "relative_path": path.relative_to(ROOT).as_posix(),
            "sha256": digest_bytes(data),
            "camera_state_sha256": canonical_hash(state.get("camera_info")),
            "world_state_sha256": state_hash,
        })

    def wait_for_images(self, count, timeout_wall_s=30.0):
        deadline = time.monotonic() + timeout_wall_s
        with self.lock:
            target = self.image_counter + count
            while self.image_counter < target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("camera image timeout")
                self.lock.wait(timeout=remaining)

    def wait_for_scenario_frames(self, timeout_wall_s):
        deadline = time.monotonic() + timeout_wall_s
        with self.lock:
            while self.frame_index < self.target_frames:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"scenario {self.active} captured {self.frame_index}/{self.target_frames} frames"
                    )
                self.lock.wait(timeout=min(remaining, 0.25))


def set_pose(service, model_name, xyz_yaw):
    x, y, yaw = xyz_yaw
    state = ModelState()
    state.model_name = model_name
    state.reference_frame = "world"
    state.pose.position.x = x
    state.pose.position.y = y
    # The frozen launch spawns Darwin's body origin at z=0.42 m. Scenarios
    # freeze SE(2), so preserve that spawn height and let physics settle.
    state.pose.position.z = 0.42
    state.pose.orientation.z = math.sin(yaw / 2.0)
    state.pose.orientation.w = math.cos(yaw / 2.0)
    result = service(state)
    if not result.success:
        raise RuntimeError(f"set_model_state failed for {model_name}: {result.status_message}")


def scenario_robot_pose(scenario, elapsed):
    pose = list(scenario["robot_pose"])
    # s15 remains behavior-only. Its deterministic kinematic translation keeps
    # the camera upright while the isolated gait node supplies leg motion.
    if scenario.get("kind") == "agent_controlled":
        pose[0] += 0.15 * elapsed
    return pose


def gazebo_pitch(requested_pitch):
    # The frozen scenario convention uses negative tilt for looking down,
    # while the Gazebo neckPitch controller uses the opposite sign.
    return -float(requested_pitch)


def stabilize_warmup(capture, count, timeout_wall_s, set_model, scenario, yaw_pub, pitch_pub, pan, tilt):
    deadline = time.monotonic() + timeout_wall_s
    with capture.lock:
        target = capture.image_counter + count
    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        with capture.lock:
            if capture.image_counter >= target:
                return
        if time.monotonic() >= deadline:
            raise RuntimeError("camera image timeout during stabilized warmup")
        set_pose(set_model, "darwin", scenario_robot_pose(scenario, 0.0))
        yaw_pub.publish(Float64(float(pan)))
        pitch_pub.publish(Float64(gazebo_pitch(tilt)))
        rate.sleep()


def set_ball(service, xyz):
    state = ModelState()
    state.model_name = "kidsize_ball"
    state.reference_frame = "world"
    state.pose.position.x, state.pose.position.y, state.pose.position.z = xyz
    state.pose.orientation.w = 1.0
    result = service(state)
    if not result.success:
        raise RuntimeError("set_model_state failed for kidsize_ball: " + result.status_message)


def main():
    rospy.init_node("perception_scenario_capture", anonymous=False)
    scenarios = json.loads((ROOT / "configs" / "scenarios.json").read_text(encoding="utf-8"))
    rospy.set_param("/use_sim_time", True)
    rospy.wait_for_service("/gazebo/set_model_state", timeout=30)
    rospy.wait_for_service("/gazebo/reset_world", timeout=30)
    rospy.wait_for_service("/gazebo/unpause_physics", timeout=30)
    set_model = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
    unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
    yaw_pub = rospy.Publisher("/darwin/neckYawPositionController/command", Float64, queue_size=1, latch=True)
    pitch_pub = rospy.Publisher("/darwin/neckPitchPositionController/command", Float64, queue_size=1, latch=True)
    walk_pub = rospy.Publisher("/darwin/cmd_vel", Twist, queue_size=1)
    capture = Capture()
    state_path = ROOT / "replay" / "simulator_state.jsonl"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as state_stream:
        capture.state_stream = state_stream
        unpause()
        capture.wait_for_images(1, timeout_wall_s=60.0)
        for scenario in scenarios["scenarios"]:
            reset_world()
            set_pose(set_model, "darwin", scenario["robot_pose"])
            objects = scenario.get("objects", {})
            if "ball" in objects:
                set_ball(set_model, objects["ball"])
            elif "ball_keyframes" in objects:
                set_ball(set_model, lerp_keyframes(objects["ball_keyframes"], 0.0))
            elif "ball_keyframes" not in objects:
                set_ball(set_model, [50.0, 50.0, 0.075])
            initial_pan = (
                lerp_keyframes(scenario["head"]["pan_keyframes"], 0.0)
                if "pan_keyframes" in scenario["head"] else scenario["head"].get("pan", 0.0)
            )
            initial_tilt = (
                lerp_keyframes(scenario["head"]["tilt_keyframes"], 0.0)
                if "tilt_keyframes" in scenario["head"] else scenario["head"].get("tilt", -0.35)
            )
            yaw_pub.publish(Float64(float(initial_pan)))
            pitch_pub.publish(Float64(gazebo_pitch(initial_tilt)))
            stabilize_warmup(
                capture, int(scenarios["capture"]["warmup_frames"]), 60.0,
                set_model, scenario, yaw_pub, pitch_pub, initial_pan, initial_tilt,
            )

            with capture.lock:
                model_names = set(capture.model_states.name) if capture.model_states else set()
            if "opponent" in objects and "darwin2" not in model_names:
                capture.runtime_notes.append({
                    "scenario_id": scenario["id"],
                    "issue": "opponent_requested_but_darwin2_not_spawned",
                    "effect": "PIPELINE_BEHAVIOR_ONLY",
                })
            elif "opponent" in objects:
                set_pose(set_model, "darwin2", objects["opponent"])
            if scenario.get("kind") == "agent_controlled" and walk_pub.get_num_connections() == 0:
                capture.runtime_notes.append({
                    "scenario_id": scenario["id"],
                    "issue": "no_deterministic_walker_subscriber",
                    "effect": "PIPELINE_BEHAVIOR_ONLY",
                })

            start = rospy.Time.now().to_sec()
            duration = float(scenario["duration_s"])
            target_frames = int(round(duration * float(scenarios["camera"]["frame_rate_hz"])))
            with capture.lock:
                capture.active = scenario["id"]
                capture.frame_index = 0
                capture.target_frames = target_frames
            deadline = time.monotonic() + max(60.0, duration * 10.0)
            rate = rospy.Rate(60)
            previous_index = -1
            while not rospy.is_shutdown():
                with capture.lock:
                    frame_index = capture.frame_index
                if frame_index >= target_frames:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"scenario {scenario['id']} captured {frame_index}/{target_frames} frames"
                    )
                elapsed = min(duration, frame_index / float(scenarios["camera"]["frame_rate_hz"]))
                set_pose(set_model, "darwin", scenario_robot_pose(scenario, elapsed))
                if frame_index == previous_index:
                    rate.sleep()
                    continue
                previous_index = frame_index
                head = scenario["head"]
                pan = lerp_keyframes(head["pan_keyframes"], elapsed) if "pan_keyframes" in head else head.get("pan", 0.0)
                tilt = lerp_keyframes(head["tilt_keyframes"], elapsed) if "tilt_keyframes" in head else head.get("tilt", -0.35)
                if "ball_keyframes" in objects:
                    set_ball(set_model, lerp_keyframes(objects["ball_keyframes"], elapsed))
                yaw_pub.publish(Float64(float(pan)))
                pitch_pub.publish(Float64(gazebo_pitch(tilt)))
                walking = scenario.get("kind") == "agent_controlled"
                walk_cmd = Twist()
                walk_cmd.linear.x = 0.15 if walking else 0.0
                walk_pub.publish(walk_cmd)
                rate.sleep()
            walk_pub.publish(Twist())
            rospy.sleep(0.25)
            with capture.lock:
                capture.active = None
                capture.target_frames = None
            rospy.sleep(0.25)
    manifest_path = ROOT / "manifests" / "rendered_frame_replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status":"CAPTURED_PENDING_HASH_REPRODUCTION_AND_GT_VALIDATION",
        "frame_count":len(capture.frames),
        "runtime_notes": capture.runtime_notes,
        "frames":capture.frames,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(capture.frames), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
