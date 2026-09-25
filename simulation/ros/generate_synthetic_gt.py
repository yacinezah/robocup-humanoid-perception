#!/usr/bin/env python3
"""Project the scoreable deterministic perception Gazebo geometry into the frozen replay."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


import os
ROOT = Path(os.environ.get('PERCEPTION_REPLAY_ROOT', '.')).resolve()
CLASSES = ["ball", "goalpost", "robot", "L-intersection", "T-intersection", "X-intersection"]
# Manual first/middle/last render review is applied per scenario and category.
# This deliberately excludes plausible-looking projections that are outside the
# purpose of a scenario (for example the stale/default ball in goalpost scenes).
# Robot and X labels remain unsupported by the frozen world/ontology.
SCENARIO_VALID_BOX_CLASS_IDS = {
    "s02_head_pan_sweep": {0},
    "s03_head_tilt_sweep": {0, 1},
    "s04_ball_near": {0},
    "s05_ball_far_small": {0},
    "s06_ball_image_border": {0},
    "s07_ball_field_border_outside": {0},
    "s09_goalpost_border": {1},
    "s10_goalpost_truncated": {1},
}
SCENARIO_VALID_POINT_CLASS_IDS = {
    "s02_head_pan_sweep": {3, 4},
    # The frozen s11/s12 camera poses did not show their named junctions.
    "s11_l_junction": set(),
    "s12_t_junction": set(),
}
SCENARIO_VALID_CLASS_IDS = {
    scenario_id: SCENARIO_VALID_BOX_CLASS_IDS.get(scenario_id, set()) | point_ids
    for scenario_id, point_ids in SCENARIO_VALID_POINT_CLASS_IDS.items()
}
for scenario_id, box_ids in SCENARIO_VALID_BOX_CLASS_IDS.items():
    SCENARIO_VALID_CLASS_IDS.setdefault(scenario_id, set()).update(box_ids)
SCOREABLE_SCENARIOS = {
    scenario_id for scenario_id, class_ids in SCENARIO_VALID_CLASS_IDS.items() if class_ids
}

JUNCTIONS = [
    # Outer field corners and penalty-area inner corners are L junctions.
    *[(3, [x, y, 0.012], name) for x, y, name in (
        (-4.5, -3.0, "field_corner_nw"), (-4.5, 3.0, "field_corner_sw"),
        (4.5, -3.0, "field_corner_ne"), (4.5, 3.0, "field_corner_se"),
        (-3.5, -1.5, "penalty_left_inner_n"), (-3.5, 1.5, "penalty_left_inner_s"),
        (3.5, -1.5, "penalty_right_inner_n"), (3.5, 1.5, "penalty_right_inner_s"),
    )],
    # Halfway/sideline and penalty/endline contacts are T junctions.
    *[(4, [x, y, 0.012], name) for x, y, name in (
        (0.0, -3.0, "halfway_n"), (0.0, 3.0, "halfway_s"),
        (-4.5, -1.5, "penalty_left_end_n"), (-4.5, 1.5, "penalty_left_end_s"),
        (4.5, -1.5, "penalty_right_end_n"), (4.5, 1.5, "penalty_right_end_s"),
    )],
]


def quaternion_matrix(xyzw):
    x, y, z, w = [float(value) for value in xyzw]
    norm = x*x + y*y + z*z + w*w
    if norm <= 1e-18:
        return np.eye(3, dtype=np.float64)
    scale = 2.0 / norm
    return np.asarray([
        [1-scale*(y*y+z*z), scale*(x*y-z*w), scale*(x*z+y*w)],
        [scale*(x*y+z*w), 1-scale*(x*x+z*z), scale*(y*z-x*w)],
        [scale*(x*z-y*w), scale*(y*z+x*w), 1-scale*(x*x+y*y)],
    ], dtype=np.float64)


class Camera:
    def __init__(self, state):
        info = state["camera_info"]
        if info is None:
            raise RuntimeError("missing CameraInfo")
        self.width = int(info["width"])
        self.height = int(info["height"])
        matrix = info["K"]
        self.fx, self.fy = float(matrix[0]), float(matrix[4])
        self.cx, self.cy = float(matrix[2]), float(matrix[5])
        source = state["camera_pose_source"]
        head = source["head_link_world_pose"]
        if head is None:
            raise RuntimeError("missing darwin::head link pose")
        self.rotation = quaternion_matrix(head["orientation_xyzw"])
        self.origin = np.asarray(head["position"], dtype=np.float64) + self.rotation @ np.asarray(
            source["sensor_pose_in_head_xyz_rpy"][:3], dtype=np.float64
        )

    def project(self, world_points):
        points = np.asarray(world_points, dtype=np.float64).reshape((-1, 3))
        link = (self.rotation.T @ (points - self.origin).T).T
        # Gazebo camera: +X forward, +Y left, +Z up. ROS optical/image:
        # +Z forward, +X right, +Y down.
        optical = np.column_stack((-link[:, 1], -link[:, 2], link[:, 0]))
        valid = optical[:, 2] > 0.001
        uv = np.empty((len(points), 2), dtype=np.float64)
        uv[:, 0] = self.fx * optical[:, 0] / optical[:, 2] + self.cx
        uv[:, 1] = self.fy * optical[:, 1] / optical[:, 2] + self.cy
        return uv, valid, optical[:, 2]

    def clipped_box(self, world_points):
        uv, valid, depth = self.project(world_points)
        if not np.any(valid):
            return None
        uv = uv[valid]
        x1, y1 = np.min(uv, axis=0)
        x2, y2 = np.max(uv, axis=0)
        if x2 < 0 or y2 < 0 or x1 > self.width or y1 > self.height:
            return None
        box = [
            float(np.clip(x1, 0, self.width)), float(np.clip(y1, 0, self.height)),
            float(np.clip(x2, 0, self.width)), float(np.clip(y2, 0, self.height)),
        ]
        if box[2] - box[0] < 1.0 or box[3] - box[1] < 1.0:
            return None
        return box, float(np.min(depth[valid]))


def sphere_points(center, radius=0.075):
    center = np.asarray(center, dtype=np.float64)
    points = []
    for polar in np.linspace(0.0, math.pi, 25):
        for azimuth in np.linspace(0.0, 2.0 * math.pi, 49, endpoint=False):
            points.append(center + radius * np.asarray([
                math.sin(polar) * math.cos(azimuth),
                math.sin(polar) * math.sin(azimuth),
                math.cos(polar),
            ]))
    return points


def oriented_box_points(pose, half_extents):
    center = np.asarray(pose["position"], dtype=np.float64)
    rotation = quaternion_matrix(pose["orientation_xyzw"])
    hx, hy, hz = half_extents
    return [
        center + rotation @ np.asarray([sx*hx, sy*hy, sz*hz])
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ]


def junction_patch(center, radius=0.075):
    x, y, z = center
    return [[x-radius, y-radius, z], [x-radius, y+radius, z],
            [x+radius, y-radius, z], [x+radius, y+radius, z]]


def main():
    scenarios = {
        item["id"]: item
        for item in json.loads((ROOT / "configs" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    }
    replay = json.loads((ROOT / "manifests" / "rendered_frame_replay_manifest.json").read_text(encoding="utf-8"))
    states = {}
    with (ROOT / "replay" / "simulator_state.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            states[(record["scenario_id"], record["frame_index"])] = record
    output_path = ROOT / "replay" / "synthetic_ground_truth.jsonl"
    overlay_root = ROOT / "results" / "gt_validation_overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)
    selected_indices = {}
    for scenario_id in SCOREABLE_SCENARIOS:
        count = sum(frame["scenario_id"] == scenario_id for frame in replay["frames"])
        selected_indices[scenario_id] = {0, count // 2, count - 1}
    counts = {name: 0 for name in CLASSES}
    with output_path.open("w", encoding="utf-8") as output:
        for frame in replay["frames"]:
            key = (frame["scenario_id"], frame["frame_index"])
            state_record = states[key]
            valid_class_ids = sorted(SCENARIO_VALID_CLASS_IDS.get(frame["scenario_id"], set()))
            valid_box_class_ids = sorted(SCENARIO_VALID_BOX_CLASS_IDS.get(frame["scenario_id"], set()))
            valid_point_class_ids = sorted(SCENARIO_VALID_POINT_CLASS_IDS.get(frame["scenario_id"], set()))
            scoreable = bool(valid_class_ids)
            labels = []
            if scoreable:
                state = state_record["state"]
                camera = Camera(state)
                ball = state["models"].get("kidsize_ball")
                if 0 in valid_class_ids and ball is not None:
                    projected = camera.clipped_box(sphere_points(ball["position"]))
                    if projected is not None:
                        box, depth = projected
                        labels.append({"class_id": 0, "class_name": "ball", "xyxy": box,
                                       "distance_m": depth, "identity": "kidsize_ball"})
                # The frozen world nests each static goal, so Gazebo omits its
                # links from /gazebo/link_states. Project the exact fixed post
                # geometry from robocup2023.world instead.
                for goal_x in ((-4.5, 4.5) if 1 in valid_class_ids else ()):
                    for goal_y in (-1.3, 1.3):
                        identity = f"goalpost_x{goal_x:+.1f}_y{goal_y:+.1f}"
                        pose = {"position": [goal_x, goal_y, 0.925], "orientation_xyzw": [0, 0, 0, 1]}
                        projected = camera.clipped_box(oriented_box_points(pose, (0.05, 0.05, 0.925)))
                        if projected is not None:
                            box, depth = projected
                            labels.append({"class_id": 1, "class_name": "goalpost", "xyxy": box,
                                           "distance_m": depth, "identity": identity})
                for class_id, center, identity in JUNCTIONS:
                    if class_id not in valid_class_ids:
                        continue
                    projected = camera.clipped_box(junction_patch(center))
                    if projected is not None:
                        box, depth = projected
                        center_uv, center_valid, _ = camera.project([center])
                        if not center_valid[0]:
                            continue
                        center_xy = [float(center_uv[0, 0]), float(center_uv[0, 1])]
                        if not (0.0 <= center_xy[0] <= camera.width and 0.0 <= center_xy[1] <= camera.height):
                            continue
                        labels.append({"class_id": class_id, "class_name": CLASSES[class_id], "xyxy": box,
                                       "center_world": center, "center_xy": center_xy,
                                       "distance_m": depth, "identity": identity})
            for label in labels:
                counts[label["class_name"]] += 1
                x1, y1, x2, y2 = label["xyxy"]
                label.setdefault("center_xy", [(x1+x2)/2.0, (y1+y2)/2.0])
                label["apparent_size_px"] = math.sqrt(max(0.0, (x2-x1)*(y2-y1)))
            payload = {
                "scenario_id": frame["scenario_id"], "frame_index": frame["frame_index"],
                "frame_sha256": frame["sha256"], "scoreable": scoreable,
                "valid_class_ids": valid_class_ids,
                "valid_box_class_ids": valid_box_class_ids,
                "valid_point_class_ids": valid_point_class_ids,
                "scenario_gt_contract": scenarios[frame["scenario_id"]]["gt"], "labels": labels,
            }
            output.write(json.dumps(payload, separators=(",", ":")) + "\n")
            if scoreable and frame["frame_index"] in selected_indices[frame["scenario_id"]]:
                image = cv2.imread(str(ROOT / frame["relative_path"]), cv2.IMREAD_COLOR)
                for label in labels:
                    x1, y1, x2, y2 = [int(round(value)) for value in label["xyxy"]]
                    color = [(0,165,255), (255,255,255), (255,0,255), (0,255,255), (255,255,0), (0,0,255)][label["class_id"]]
                    cv2.rectangle(image, (x1,y1), (x2,y2), color, 1)
                    cv2.putText(image, label["class_name"], (x1, max(12,y1-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                target = overlay_root / f"{frame['scenario_id']}__{frame['frame_index']:06d}.png"
                cv2.imwrite(str(target), image)
    summary = {
        "status": "CATEGORY_SCOPED_FIRST_MIDDLE_LAST_RENDER_VALIDATED",
        "scoreable_scenarios": sorted(SCOREABLE_SCENARIOS),
        "scenario_valid_class_ids": {
            key: sorted(value) for key, value in SCENARIO_VALID_CLASS_IDS.items()
        },
        "scenario_valid_box_class_ids": {
            key: sorted(value) for key, value in SCENARIO_VALID_BOX_CLASS_IDS.items()
        },
        "scenario_valid_point_class_ids": {
            key: sorted(value) for key, value in SCENARIO_VALID_POINT_CLASS_IDS.items()
        },
        "behavior_only_after_render_review": ["s11_l_junction", "s12_t_junction"],
        "labels_by_class": counts,
        "ground_truth_jsonl": output_path.relative_to(ROOT).as_posix(),
        "overlay_directory": overlay_root.relative_to(ROOT).as_posix(),
        "camera_axis_contract": "Gazebo +X forward/+Y left/+Z up to image +X right/+Y down",
    }
    (ROOT / "results" / "synthetic_gt_projection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
