"""Frozen contact-region features; simulator Camera is explicitly geometric."""
import math,time
import cv2
import numpy as np
FIELD=(4.5,3.0)
def quaternion_matrix(xyzw):
    x, y, z, w = map(float, xyzw)
    scale = 2.0 / (x*x + y*y + z*z + w*w)
    return np.asarray([
        [1-scale*(y*y+z*z), scale*(x*y-z*w), scale*(x*z+y*w)],
        [scale*(x*y+z*w), 1-scale*(x*x+z*z), scale*(y*z-x*w)],
        [scale*(x*z-y*w), scale*(y*z+x*w), 1-scale*(x*x+y*y)],
    ], dtype=np.float64)


class Camera:
    def __init__(self, state):
        info = state["camera_info"]
        self.width, self.height = int(info["width"]), int(info["height"])
        self.fx, self.fy, self.cx, self.cy = float(info["K"][0]), float(info["K"][4]), float(info["K"][2]), float(info["K"][5])
        source = state["camera_pose_source"]
        head = source["head_link_world_pose"]
        self.rotation = quaternion_matrix(head["orientation_xyzw"])
        self.origin = np.asarray(head["position"], dtype=np.float64) + self.rotation @ np.asarray(source["sensor_pose_in_head_xyz_rpy"][:3], dtype=np.float64)

    def project(self, points):
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        link = (self.rotation.T @ (points - self.origin).T).T
        optical = np.column_stack((-link[:, 1], -link[:, 2], link[:, 0]))
        valid = optical[:, 2] > 0.001
        uv = np.empty((len(points), 2), dtype=np.float64)
        uv[:, 0] = self.fx * optical[:, 0] / optical[:, 2] + self.cx
        uv[:, 1] = self.fy * optical[:, 1] / optical[:, 2] + self.cy
        return uv, valid, optical[:, 2]

    def clipped_sphere_box(self, center, radius=0.075):
        center = np.asarray(center, dtype=np.float64)
        points = []
        for polar in np.linspace(0.0, math.pi, 21):
            for azimuth in np.linspace(0.0, 2.0 * math.pi, 41, endpoint=False):
                points.append(center + radius * np.asarray([math.sin(polar)*math.cos(azimuth), math.sin(polar)*math.sin(azimuth), math.cos(polar)]))
        uv, valid, depth = self.project(points)
        if not np.any(valid):
            return None
        uv = uv[valid]
        x1, y1 = np.min(uv, axis=0); x2, y2 = np.max(uv, axis=0)
        if x2 < 0 or y2 < 0 or x1 > self.width or y1 > self.height:
            return None
        box = [float(np.clip(x1, 0, self.width)), float(np.clip(y1, 0, self.height)), float(np.clip(x2, 0, self.width)), float(np.clip(y2, 0, self.height))]
        return None if box[2]-box[0] < 1 or box[3]-box[1] < 1 else box

    def ground_projection(self, pixel):
        u, v = map(float, pixel)
        optical = np.asarray([(u-self.cx)/self.fx, (v-self.cy)/self.fy, 1.0])
        link_ray = np.asarray([optical[2], -optical[0], -optical[1]])
        ray = self.rotation @ link_ray
        if ray[2] >= -1e-9:
            return {"valid": False, "reason": "above_geometric_horizon"}
        distance = -self.origin[2] / ray[2]
        if distance <= 0:
            return {"valid": False, "reason": "behind_camera"}
        world = self.origin + distance * ray
        signed = min(FIELD[0]-abs(world[0]), FIELD[1]-abs(world[1]))
        return {"valid": True, "world_xy_m": [float(world[0]), float(world[1])], "range_m": float(np.linalg.norm(world[:2]-self.origin[:2])), "signed_field_boundary_distance_m": float(signed)}


def precompute(prob):
    start = time.perf_counter_ns()
    small = cv2.resize(prob, (160, 160), interpolation=cv2.INTER_AREA)
    binary = (small >= 0.5).astype(np.uint8)
    area = float(binary.mean())
    valid = bool(np.isfinite(prob).all() and 0.01 <= area <= 0.95)
    if valid:
        signed_small = cv2.distanceTransform(binary, cv2.DIST_L2, 3) - cv2.distanceTransform(1-binary, cv2.DIST_L2, 3)
        signed = cv2.resize(signed_small, (320, 320), interpolation=cv2.INTER_LINEAR) * 2.0
    else:
        signed = np.zeros((320, 320), np.float32)
    return {"prob": prob, "signed": signed, "area": area, "valid": valid, "precompute_ms": (time.perf_counter_ns()-start)/1e6}


def sampled_feature(context, box, camera, image_shape):
    start = time.perf_counter_ns()
    h, w = image_shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bh = max(1.0, y2-y1); radius = float(np.clip(0.15*bh, 2.0, 12.0))
    cx = 0.5*(x1+x2); cy = min(h-1.0, y2+0.25*radius)
    sx, sy = 320.0/w, 320.0/h
    px, py, rr = cx*sx, cy*sy, max(1.0, radius*0.5*(sx+sy))
    angles = np.linspace(0, 2*math.pi, 16, endpoint=False)
    disk_xy = [(px, py)] + [(px + rr*s*math.cos(a), py + rr*s*math.sin(a)) for s in (0.45, 0.9) for a in angles]
    ring_xy = [(px + rr*s*math.cos(a), py + rr*s*math.sin(a)) for s in (1.6, 2.2) for a in angles]
    def values(points):
        xx = np.clip(np.rint([p[0] for p in points]).astype(int), 0, 319)
        yy = np.clip(np.rint([p[1] for p in points]).astype(int), 0, 319)
        return context["prob"][yy, xx]
    disk = values(disk_xy); ring = values(ring_xy)
    uncertainty = 2.0*np.minimum(disk, 1.0-disk)
    xi, yi = int(np.clip(round(px), 0, 319)), int(np.clip(round(py), 0, 319))
    center_y = int(np.clip(round(0.5*(y1+y2)*sy), 0, 319))
    center_prob = float(context["prob"][center_y, xi])
    projection = camera.ground_projection((cx, cy))
    expected_height = None; scale_error = None
    if projection.get("valid") and projection["range_m"] > 0.05:
        expected_height = 2*0.075*camera.fy/projection["range_m"]
        scale_error = abs(math.log(max(bh, 1.0)/max(expected_height, 1.0)))
    result = {
        "contact_disk_field_probability": float(np.median(disk)),
        "ring_field_probability": float(np.median(ring)),
        "predicted_signed_boundary_distance_px": float(context["signed"][yi, xi] * w/320.0),
        "predicted_boundary_distance_box_heights": float(context["signed"][yi, xi] * w/320.0 / bh),
        "local_segmentation_uncertainty": float(np.median(uncertainty)),
        "center_field_probability": center_prob,
        "contact_minus_center_probability": float(np.median(disk)-center_prob),
        "segmentation_valid": context["valid"],
        "segmentation_area_fraction": context["area"],
        "projection_valid": bool(projection.get("valid")),
        "projection": projection,
        "expected_ball_height_px": expected_height,
        "apparent_scale_log_error": scale_error,
    }
    result["candidate_feature_ms"] = (time.perf_counter_ns()-start)/1e6
    return result
