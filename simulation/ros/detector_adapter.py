"""Config-selected detector adapter for deterministic replay's GazeboSoccerPlayer overlay.

The production YOLOv8 path deliberately retains the deployed RGB->BGR tensor
quirk and the team's Ultralytics NMS implementation. historical native-320 study and controlled detector study
use direct immutable OpenVINO inference with their frozen preprocessing and
decoder contracts. This module is copied into the team's CNN Python directory
only in an isolated experiment checkout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


class DetectorAdapter:
    def __init__(self, registry_path: str, detector_id: str):
        import cv2
        import numpy as np

        self.cv2 = cv2
        self.np = np
        registry_file = Path(registry_path).resolve()
        registry = json.loads(registry_file.read_text(encoding="utf-8"))
        self.root = registry_file.parent.parent
        self.cpp_wrapper_confidence = float(registry["cpp_wrapper_contract"]["confidence"])
        matches = [d for d in registry["detectors"] if d["id"] == detector_id and d.get("enabled")]
        if len(matches) != 1:
            raise ValueError("enabled detector id not found exactly once: " + detector_id)
        self.cfg = matches[0]
        self.detector_id = detector_id
        self.timing_path = Path(os.environ.get("PERCEPTION_TIMING_JSONL", str(self.root / "results" / "pipeline_timing.jsonl")))
        self.timing_path.parent.mkdir(parents=True, exist_ok=True)
        if detector_id == "production_reality_yolov8_v2_fp32_640":
            team_dir = os.environ.get("PERCEPTION_TEAM_CNN_DIR")
            if not team_dir:
                raise RuntimeError("PERCEPTION_TEAM_CNN_DIR is required for the exact team YOLOv8 decoder")
            sys.path.insert(0, team_dir)
            # The frozen team module references utils.ops. Ultralytics 8.4.x
            # relocated the same NMS function; install only the import alias so
            # the production preprocessing/postprocess implementation remains
            # byte-for-byte untouched.
            from ultralytics.utils import ops as team_ops
            if not hasattr(team_ops, "non_max_suppression"):
                from ultralytics.utils.nms import non_max_suppression
                team_ops.non_max_suppression = non_max_suppression
            try:
                import openvino.runtime  # noqa: F401
            except ModuleNotFoundError:
                # OpenVINO 2026 removed the runtime namespace while retaining
                # the same public Core/Model classes at package top level.
                import types
                import openvino as ov
                runtime_module = types.ModuleType("openvino.runtime")
                runtime_module.Core = ov.Core
                runtime_module.Model = ov.Model
                sys.modules["openvino.runtime"] = runtime_module
            import yolov8_detector as team
            self.team = team
            self.impl = team.YOLOv8Detector(str(self.root / self.cfg["xml"]))
            self.mode = "production_yolov8"
        elif detector_id in {
            "historical_yolo26n_320",
            "controlled_yolov8n_640",
            "controlled_yolo26n_640",
        }:
            import openvino as ov
            import torch
            try:
                # Ultralytics 8.4.x moved NMS out of utils.ops.
                from ultralytics.utils.nms import non_max_suppression
            except ImportError:
                # Retain compatibility with the older production-era package.
                from ultralytics.utils.ops import non_max_suppression

            core = ov.Core()
            model = core.read_model(str(self.root / self.cfg["xml"]))
            self.impl = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            self.output_port = self.impl.output(0)
            self.torch = torch
            self.non_max_suppression = non_max_suppression
            self.mode = "standard_yolov8" if detector_id == "controlled_yolov8n_640" else "end2end_yolo26"
        else:
            raise ValueError("unsupported detector: " + detector_id)

    @staticmethod
    def _ms(start: float, end: float) -> float:
        return (end - start) * 1000.0

    def _write_timing(self, record):
        record["detector_id"] = self.detector_id
        record["pid"] = os.getpid()
        record["monotonic_ns"] = time.monotonic_ns()
        with self.timing_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _production_yolov8(self, rgb_image):
        t0 = time.perf_counter()
        bgr = self.cv2.cvtColor(rgb_image, self.cv2.COLOR_RGB2BGR)
        preprocessed = self.team.preprocess_image(bgr)
        tensor = self.team.image_to_tensor(preprocessed)
        t1 = time.perf_counter()
        raw = self.impl.model(tensor)
        t2 = time.perf_counter()
        boxes = raw[self.impl.model.output(0)]
        masks = raw[self.impl.model.output(1)] if len(self.impl.model.outputs) > 1 else None
        detections = self.team.postprocess(
            pred_boxes=boxes,
            input_hw=tensor.shape[2:],
            orig_img=bgr,
            min_conf_threshold=self.cfg["confidence"],
            nms_iou_threshold=self.cfg["nms_iou"],
            agnosting_nms=self.cfg["class_agnostic_nms"],
            max_detections=self.cfg["max_detections"],
            pred_masks=masks,
        )[0]["det"]
        t3 = time.perf_counter()
        output = self.np.asarray(detections, dtype=self.np.float64).reshape((-1, 6))
        t4 = time.perf_counter()
        self._write_timing({
            "preprocess_ms": self._ms(t0, t1),
            "openvino_inference_ms": self._ms(t1, t2),
            "decode_nms_ms": self._ms(t2, t3),
            "python_output_conversion_ms": self._ms(t3, t4),
            "python_complete_ms": self._ms(t0, t4),
            "detections": int(output.shape[0]),
        })
        return output

    def _letterbox(self, rgb_image):
        size = int(self.cfg["input_shape"][2])
        height, width = rgb_image.shape[:2]
        gain = min(size / height, size / width)
        new_width, new_height = int(round(width * gain)), int(round(height * gain))
        resized = self.cv2.resize(rgb_image, (new_width, new_height), interpolation=self.cv2.INTER_LINEAR)
        dw, dh = (size - new_width) / 2.0, (size - new_height) / 2.0
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        boxed = self.cv2.copyMakeBorder(
            resized, top, bottom, left, right, self.cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        tensor = self.np.ascontiguousarray(boxed.transpose(2, 0, 1)[None]).astype(self.np.float32) / 255.0
        return tensor, gain, (left, top)

    def _scale_boxes(self, rows, gain, pad, original_shape):
        if rows.size == 0:
            return rows.reshape((0, 6))
        rows = rows.astype(self.np.float64, copy=True)
        rows[:, [0, 2]] = (rows[:, [0, 2]] - pad[0]) / gain
        rows[:, [1, 3]] = (rows[:, [1, 3]] - pad[1]) / gain
        height, width = original_shape[:2]
        rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, width)
        rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, height)
        return rows

    def _standard_openvino(self, rgb_image):
        t0 = time.perf_counter()
        tensor, gain, pad = self._letterbox(rgb_image)
        t1 = time.perf_counter()
        raw = self.impl([tensor])[self.output_port]
        t2 = time.perf_counter()
        floor = float(self.cfg.get("raw_candidate_floor", self.cfg["confidence"]))
        if self.mode == "standard_yolov8":
            rows = self.non_max_suppression(
                self.torch.from_numpy(raw),
                conf_thres=floor,
                iou_thres=float(self.cfg["nms_iou"]),
                agnostic=bool(self.cfg["class_agnostic_nms"]),
                max_det=int(self.cfg["max_detections"]),
                nc=6,
            )[0].cpu().numpy()
        else:
            rows = self.np.asarray(raw[0], dtype=self.np.float64)
            rows = rows[(rows[:, 4] >= floor) & (rows[:, 5] >= 0)]
        raw_candidates = self._scale_boxes(rows, gain, pad, rgb_image.shape)
        primary = raw_candidates[raw_candidates[:, 4] >= float(self.cfg["confidence"])]
        t3 = time.perf_counter()
        primary = self.np.ascontiguousarray(primary, dtype=self.np.float64)
        raw_candidates = self.np.ascontiguousarray(raw_candidates, dtype=self.np.float64)
        t4 = time.perf_counter()
        self._write_timing({
            "preprocess_ms": self._ms(t0, t1),
            "openvino_inference_ms": self._ms(t1, t2),
            "decode_nms_ms": self._ms(t2, t3),
            "python_output_conversion_ms": self._ms(t3, t4),
            "python_complete_ms": self._ms(t0, t4),
            "raw_candidate_floor": floor,
            "raw_detections": int(raw_candidates.shape[0]),
            "detections": int(primary.shape[0]),
        })
        return primary, raw_candidates

    def _production_wrapper(self, rows):
        kept = []
        ball_seen = False
        goalposts = 0
        classes = ["ball", "goalpost", "robot", "L-intersection", "T-intersection", "X-intersection"]
        for row in rows:
            class_id = int(row[5])
            if (
                class_id < 0
                or class_id >= len(classes)
                or row[4] < self.cpp_wrapper_confidence
            ):
                continue
            if class_id == 0:
                if ball_seen:
                    continue
                ball_seen = True
            elif class_id == 1:
                if goalposts >= 2:
                    continue
                goalposts += 1
            kept.append(row)
        return self.np.asarray(kept, dtype=self.np.float64).reshape((-1, 6))

    def detect_with_raw(self, rgb_image):
        if self.mode == "production_yolov8":
            primary = self._production_yolov8(rgb_image)
            raw = primary.copy()
        else:
            primary, raw = self._standard_openvino(rgb_image)
        return {
            "primary": primary,
            "raw_post_nms": raw,
            "production_wrapper": self._production_wrapper(primary),
        }

    def detect(self, rgb_image):
        return self.detect_with_raw(rgb_image)["primary"]
