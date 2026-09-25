#!/usr/bin/env python3
"""Run one configured detector on the one frozen deterministic replay PNG replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2

import os
ROOT = Path(os.environ.get('PERCEPTION_REPLAY_ROOT', '.')).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector_adapter import DetectorAdapter


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifests" / "rendered_frame_replay_manifest.json").read_text(encoding="utf-8"))
    if not manifest["frames"]:
        raise RuntimeError("replay manifest contains no frames")
    adapter = DetectorAdapter(str(ROOT / "configs" / "detectors.json"), args.detector)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in manifest["frames"]:
            image_path = ROOT / record["relative_path"]
            if sha256(image_path) != record["sha256"]:
                raise RuntimeError("frame hash mismatch: " + str(image_path))
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("cannot open frame: " + str(image_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            outputs = adapter.detect_with_raw(rgb)
            payload = {
                "detector_id": args.detector,
                "scenario_id": record["scenario_id"],
                "frame_index": record["frame_index"],
                "frame_sha256": record["sha256"],
                "detections_xyxy_score_class": outputs["primary"].tolist(),
                "raw_post_nms_xyxy_score_class": outputs["raw_post_nms"].tolist(),
                "production_wrapper_xyxy_score_class": outputs["production_wrapper"].tolist(),
            }
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
