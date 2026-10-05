"""DEP-R1 separate CPU inference with explicit geometry and artifact identity.

This is an inference-only extraction, not the historical benchmark harness.
Weights and evaluation data are never downloaded. One instance is non-reentrant.
"""
import argparse
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import threading

import cv2
import numpy as np

from robocup_perception.localization.sparse_support import support_pixels, source_coordinates

CLASSES = ["ball", "goalpost", "robot", "L", "T", "X"]
DENSE_CLASSES = ["background", "line", "exclusive_field"]


def references():
    return json.loads(files(__package__).joinpath("separate_models.json").read_text(encoding="utf-8"))


def verify_artifact(xml, expected):
    """Check both IR files before OpenVINO reads them; never trust shape alone."""
    xml = Path(xml)
    if xml.suffix.lower() != ".xml":
        raise ValueError("An OpenVINO XML and its sibling BIN are required")
    actual = {}
    for suffix in ("xml", "bin"):
        path = xml if suffix == "xml" else xml.with_suffix(".bin")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        actual[suffix] = h.hexdigest()
        if actual[suffix] != expected[suffix + "_sha256"]:
            raise ValueError(f"Artifact identity mismatch: {path.name} ({suffix})")
    return actual


def prepare(bgr, height=480, buffers=None):
    """No-scaleup detector letterbox and independent 320-square dense warp."""
    if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected a nonempty HxWx3 uint8 BGR image")
    h, w = bgr.shape[:2]
    if min(h, w) <= 0 or height not in (480, 640):
        raise ValueError("Nonempty image and detector height 480 or 640 required")
    r = min(1., height / h, 640 / w)
    nw, nh = round(w * r), round(h * r)
    if min(nw, nh) <= 0:
        raise ValueError("Image aspect ratio collapses a resized dimension")
    left, top = (640 - nw) // 2, (height - nh) // 2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (nw, nh)) if (nw, nh) != (w, h) else rgb
    padded = cv2.copyMakeBorder(resized, top, height - nh - top, left, 640 - nw - left,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    small = cv2.resize(rgb, (320, 320))
    if buffers is None:
        x = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.
        dense = np.ascontiguousarray(small.transpose(2, 0, 1)[None], dtype=np.float32) / 255.
    else:
        x, dense = buffers
        for array, shape in ((x, (1, 3, height, 640)), (dense, (1, 3, 320, 320))):
            if array.shape != shape or array.dtype != np.float32 or not array.flags.c_contiguous:
                raise ValueError("Reusable buffers must have the exact contiguous float32 input shapes")
        np.copyto(x[0], padded.transpose(2, 0, 1))
        np.divide(x, 255., out=x)
        np.copyto(dense[0], small.transpose(2, 0, 1))
        np.divide(dense, 255., out=dense)
    return x, dense, {"r": r, "pad": (left, top), "hw": (h, w),
                      "size": 640, "canvas_hw": (height, 640)}


def _ious(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    inter = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(2)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
    bb = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-12)


def decode_probabilities(raw):
    """Production xywh plus six probabilities, NOT the 10-channel raw YOLO head."""
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[0:2] != (1, 10) or raw.shape[2] not in (6300, 8400):
        raise ValueError("Expected production [1,10,6300|8400] xywh/probability output")
    if not np.isfinite(raw).all():
        raise FloatingPointError("Nonfinite detector output")
    if np.any((raw[:, 4:] < 0) | (raw[:, 4:] > 1)) or np.any(raw[:, 2:4] < 0):
        raise ValueError("Invalid detector probabilities or box dimensions")
    p = raw[0].T
    cid = p[:, 4:].argmax(1)
    scores = p[np.arange(len(p)), cid + 4]
    valid = scores > .25
    p, scores, cid = p[valid], scores[valid], cid[valid]
    boxes = np.column_stack((p[:, :2] - p[:, 2:4] / 2, p[:, :2] + p[:, 2:4] / 2))
    out = np.column_stack((boxes, scores, cid)).astype(np.float32)
    order = np.argsort(-scores, kind="stable")[:30000]
    shifted = boxes + cid[:, None] * 7680.
    keep = []
    while len(order) and len(keep) < 300:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        order = rest[_ious(shifted[i:i + 1], shifted[rest])[0] <= .7]
    return out[keep]


def restore(boxes, transform):
    boxes = boxes.copy()
    h, w = transform["hw"]
    left, top = transform["pad"]
    r = transform["r"]
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / r).clip(0, w)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / r).clip(0, h)
    return boxes


def probabilities(logits):
    logits = np.asarray(logits)
    if logits.shape != (1, 3, 320, 320):
        raise ValueError("Expected background/line/exclusive-field [1,3,320,320] logits")
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite dense output")
    a = np.asarray(logits, dtype=np.float32)[0]
    e = np.empty_like(a)
    np.subtract(a, a.max(0, keepdims=True), out=e)
    np.exp(e, out=e)
    np.divide(e, e.sum(0, keepdims=True), out=e)
    return e


class SeparateStack:
    """Validated immutable reference graphs, private input buffers and owned results."""

    def __init__(self, detector_xml, segmenter_xml, profile="rect_compact_buffered", threads=4):
        refs = references()
        if profile not in refs["profiles"]:
            raise ValueError(f"Unknown reference profile: {profile}")
        if threads not in (1, 4):
            raise ValueError("The supported inference thread counts are 1 and 4")
        selected = refs["profiles"][profile]
        self.profile = profile
        self.height = selected["height"]
        self.identities = {
            "detector": verify_artifact(detector_xml, refs["artifacts"][selected["detector"]]),
            "segmenter": verify_artifact(segmenter_xml, refs["artifacts"][selected["segmenter"]]),
        }
        import openvino as ov
        self.openvino_version = ov.__version__
        config = {"INFERENCE_NUM_THREADS": threads, "NUM_STREAMS": 1,
                  "INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY",
                  "SCHEDULING_CORE_TYPE": "PCORE_ONLY", "ENABLE_CPU_PINNING": True,
                  "ENABLE_HYPER_THREADING": False}
        self.core = ov.Core()
        self.models = [self.core.compile_model(str(p), "CPU", config)
                       for p in (detector_xml, segmenter_xml)]
        contracts = [((1, 3, self.height, 640), (1, 10, 6300 if self.height == 480 else 8400)),
                     ((1, 3, 320, 320), (1, 3, 320, 320))]
        for model, (input_shape, output_shape) in zip(self.models, contracts):
            if len(model.inputs) != 1 or len(model.outputs) != 1:
                raise ValueError("Reference graphs must have exactly one input and one output")
            if tuple(model.input().shape) != input_shape or tuple(model.output().shape) != output_shape:
                raise ValueError("Compiled model shape does not match its reference contract")
            if model.input().element_type != ov.Type.f32 or model.output().element_type != ov.Type.f32:
                raise ValueError("Reference graph inputs and outputs must be float32")
        self.requests = [model.create_infer_request() for model in self.models]
        self.properties = [{k: str(model.get_property(k)) for k in
                            ("INFERENCE_NUM_THREADS", "NUM_STREAMS", "INFERENCE_PRECISION_HINT",
                             "SCHEDULING_CORE_TYPE", "ENABLE_CPU_PINNING", "ENABLE_HYPER_THREADING")}
                           for model in self.models]
        self.buffers = [np.empty((1, 3, self.height, 640), np.float32),
                        np.empty((1, 3, 320, 320), np.float32)]
        self._busy = threading.Lock()

    def infer(self, bgr):
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("SeparateStack is non-reentrant; use one instance per concurrent caller")
        try:
            x, dense, transform = prepare(bgr, self.height, self.buffers)
            self.requests[0].infer({0: x}, share_inputs=True)
            self.requests[1].infer({0: dense}, share_inputs=True)
            raw = self.requests[0].get_output_tensor(0).data
            logits = self.requests[1].get_output_tensor(0).data
            boxes = restore(decode_probabilities(raw), transform)
            boxes = boxes[boxes[:, 4] >= np.where(boxes[:, 5] == 2, .6, .3)]
            prob = probabilities(logits)
            support = support_pixels(prob[1], .30, 32)
            h, w = transform["hw"]
            return {"detections": boxes, "probabilities": prob,
                    "source_line_pixels": source_coordinates(support, w, h), "transform": transform}
        finally:
            self._busy.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--segmenter", type=Path, required=True)
    parser.add_argument("--profile", choices=list(references()["profiles"]), default="rect_compact_buffered")
    parser.add_argument("--threads", type=int, choices=(1, 4), default=4)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end in .json")
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error("Cannot decode the supplied image")
    outputs = (args.output.resolve(), args.output.with_suffix(".npz").resolve())
    inputs = {p.resolve() for p in (args.image, args.detector, args.segmenter,
                                    args.detector.with_suffix(".bin"), args.segmenter.with_suffix(".bin"))}
    if any(p in inputs or p.exists() for p in outputs):
        parser.error("Output files must be new paths distinct from inputs")
    cv2.setNumThreads(1)
    stack = SeparateStack(args.detector, args.segmenter, args.profile, args.threads)
    result = stack.infer(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), probabilities=result["probabilities"])
    record = {"profile": args.profile, "model_sha256": stack.identities,
              "openvino": stack.openvino_version, "cpu_properties": stack.properties,
              "class_order": CLASSES, "dense_class_order": DENSE_CLASSES,
              "detections_xyxy_confidence_class": result["detections"].tolist(),
              "source_line_pixels_xy_probability": result["source_line_pixels"].tolist(),
              "transform": result["transform"], "line_threshold": .30,
              "probabilities": args.output.with_suffix(".npz").name,
              "dense_coordinate_system": "native 320x320 warped network grid",
              "sparse_coordinate_system": "source-image pixel centres; not robot-frame landmarks",
              "warning": "Research inference; no ground projection, pose inference or production approval."}
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
