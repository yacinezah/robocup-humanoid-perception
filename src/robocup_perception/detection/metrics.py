"""Frozen greedy one-to-one box/center metrics in the recorded coordinate system."""
from typing import Any
import statistics
import numpy as np
CLASSES=["ball","goalpost","robot","L","T","X"]

def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def match_counts(records: list[dict[str, Any]], thresholds: dict[int, float], point_radius: float | None = None) -> dict[str, Any]:
    counts = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(6)}
    junction_errors: list[float] = []
    for record in records:
        gt = record["gt"]
        preds = [p for p in record["predictions"] if p["confidence"] >= thresholds[p["class_id"]]]
        pairs = []
        for pi, pred in enumerate(preds):
            for gi, target in enumerate(gt):
                if pred["class_id"] != target["class_id"]:
                    continue
                if point_radius is not None and pred["class_id"] >= 3:
                    pc, gc = center(pred["box"]), center(target["box"])
                    distance = float(np.hypot(pc[0] - gc[0], pc[1] - gc[1]))
                    if distance <= point_radius:
                        pairs.append((point_radius - distance, pi, gi, distance))
                else:
                    overlap = iou(pred["box"], target["box"])
                    if overlap >= 0.5:
                        pairs.append((overlap, pi, gi, None))
        pairs.sort(reverse=True)
        used_p, used_g = set(), set()
        for _, pi, gi, distance in pairs:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            cid = preds[pi]["class_id"]
            counts[cid]["tp"] += 1
            if distance is not None:
                junction_errors.append(distance)
        for pi, pred in enumerate(preds):
            if pi not in used_p:
                counts[pred["class_id"]]["fp"] += 1
        for gi, target in enumerate(gt):
            if gi not in used_g:
                counts[target["class_id"]]["fn"] += 1
    per_class = {}
    f1s = []
    for cid, name in enumerate(CLASSES):
        row = counts[cid]
        precision = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        recall = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {**row, "precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    return {
        "macro_f1": statistics.fmean(f1s),
        "object_macro_f1": statistics.fmean(f1s[:3]),
        "junction_macro_f1": statistics.fmean(f1s[3:]),
        "per_class": per_class,
        "junction_center_error_p95_px": float(np.percentile(junction_errors, 95)) if junction_errors else None,
        "junction_center_error_median_px": float(np.median(junction_errors)) if junction_errors else None,
    }
