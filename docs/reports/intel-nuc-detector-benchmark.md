# Detector Benchmark on Intel NUC

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [11].

## Target-hardware experiment

I measured complete detector calls on Chape's Intel Core i5-4250U (two physical/four logical CPUs), Ubuntu 20.04.6 and OpenVINO 2024.2. Batch one, one stream, FP32 hint, 100 warm-ups and three ordered 500-call repetitions were frozen. Input frames were predecoded; timing includes preprocessing, inference and decoding but excludes disk I/O.

| Detector | One-thread p95 ms | Four-thread p95 ms |
|---|---:|---:|
| Controlled YOLOv8n-640 | 184.331 | 164.389 |
| Controlled YOLO26n-640 | 142.950 | 88.471 |
| Production YOLOv8-v2 | 176.710 | 104.201 |
| Historical YOLO26n-320 | 37.711 | 23.671 |

YOLO26 reduced four-thread combined p95 by 46.18% (1.858x speed ratio), while retaining the known accuracy tradeoff. Neither controlled 640 model met the 50 ms p95 diagnostic for 20 Hz.

## Tail variability matters

YOLOv8 four-thread repetition p95 values were 167.209, 103.029 and 102.505 ms. The first repetition was tail-heavy; it was retained, not replaced by the fastest run. YOLO26 was also faster on central statistics: 1.167x at p50 and 1.236x at mean. Therefore the headline p95 ratio is valid but tail-sensitive.

All measured exports loaded with finite outputs and unchanged hashes. Maximum temperature was 78 C; thermal-throttle counters did not change. No production models, configuration or robot-service state changed.

## Recommendation

The controlled YOLO26 is a target-hardware efficiency Pareto point, not an accuracy-equivalent or production-ready replacement. The native-320 historical detector is a useful speed reference, but its accuracy cannot be ranked against the controlled sealed test. Laptop timing and simulation-host timing must remain separate.

See [controlled accuracy](controlled-yolov8n-yolo26n-comparison.md) and the [runtime tools](../../src/robocup_perception/runtime/).

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `bffec333e867267867ed8a753e06a9c4e64d367a0184d4267ec82ba52e51d19c`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
