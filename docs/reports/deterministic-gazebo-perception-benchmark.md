# Deterministic Gazebo Perception Benchmark

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [14].

## System contribution

I established deterministic Gazebo 11/ROS Noetic capture and replay through the actual ITAndroids `GazeboSoccerPlayer`, cognition and detector path. Sixteen scenarios produced one frozen 2,250-frame replay. Four detectors consumed identical ordered frame hashes. Timestamps, head/camera transforms, world state and all candidate streams support downstream causal tests.

| Detector | Synthetic box mAP50-95 | Ball recall | Goalpost recall | L/T F1@10 | Live C++ p95 ms |
|---|---:|---:|---:|---:|---:|
| Production reference | 0.3958 | 0.5606 | 0.8788 | 0.7039 | 13.710 |
| Historical YOLO26n-320 | 0.1954 | 0.3556 | 0.3404 | 0.5384 | 5.044 |
| Controlled YOLOv8n-640 | 0.3144 | 0.5668 | 0.6269 | 0.7337 | 13.909 |
| Controlled YOLO26n-640 | 0.2539 | 0.5311 | 0.3712 | 0.6676 | 11.882 |

## Ground-truth validity

Box scoring covers render-validated ball and goalpost categories only. Typed L/T centers are separate. Robot and X box accuracy, exact occlusion reacquisition, and named junction scenarios without visible junctions are unsupported. Exact simulator pose does not turn every projected asset into valid image ground truth.

The original replay lacks a trustworthy odometry stream. Later localization studies generated explicitly declared command-derived noisy odometry rather than silently treating perfect pose differences as robot odometry.

## Temporal findings

During walking/camera motion, ball flicker was 0.0669 for controlled YOLOv8 versus 0.0919 for controlled YOLO26. Head-pan/tilt flicker was 0.0251/0.0112 for both. Small/far balls and image-border goalposts remained important failure modes. Exact occlusion reacquisition was unavailable, not zero.

Timing used the simulation host CPU with 24 OpenVINO inference threads. It is neither NUC latency nor a complete perception-frame budget. The synthetic ranking does not establish real-world accuracy superiority.

## Reuse

Public capture/replay code and interface contracts support new authorized scenarios. Frozen final images and identities are not included or rerun. See [simulation](../../simulation/) and [field-context ball selection](field-context-ball-selection.md).

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `adbb6e7e79a4ee294aa55321227fe4bb3491692cd590c774c322bdd084835960`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
