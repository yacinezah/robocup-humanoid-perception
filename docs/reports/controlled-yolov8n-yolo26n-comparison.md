# Controlled YOLOv8n-YOLO26n Comparison

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [10].

## Question and controlled design

I compared compact detector architectures under identical six-class data and evaluation rather than mixing historical training sets. Classes are ball, goalpost, robot, L, T and X intersections. The 12,018-image source was regrouped into 8,188 training, 1,425 calibration, 1,190 validation and 1,215 sealed-test images with zero component overlap.

Eight authorized fits included a resolution study and three-seed 640 comparisons. Representatives were chosen by a frozen median policy. The sealed test was opened once after selection. No test images or identities are republished here.

| Three-seed sealed-test mean | YOLOv8n-640 | YOLO26n-640 |
|---|---:|---:|
| mAP50-95 | 0.529598 | 0.519570 |
| Macro F1 | 0.773583 | 0.771719 |
| Ball F1 | 0.957776 | 0.932245 |
| Goalpost F1 | 0.800606 | 0.782725 |
| Typed junction F1@10 px | 0.758542 | 0.760153 |
| Junction-center p95 px | 5.962636 | 6.071726 |

YOLOv8 remained the controlled accuracy leader. YOLO26 missed the -0.010 mAP margin by approximately 0.000028 and the -0.020 ball-F1 margin by approximately 0.00553. Its slight typed-junction advantage did not satisfy complete noninferiority.

## Resolution and numerical findings

The 320-pixel YOLO26 candidate lost 0.10655 mAP50-95 and 0.13949 typed-junction F1@10 against its paired 640 candidate. Native 320 was not a quality-preserving substitute for this study.

Static ONNX/OpenVINO exports passed decoded count/class agreement on 64 fixtures, with source-coordinate differences below 0.0011 pixels. Strict order-sensitive raw parity failed. YOLO26's large raw maximum reflected permutation among low-ranked end-to-end rows, not a corresponding operational mismatch; the original failed verdict remains unchanged.

The original terminal report deferred NUC timing. The [subsequent NUC benchmark](intel-nuc-detector-benchmark.md) resolves that dependency. Historical fast-detector and production accuracy protocols remain incompatible with these sealed-test means.

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `0e1aa72434cab89568390b40d4b3136870c3adc0780c2b5af3f71baa2883a4c4`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
