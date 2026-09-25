# FP32 Segmentation Benchmark on Intel NUC

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [9].

## Protocol and contribution

I benchmarked frozen FP32 exports on the actual Chape Intel NUC using batch one, one stream, one/four OpenVINO threads, 100 warm-ups and three 500-call repetitions. The same 100 timing images were predecoded. Disk decoding was excluded. The study created no new accuracy evidence.

| Configuration | One-thread p95 ms | Four-thread p95 ms |
|---|---:|---:|
| Fast-SCNN field segmenter | 19.724 | 13.079 |
| MobileNetV3-Small field segmenter | 39.903 | 24.855 |
| Boundary-transfer shared model | 57.511 | 36.915 |
| Earlier compact shared adapter | 48.419 | 30.803 |
| Expanded-data shared adapter | 58.763 | 36.808 |
| Historical YOLO26n-320 + Fast-SCNN | 55.333 | 35.620 |
| Historical YOLO26n-320 + MobileNet | 75.706 | 47.547 |

Sequential-stack values are measured complete-call distributions, not sums of isolated p95 values. No throttling was observed in this benchmark.

## Interpretation and erratum

Fast-SCNN supplies the field speed point; MobileNet supplies the quality-oriented alternative. Shared-model latency alone cannot establish superiority without its associated quality and export status. The explicit-line model remains a different task and costs 108.223 ms four-thread p95 in its historical NUC benchmark.

The original report described redundant functional detector work inside the field exports. Subsequent XML inspection disproved that description: their detector-shaped output is constant. This public edition corrects the interpretation without altering any measured timing. The 13.079 ms field-only result must not be advertised as a complete six-class multitask detector.

Fast-SCNN strict parity passed. MobileNet and the boundary-transfer shared model retain unresolved raw-parity warnings. No deployment replacement followed this benchmark. See [field-quality evidence](binary-field-segmentation.md).

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `e363d665c2ca10bc12f649f61103de5d942e355178ec5b4d668bd1fc7c994303`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
