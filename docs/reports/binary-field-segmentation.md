# Binary Field Segmentation: Quality and Efficiency

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [8].

## Design

I compared Fast-SCNN, MobileNetV3-Small U-Net and a boundary-transfer shared model using 6,649 eligible masks. Field-union supervision treats visible field and its lines as one class. The comparison asks whether an independent compact segmenter or a shared detector adapter offers a better field-quality/cost tradeoff.

## Open-development results

These values use a repeatedly used 77-image bridge. They are not independent Chape generalization evidence.

| Architecture | Dice | Boundary F1@3 | False-field fraction | Border recall |
|---|---:|---:|---:|---:|
| Fast-SCNN | 0.981916 | 0.761577 | 0.014012 | 0.942590 |
| MobileNetV3-Small U-Net | 0.990325 | 0.817604 | 0.006470 | 0.951036 |
| Boundary-transfer shared model | 0.989132 | 0.791299 | 0.008376 | 0.945368 |

MobileNet is the field-quality reference. Fast-SCNN offers the lowest measured CPU cost. The shared model improved its earlier boundary-transfer result but did not preserve all boundary/border requirements. Clean controls were historical rather than retrained, limiting causal claims about data policy alone.

## Export contract and correction

Fast-SCNN passed matched CPU FP32 strict parity. MobileNet produced exact masks on 16 fixtures, but maximum raw deviation 0.0002637 exceeded the frozen 0.0001 ceiling. Operational agreement does not remove this warning.

**Editorial correction:** the archived field IRs contain a constant `detections` output of shape 1x300x6. Only `field_logits`, shape 1x1x320x320, is functional. They are field segmenters, not functional detector-plus-field networks. Never consume the placeholder as detections.

The unused 8,894-row exploratory arms lack an independent evaluation contract and do not establish a quality result. The largest mask count is not automatically the best scientific evidence.

See [actual NUC measurements](intel-nuc-segmentation-benchmark.md). Their timing remains valid after the graph-semantic correction.

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `d19a2464041f8c435648342a91de2c092eaeb7990c8b5b0f88129a256511ab95`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
