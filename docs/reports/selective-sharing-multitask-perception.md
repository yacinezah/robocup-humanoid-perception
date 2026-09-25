# Selective Sharing and Private Semantics for Multitask Perception

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [13].

## Architecture and contribution

I replaced the shallow detector-feature adapter with a field-private semantic hierarchy. A 640-pixel six-class detector supplies stride-4/8 detail. The field pathway learns its own downsampling context and lightweight additive decoder, returning 320-pixel logits. Controls separate private capacity, shared-feature adaptation, detector-only adaptation and specialist distillation.

The selected efficient frozen-sharing model preserves detector tensors exactly. A jointly adapted YOLOv8 model with distillation is the quality alternative. Eleven bounded fits completed; no new independent corpus was available, so all claims remain development evidence.

## Matched OpenVINO development results

Field metrics use canonical320 source views on 77 teacher/history-exposed frames. Detection uses 688 filtered open frames, with thresholds calibrated on 1,137 other open frames. Boundary values are not interchangeable with older differently scaled metrics.

| System | Dice | Boundary F1 | False field | Border recall | Detector mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Efficient frozen sharing + KD | 0.990232 | 0.878308 | 0.006828 | 0.946933 | 0.674028 |
| Joint sharing + KD | 0.990761 | 0.885570 | 0.006790 | 0.954108 | 0.687033 |
| Separate YOLO26 + MobileNet | 0.989307 | 0.887347 | 0.007964 | 0.951381 | 0.674028 |

## What mattered causally

The frozen-prefix and matched field-only controls were close: joint prefix adaptation was not shown necessary for strong selected field quality. Joint KD improved Dice/false-field against joint training without KD, but not every boundary metric. The detector-only control explained most detector gain: joint-KD minus detector-only mAP was only 0.000720 in the original PyTorch discovery comparison. Both KD objectives were enabled together, so their individual effects are not isolated.

Efficient-model Dice across confirmation seeds was 0.990232, 0.984483 and 0.987780. Boundary sensitivity also appeared in field-only controls. Selected-seed quality is not a robust recipe-level noninferiority result. Group/seed bootstrap intervals remain conditional on development selection and exposure.

## Laptop cost and export

On the i5-12450H laptop, final AC-verified four-thread complete-call p95 repetitions were 78.44/54.53/54.90 ms for native efficient sharing versus 79.42/59.88/65.57 ms for separate YOLO26+MobileNet. Pooled reduction was 6.55%; the planned 10% gain was not consistently established. These measurements are not NUC results and are not pooled with earlier AC-at-start sessions.

Efficient and joint models have 2,743,661 and 3,244,931 parameters. Strict raw FP32 maximum errors were 0.0006122589 and 0.0001301765 against 0.0001. Both fail strict parity despite useful operational agreement. Both outputs are functional.

Retain efficient sharing as a labelled development prototype and joint KD as a quality alternative. Separate specialists remain the safer robust reference. See [multitask code](../../src/robocup_perception/multitask/) and [model cards](../model-cards/).

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `e8d371b281078290c1c37c98373aa8d079fa010600686ff746f7c4aa9302fbd4`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
