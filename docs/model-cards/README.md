# Model catalogue

These are research artifacts, not replacements authorized for the robot. Model
files are distributed separately from Git. The release artifact index records
exact checksums and whether a binary is included or withheld. No dataset is
downloaded with a model.

| Model | Intended role | Evidence | Important limitation |
|---|---|---|---|
| Three-class field/line segmenter | Explicit white-line reference | Corrected final-only real-image evaluation | Slow on NUC; operational agreement does not imply strict raw parity |
| MobileNet field segmenter | Field-quality specialist | Open development bridge | Constant detector placeholder; raw parity warning |
| Fast-SCNN field segmenter | Field-speed specialist | Open development plus actual NUC timing | Lower boundary quality than MobileNet |
| YOLOv8n-640 | Controlled six-class accuracy reference | Three-seed sealed real-image test | NUC p95 exceeds 50 ms; raw parity not passed |
| YOLO26n-640 | Controlled efficiency reference | Same-data comparison | Full accuracy noninferiority not established |
| Historical YOLO26-320 | Fast descriptive reference | Different historical accuracy protocol | Do not rank its accuracy against the controlled 640 study |
| Efficient sharing + KD | Detector-preserving multitask prototype | Selected-seed development evidence | Seed sensitivity; modest CPU gain; raw parity not passed |
| Joint sharing + KD | Multitask quality alternative | Matched development evidence | No independent generalization result; raw parity not passed |

## Segmentation contract

Inputs are RGB, directly resized to 320 x 320 and ImageNet-normalized
(`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`). Binary field is visible
field plus white lines. Thresholds are 0.575 for MobileNet and 0.5 for Fast-SCNN.
Only field logits are functional in these binary graphs. **Ignore their constant
`detections` output.** It is a compatibility placeholder, not a detector.

The three-class model uses background / line / field order. Its selected line
decision uses line probability >= 0.45, otherwise the background/field decision.
Raw masks encode background=0, line=128 and field=255. Do not silently confuse
stored intensities with class indices 0/1/2. Historical raw export error reached
0.031023 despite essentially identical operational masks; do not mark it as
passing a 1e-4 maximum-error gate.

## Detection contract

Six-class order is **ball, goalpost, robot, L, T, X**. Use RGB letterboxing with
padding 114 and float32 division by 255. Controlled models use 640 pixels, the
historical speed reference 320. Restore boxes and centers through the recorded
letterbox transform. X has no validated localization ontology.

Native YOLOv8 uses external decode/NMS. Native YOLO26 may expose end-to-end
`1 x 300 x 6` detections. The controlled detector study and the matched multitask
study have different decoding contracts: the latter deliberately uses a common
class-aware NMS at IoU 0.7 on raw heads. Use the release-specific contract and
never silently exchange decoders. Historical speed-model confidence is 0.375;
the controlled detector descriptive operating point is 0.25.

## Multitask contract

See the [full model/training contract](../methods/multitask-contract.md).
Input is `1 x 3 x 640 x 640` RGB letterbox, float32 [0,1]. Field output is
`1 x 1 x 320 x 320` logits. The efficient model has a YOLO26 detector, the joint
model a YOLOv8 detector. Raw-head variants use the published common decoder.
Calibrated confidence files accompany the corresponding artifacts when released.

Strict raw maximum errors were 0.0006122589 and 0.0001301765 respectively, above
the frozen 0.0001 gate. Neither model receives a strict-parity certificate here.
The final CPU evidence is i5-12450H laptop evidence, not Intel NUC evidence.

## Rights and intended use

See [component notices](../../THIRD_PARTY_NOTICES.md). Ultralytics-derived models
and combined code retain upstream obligations; MIT utility licensing does not
make a full YOLO stack permissively licensed. The production detector is not
redistributed. No real-robot accuracy or integration approval is implied.
