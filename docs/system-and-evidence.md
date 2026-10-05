# Current system and evidence — 4 October 2026

The current CPU-oriented reference is **DEP-R1: buffered rectangular detector +
S-compact explicit field/line segmentation + native-grid sparse32**. Quality,
CPU efficiency and pose robustness remain separate decisions. Nothing in this
repository represents an automatic robot deployment or permission to reopen a
closed evaluation. This page supersedes older *current-reference* wording;
historical reports, measurements and failed gates retain their original meaning.

## Final reference table

| Objective | Final reference | Evidence and practical boundary |
|---|---|---|
| Detection quality | DET-R1 corrected-square production geometry | mAP50–95 **0.744006**, native geometry **0.706104**, on 688 reused development frames. Same weights/input; recorded source restoration fixed. Production training exposure unknown. |
| CPU detector | DET-R1 weight-preserving 480×640 rectangular IR | mAP **0.734455** on the same surface; laptop four-thread complete-call **30.602 ms** median repetition p95. Accuracy equivalence to corrected square and the frozen material-speed gate were not established. |
| Field/line quality | SEG-R1 C0, epoch 36 selection; separate all-2,400-mask refit | Manual-supervision geometry reference, line threshold **0.30**. Selected pre-refit evidence is exposed development; final refit has no independent accuracy estimate. Strict raw export parity remains failed. |
| CPU field/line | MTL-R2 S-compact standalone | Keeps genuine background/line/exclusive-field output at lower decoder cost. DEP-R1 reused dense surface: line Dice **0.903967**, group centerline F1 **0.803971**. Historical initializer/teacher exposure remains. |
| Full CPU perception call | DEP-R1 `rect_compact_buffered` | **46.993 ms** median repetition p95 versus **51.105 ms** original; **8.05%** reduction, exact retained outputs. Modest hardware-specific gain; failed frozen 10% material target. Ground projection/PF/ROS excluded. |
| Localization | LOC-R6 native-grid sparse32 with unchanged line potential | Useful same-world simulated tracking, **7/18** reset recovery; not robust global recovery and not superior to matched old lines. No compact-model pose validation or real-robot pose claim. |
| Multitask research | MTL-R1 development prototypes; MTL-R2 negative preservation result; MTL-R3/R3.1 stopped | No specialist-preserving shared Pareto point. R3 stopped before training; R3.1 closed **CPU_MEASUREMENT_UNRESOLVED**, zero optimizer updates and no training authorization. |

The older Fast-SCNN binary field model remains a useful **actual NUC** speed
reference ([report](reports/intel-nuc-segmentation-benchmark.md)). It does not
replace an explicit white-line model. The controlled three-seed detector study
remains stronger evidence for a same-data architecture comparison
([report](reports/controlled-yolov8n-yolo26n-comparison.md)); DET-R1's reused open
surface answers a different engineering question.

## Why the separate stack is retained

```text
source BGR image
  ├─ RGB, no-scaleup 480×640 letterbox → detector → xywh/probabilities
  │                                            → class-aware NMS → recorded inverse transform
  └─ RGB, independent 320×320 warp → S-compact → 3-channel softmax
                                               → 16×16 tiles → at most 32 diverse line pixels
```

The dense graph contains ImageNet normalization. Field union is the sum of
line and exclusive-field probabilities. The detector produces six-class
probabilities, not raw regression/class logits; tensor dimensions alone cannot
identify that distinction. The [public runtime](../src/robocup_perception/runtime/separate.py)
therefore verifies the XML **and** BIN against the
[fixed reference identities](../src/robocup_perception/runtime/separate_models.json)
before compiling. It reports source-image detections and line pixels, not
calibrated robot-frame observations or map associations.
The rectangular evidence used landscape images of aspect ratio at least 4:3;
accepting another image shape is not proof of preserved accuracy on that shape.

DEP-R1 tested model-preserving buffer reuse, concurrency and two bounded INT8
variants. Only buffer reuse was retained. Returned arrays own their data and
survive the next call; each instance rejects concurrent reentry. Lower precision
was rejected because dense-pixel scores hid losses in sparse support/geometry.
No additional fit was necessary to produce this engineering improvement.

## Matched DEP-R1 frontier

| Stack | Original complete p95, ms | Buffered complete p95, ms | Buffered paired repetition gains |
|---|---:|---:|---|
| Corrected square + refit | 80.550 | 68.981 | +24.48%, +14.36%, **−2.95%** |
| Corrected square + compact | 64.813 | 58.323 | +1.34%, +10.33%, +1.04% |
| Rectangle + refit | 66.563 | 62.666 | +10.83%, +5.21%, +14.07% |
| Rectangle + compact | **51.105** | **46.993** | **+7.38%, +8.05%, +9.55%** |

These are medians of three repetition p95s: AC i5-12450H laptop, OpenVINO
2026.2.0, four audited P-core primaries, one stream, no hyperthreading. Each
condition used 100 warmups and 500 measured calls per repetition. They are not
NUC results, thermal certification, isolated peak-memory measurements or
independent acquisitions. All tails were retained. Square+compact narrowly met
the four-thread median-summary criterion, but its median paired gain was only
1.34% and final one-thread preservation was not measured. Rectangle+compact's
more consistent modest improvement is the practical reference.

Complete-call scope includes color conversion, letterbox/warp/tensors, both
networks, finite checks, decode/NMS, source restoration/application thresholds,
softmax and sparse32 extraction. Disk, compilation, calibrated ground projection,
particle filtering, camera acquisition and ROS are excluded. Re-exporting this
public code does not make its fresh execution a repeat of the historical timing.
Its additional input/identity checks and concurrency guard are engineering
changes; **46.993 ms is the archived DEP runtime measurement**, not a new
benchmark of the public wrapper.

DEP-R1 detector mAP **0.765331 / 0.747132** for square/rectangle comes from the
different MTL-R2 open surface (572 development, 845 calibration records).
Those values must not be compared as deltas against DET-R1's table above. All
102 reused dense development/calibration images entered the full refit; its
line Dice **0.927094** and group centerline F1 **0.851019** are training-exposed
descriptions. The refit–compact gap is not a decoder-only causal comparison.

## What localization and sharing actually established

LOC-R6 froze 12 new same-world acquisition groups / 2,520 frames, with paired
seeds 0/1/2 inside scenario clusters. SEG median/p95 translation was
**0.064676 / 6.098762 m**; matched old lines were **0.062051 / 6.009872 m**.
Against the alternative goals+L/T cue set without lines, median improvement
was **63.98% [44.94, 72.22]%**. This is not incremental benefit on an otherwise
identical L/T stream. Reset recovery **7/18** and mirroring **10/36** prevent a
global-recovery claim. Synthetic heading/odometry, initialization priors and
simulator calibration qualify the entire result.

Native sparse32 preserved development utility and cut the measured full line
call p95 **147.372 → 55.312 ms**; 16 points lost approximately 18% median accuracy.
That line call includes a potential for 1,000 particles, unlike the DEP image
call. Its figures cannot be added to DEP timings as a measured end-to-end system.

MTL-R1 explored private field context and selective sharing on exposed development
data; its binary-field results remain historical evidence, not proof of explicit
line preservation. MTL-R2's compact shared model preserved the detector exactly,
but line Dice **0.882432** and centerline F1 **0.769259** lagged matched compact
separate **0.904749 / 0.802682** at the common 0.45 threshold. Sharing saved only
**2.04%** convolution MACs versus compact separate. Its 2.96% pooled four-thread
gain had inconsistent paired repetitions. Decoder compression supplied most of
the savings; a specialist-preserving shared advantage was not established.

MTL-R3 moved the proposed sharing boundary but stopped at CPU feasibility:
the F graph's median p95 reduction was **5.13%**, below the frozen **10%** gate,
with severe repetition variability. The graphs contain untrained connectors.
R3.1 retained all 30 blocks / 15,360 measured calls, concluded
**CPU_MEASUREMENT_UNRESOLVED**, and disclosed a launcher/worker telemetry PID
mismatch. Neither study supplies learned quality or authorizes training.

## Verification and useful next work

Start with the [separate-stack run instructions](reproducibility.md#current-separate-stack).
The public suite checks geometry, decoded probability semantics, class-aware NMS,
checksum rejection, buffer/output lifetime, native sparse support and actual
OpenVINO requests using tiny artificial graphs. No private images, weights,
training or evaluation payloads are used in CI. The publication extraction was
also compared with the archived DEP-R1 runtime on artificial images and existing
local reference graphs; see the [validation receipt](../reproducibility/consolidation-validation.json).
This checks extraction integrity, not generalization or timing.
The [separate-stack source manifest](../reproducibility/separate-source-provenance.json)
pins the source and public implementation hashes without changing the historical manifest.

The [source authority manifest](../reproducibility/terminal-evidence.json) records
hashes of the team-held terminal reports and machine-readable decisions used
here, without publishing frame identities or host paths. Historical reports and
their immutable release references remain in the [report index](reports/index.md).
The highest-value future integration work requires independent real-robot
accuracy/pose truth, synchronized calibration and an agreed target CPU budget.
It is a separately scoped team decision, not a queued experiment.
