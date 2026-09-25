# Data, coordinates and evidence boundaries

## Evidence classes

| Label | Meaning |
|---|---|
| Real held-out accuracy | Frozen group-safe real-image evaluation, opened under its original protocol |
| Open development | Selection, exposed history, teacher exposure or reused development evidence |
| Gazebo synthetic | Simulator-generated observations and ground truth, with category-specific validity |
| Oracle diagnostic | Ground-truth information deliberately supplied to isolate a causal gap |
| Actual NUC runtime | Measurements on the Chape Intel NUC, not laptop extrapolation |
| Laptop runtime | Measurements on the stated laptop/configuration only |
| Real-Chape runtime only | Actual robot-path observations without independent pose ground truth |
| Code regression | Analytic, component or interface checks, not accuracy evidence |

## Dataset boundaries

Group sequences/sessions and verified exact/near duplicates before splitting.
Include teacher exposure when assessing independence. Missing labels are not
negative labels. Masks must distinguish known pixels from ignored/unknown pixels.
No protected final, OOD, legacy or deferred identities are distributed here.

The controlled detector source contains 12,018 images, regrouped into
8,188/1,425/1,190/1,215 train/calibration/validation/test. The selected multitask
study instead reports 688 filtered open detector frames and a 77-frame exposed
field bridge. Their accuracy values cannot be merged or ranked as one protocol.

## Task definitions

- Explicit segmentation: mask bytes 0 background, 128 line, 255 field, encoded as
  training classes 0/1/2. Binary field union merges field and line.
- Detection order: ball, goalpost, robot, L, T, X. X is not enabled for localization.
- Box coordinates are `x1,y1,x2,y2`; report source-image restoration explicitly.
- Typed landmark centers are evaluated with the study's canonical pixel scale.
  A ten-pixel tolerance on resized images is not automatically interchangeable.
- Field Dice aggregates overlap counts. Image-macro Dice weights each frame
  equally. Boundary F1 must specify image scale, tolerance and boundary operator.
- False-field fraction is false-positive field pixels divided by all pixels in
  the documented implementation, not universally by non-field pixels.

## Statistical units

Use paired sequence/component/family bootstrap as the protocol specifies. PF
seeds are repeated stochastic runs, not independent captures. Conditional
development intervals do not repair selection bias or teacher exposure. A
median benefit and a tail benefit answer different questions.

## Latency

Record device, power state, runtime version, precision, batch, stream/thread
counts, warm-ups, repetitions and preprocessing boundaries. Measure sequential
stacks directly; never add isolated p95 values. Keep model-only and complete-call
times separate. Publication smoke timings are not replacements for frozen results.
