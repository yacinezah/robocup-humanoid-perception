# Sensor-Aided Localization with Typed Landmarks

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [19].

## Design

I evaluated a normalized Student-t observation mixture with uniform clutter, confidence/covariance weighting, effective-sample-size resampling and modal pose readout. Landmarks are softly marginalized over same-type map identities. This is not one-to-one association; diagnostic maximum-posterior labels do not drive hard assignments.

Twenty fresh Gazebo trajectories form ten paired families, with three particle-filter seeds. Sixty runs per condition are not sixty independent captures. All conditions share synthetic command odometry, noisy relative heading, an informed initial placement/heading prior and notified resets. Simulator pose is scoring truth, not an estimator input.

| Condition | Translation median/p95 m | Yaw median/p95 deg | Catastrophic | Mirrored | Reset recovery |
|---|---:|---:|---:|---:|---:|
| Same-sensor non-junction baseline | 0.629 / 2.066 | 4.84 / 58.96 | 13/60 | 6/60 | 0/12 |
| Exact oracle L/T | 0.156 / 0.781 | 2.83 / 15.42 | 0/60 | 0/60 | 6/12 |
| Robust mixture + YOLOv8 L/T | 0.111 / 2.911 | 1.13 / 37.22 | 7/60 | 7/60 | 2/12 |
| Same robust estimator, goalposts only | 0.561 / 2.368 | 4.55 / 34.26 | 8/60 | 5/60 | 1/12 |
| Frozen YOLO26 transfer | 0.109 / 4.296 | 1.19 / 169.64 | 11/60 | 10/60 | 2/12 |

## Tracking benefit and recovery limitation

Typical tracking improved strongly: median translation fell 82.3%, with paired family-bootstrap benefit 0.517 m [0.201,0.750]. But p95 worsened 40.9%. Against the matched robust goalpost-only control, typed observations contributed median benefit 0.450 m [0.143,0.780], without supported full-sequence tail benefit.

On sixteen non-reset trajectories the robust model reached 0.101/0.370 m median/p95. Across four reset trajectories it reached 0.379/4.581 m. A sparse non-reset failure also remained, so heading loss alone does not explain every error. The likelihood, resampling and readout changes were tested as one recipe, not individually isolated effects.

## Interpretation

The corrected map contains 12 L and 10 T landmarks; its rendered line plane is 0.021 m. Relative heading does not reveal an unknown global field side after a reset. The oracle condition uses a different inherited estimator and is not an optimal bound on the robust recipe: no greater-than-100% oracle-recovery claim is appropriate.

This demonstrates synthetic tracking utility, not real IMU performance or reliable global recovery. See [observability analysis](map-observability-and-symmetry.md) and [research code](../../src/robocup_perception/localization/).

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `026d14e87044a334bb99916c6866d01eb78dbc6c73b7f5347fa3bcc8d8eec00e`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
