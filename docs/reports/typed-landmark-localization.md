# Typed-Landmark Localization: Measurement and Association

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [16].

## Causal question

Do L/T observations help pose estimation, and is lost utility caused by measurement quality or map identity? After correcting observation gating, I froze 18 new Gazebo trajectories in 6/6/6 development/calibration/final groups, with five particle-filter replicates. Exact simulator pose is evaluation truth; prediction uses declared synthetic command-derived noisy odometry.

The oracle-identity diagnostic preserves actual detector measurements, projection noise, misses and explicit unmatched false detections. It supplies a correct map identity only where ground-truth matching permits. It is not an oracle-measurement or deployable condition.

| Condition | Translation median/p95 m | Yaw median/p95 deg | Convergence | Recovery |
|---|---:|---:|---:|---:|
| Non-junction baseline | 0.971 / 2.555 | 17.60 / 173.57 | 23.3% | 0% |
| Exact oracle L/T | 0.299 / 0.718 | 3.73 / 23.53 | 100% | 80% |
| Actual YOLOv8 measurements + oracle ID | 0.437 / 0.771 | 4.95 / 24.55 | 93.3% | 60% |
| YOLOv8 learned one-to-one association | 0.641 / 4.372 | 13.85 / 178.49 | 53.3% | 0% |
| Actual YOLO26 measurements + oracle ID | 0.437 / 0.775 | 4.92 / 24.54 | 93.3% | 60% |

Actual YOLOv8 measurements recover 79.5% of the median oracle translation gain: (0.971-0.437)/(0.971-0.299). This ratio is distinct from the paired trajectory estimator. Paired translation benefits versus baseline are 0.754 m [0.418,1.208] for exact oracle, 0.644 m [0.297,1.081] for oracle identity, and 0.071 m [-0.489,0.465] for learned association.

## Interpretation

Useful detected measurements were available, but the learned matcher left poor tails and recovery. Association precision/recall/unmatched was 57.9/25.4/73.4%. Goalpost oracle identities also improved the baseline median to 0.656 m, showing a wider identity-association issue.

**Later qualification:** [map and observability analysis](map-observability-and-symmetry.md) identified incomplete map geometry and projection-plane inconsistencies in the inherited setup. Preserve the historical results, but do not extrapolate them into a claim that association alone explains all failures. The [sensor-aided study](sensor-aided-localization.md) tests corrected interfaces separately. None of these studies demonstrates real-robot pose-accuracy improvement.

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `3f9d8ff769c416b2ba77849bad6c147fbbe6d1c64c2d5dcfdc0dacde09c9a820`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
