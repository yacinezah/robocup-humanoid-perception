# Chape Landmark-Rich Runtime Regression

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [18].

## Purpose

I validated that actual goalpost and L/T observations from Chape's perception/projection interfaces reach the observation model and remain safe with the likelihood correction. Identical observations were replayed through isolated unpatched and patched localizers. No deployed localizer was overwritten.

## Evidence

Eleven real repository frames supplied 33 observations: 14 goalposts, 11 L and 8 T. Four frames have paired camera transforms; seven use an unsynchronized default transform. This is therefore runtime/interface regression evidence, not a pose-accuracy benchmark.

| Check | Outcome |
|---|---|
| Landmark observations reach ObservationModel | Passed |
| Incompatible observation contribution | Neutral 0 before, clipped -10 after |
| Finite normalized particle weights | Passed |
| NaN, Inf or crash | None observed |
| Missing observations and goalpost/L/T interfaces | Safe in the tested regression |
| Update p95, unpatched / patched | 3.245434 / 3.320454 ms |

The p95 change was +0.075020 ms, approximately +2.31%, below the frozen material-regression criterion. This small regression set does not establish long-duration reliability or statistically precise production timing.

## Consequence

The test strengthens the engineering evidence for reviewing the isolated C++ correction. It does not demonstrate improved real localization accuracy, authorize deployment or validate the experimental learned association. Independent robot pose truth and synchronized sensor capture remain necessary for those claims.

Raw camera payloads and per-frame observations are intentionally omitted from the public repository. The [software handoff](../handoff/localizer-review.md) preserves code and provenance without republishing them.

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `e3a7e96052d63a73271e3149a06415f4e22c3875309c0094367b268f7d9766e5`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
