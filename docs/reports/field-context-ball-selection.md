# Field-Context Ball Selection and Generalization

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [15].

## Question and formulation

Can field context reject false balls without losing real balls near the boundary? I preserved all post-NMS candidates and compared production first-candidate behavior, confidence ranking, continuous field reranking, added geometry and offline temporal association.

Field evidence comes from a scale-aware ground-contact disk and surrounding ring, signed predicted boundary distance and local uncertainty. The ball center is not used as the sole field test. Frame-level distance quantities are precomputed. Invalid/uncertain segmentation fails open.

The original static evaluation had only one off-field false event and 29 near-boundary true instances. Hard gating lost too many true detections. A new 36-scenario/4,230-frame simulation stress corpus provided stronger controlled support.

## Frozen results

| Surface / policy | True-ball recall | Boundary recall | Off-field accepts |
|---|---:|---:|---:|
| Held-out stress, production first-candidate | 0.3424 | 0.6330 | 300 |
| Held-out stress, soft reject-only | 0.3424 | 0.6330 | 0 |
| Held-out stress, soft reranking | 0.4491 | 0.8723 | 0 |

On stress scenes, reranking improved candidate choice and acquisition: mean reported acquisition delay changed from 1.342 to 0.967 seconds. Reacquisition remained 0.267 seconds; flicker and longest-miss p95 did not improve. MobileNet and Fast-SCNN produced identical decisions. Geometry and temporal stages added no incremental benefit.

## Generalization check

On the untouched original simulation replay, overall recall fell by 0.017081 and boundary recall by 0.099099. That replay contained no supported production off-field false-ball event, so an off-field reduction cannot be computed there. Stress gains were concentrated in clutter sequences and several sequence-bootstrap improvement intervals included zero.

Fast-SCNN incremental filtering cost was approximately 0.376 ms p95 with logits already available, excluding segmentation. This is Python simulation-host timing, not reviewed NUC C++ timing.

The formulation demonstrates useful contextual signal but does not justify integration. The separate tracker-shadowing defect was isolated; no production tracker change is part of the filtering claim. See [feature/scoring code](../../src/robocup_perception/ball_selection/).

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `58a2772312c5b8b38d711830c037fcb592add6bdd42db7add590f799f997b756`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
