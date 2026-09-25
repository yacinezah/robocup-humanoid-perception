# C++ Observation-Likelihood Correction

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [17].

## Defect and minimal repair

I traced catastrophic oracle-landmark behavior to a log-likelihood ordering error. An observation incompatible with every same-type map landmark could contribute neutral 0, while a compatible hypothesis contributed a negative log likelihood. Gating therefore made an incompatible particle artificially preferable.

The isolated correction retains the best same-type likelihood and clips its contribution to the already existing finite floor -10. Diagnostic match state remains rejected below threshold. Missing observations remain neutral. This is a correction to the existing model, not a new association architecture.

## Reviewable contribution

| Provenance item | Value |
|---|---|
| Review base | 6740b3321d6521721f4e2937d0008a5f0d1f4c02 |
| Review commit | 46c0660e470a314d2464bf356039ce0a5ab26f36 |
| Patch SHA-256 | f4adc6807e7ed9254029cc6cdcd44bb853574b34f89001a61a949d4bc5e70ddd |

The mail-form patch preserves original commit attribution and contains the focused C++ component test. It does not contain learned association or tracker redesign. Team source terms remain separate from this portfolio's original utilities.

## Validation and integration boundary

Component tests and isolated Gazebo/ROS runtime validation checked likelihood ordering, finite normalized weights and interface behavior. The later [real-Chape landmark regression](chape-landmark-runtime-regression.md) adds real perception-path observations without independent pose truth.

The patch is not represented as merged or deployed. Apply it only after reviewing the current team's source against the recorded base. Build the component test and player in an isolated worktree, then use normal team review. Rollback is a revert of the eventual integration commit; no model, data or configuration migration is required.

See the [exact patch and test](../../patches/observation-likelihood/) and [review instructions](../handoff/localizer-review.md).

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `a1602e201a3bf588ab06b2904e1ad9043450272e05d3e72d4006dd7e349fcf73`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
