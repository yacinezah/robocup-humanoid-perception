# Localizer likelihood review

The patch in `patches/observation-likelihood/` preserves review commit
`46c0660e470a314d2464bf356039ce0a5ab26f36` against base
`6740b3321d6521721f4e2937d0008a5f0d1f4c02`.

Review the current source before applying the mail patch. It fixes incompatible
observation accumulation using the existing -10 floor and includes a focused
component test. It does not install a model or add learned association.

1. Create an isolated worktree of the team's intended integration revision.
2. Check patch applicability with `git apply --check` and inspect any source drift.
3. Apply through normal team review, preserving upstream attribution.
4. Build/run `ObservationModelGate_ComponentTest` and `GazeboSoccerPlayer` in the
   team's supported environment. The standalone portfolio CI does not pretend
   to build the complete robot dependency stack.
5. Confirm finite normalized weights, missing-observation behavior, clipped
   incompatibility and compatible-over-incompatible ordering.
6. Revert the eventual integration commit for rollback. No data/config migration.

Component, Gazebo and real-Chape tests were performed in the archived engineering
study. The patch is not represented as merged. Real-Chape evidence covers 11
frames/33 observations and is runtime-only, with only four paired transforms.

Research association code under `src/robocup_perception/localization/` is
deliberately separate and must not be bundled into this review.
