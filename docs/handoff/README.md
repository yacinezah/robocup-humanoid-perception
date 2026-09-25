# ITAndroids engineering handoff

## Reviewable software

Start with the [isolated C++ likelihood correction](localizer-review.md).
The original patch and component test are separate from every research
association and line mechanism. Check applicability to the team's current
checkout, build the component test and player, and review before integration.
There is no model/configuration migration. Roll back an eventual integration
with a normal revert of its merge commit.

## Model candidates

Read the [model contracts](../model-cards/README.md) before loading exports.
Fast-SCNN is the field-speed candidate, MobileNet the field-quality candidate.
The controlled 640 detector pair establishes an accuracy/CPU tradeoff, but
neither is a 20 Hz NUC solution. The historical 320 detector is faster under a
different accuracy protocol. Efficient multitask sharing is a development
prototype, not evidence of independent accuracy preservation.

The deployed model is intentionally absent. Do not replace production binaries
or change class order, preprocessing or decoded thresholds using this package.

## Research-only mechanisms

Field-aware ball reranking, probabilistic map association, synthetic heading
and measured-line potentials are research modules. Their studies establish
useful causal findings but do not provide validated real-robot global recovery.
Oracle identities, exact simulator pose and synthetic sensors must never enter
a production observation stream as if measured.

## Integration review checklist

1. Agree the actual complete perception/update budget and hardware power state.
2. Confirm RGB order, normalization, padding, coordinate restoration and labels.
3. Keep both functional outputs and reject constant compatibility placeholders.
4. Repeat strict and operational parity checks without weakening thresholds.
5. Collect independently labelled real data before model promotion.
6. Preserve exact rollback binaries/configurations and upstream provenance.

Publication tests exercise extracted Python and synthetic analytic cases. They
do not substitute for a fresh C++ build, ROS integration test or robot safety
review on the team's current branch.
