# Supervision Scale and Representation Limits in Multitask Perception

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [12].

## Hypothesis and implementation

Did the original compact frozen-detector adapter fail mainly because it had too few trusted field masks? I held the 26,225-parameter adapter family fixed, preserved detector weights/buffers/output tensors and tested the previously missing 6,649-mask regime. A newer frozen detector representation provided a second causal contrast.

| Condition | Masks | Dice | Boundary F1@3 | False field | Border recall |
|---|---:|---:|---:|---:|---:|
| Original compact adapter | 284 | 0.959882 | 0.631050 | 0.039259 | 0.925027 |
| Expanded exact adapter | 1,381 | 0.985382 | 0.743364 | 0.010771 | 0.928736 |
| Historical detector + adapter | 6,649 | 0.983832 | 0.733400 | 0.011681 | 0.928643 |
| Newer detector + adapter | 6,649 | 0.981950 | 0.752552 | 0.015820 | 0.941868 |
| Separate MobileNet reference | 6,649 | 0.990325 | 0.817604 | 0.006470 | 0.951036 |

These are open-bridge results, not independent generalization. More data recovered much of the small-data collapse but did not improve beyond the already-expanded exact-adapter control. Newer features traded better boundaries/borders for worse Dice and false-field control.

## What this ruled out

The tested shallow frozen adapter was not rescued by this data expansion or representation substitution. This is not a theorem that all joint training or all sharing strategies fail. Detector-gradient conflict cannot explain a run with zero detector optimizer updates.

The fixed stopping rule reached 3,936 updates. Its patience tracked safety-valid improvement, so the study does not demonstrate universal optimization saturation. Both large-data artifacts failed the four frozen quality criteria and strict 1e-4 raw parity, despite useful operational agreement.

The graph has 2,532,365 parameters and approximately 0.779 billion profiled convolution MACs at 320 pixels. No new NUC latency was measured for these weights; older graph timing cannot be relabelled as a new measurement.

The [selective-sharing study](selective-sharing-multitask-perception.md) tests a materially different private-context formulation with matched controls. Its results do not rewrite this bounded negative conclusion.

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `054499e27df9bf94ec82ab1f8e3a2929216d3e6a8fbee898cfa7c34a1fceaae2`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
