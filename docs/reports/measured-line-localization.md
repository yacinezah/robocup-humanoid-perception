# Measured-Line Constraints for Localization

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [20].

## Bounded causal screen

I added measured line evidence to the sensor-aided localizer using at most 32 spatially separated support pixels from the frozen explicit-line segmenter. Projection uses declared camera geometry. A bounded mean robust map-distance potential compares support with finite field segments and the center circle. Neither simulator landmark identity nor robot pose enters the line scorer.

One frozen seed and six reused development trajectories supply 1,260 identical frames. This is a development screen, not independent evaluation.

| Condition | Translation median/p95 m | Yaw median/p95 deg | Catastrophic / mirrored |
|---|---:|---:|---:|
| Goalposts + L/T baseline | 0.136121 / 0.485863 | 1.162288 / 6.460830 | 0 / 0 of 6 |
| Goalposts + measured lines | 0.034177 / 0.243307 | 0.682023 / 5.287747 | 0 / 0 of 6 |
| Goalposts + lines + L/T | 0.052482 / 0.265097 | 0.613552 / 4.949603 | 0 / 0 of 6 |

## Result and interpretation

Measured lines provide useful geometric constraints in this screen. Adding lines to the L/T baseline reduced translation p95 by 45.44%. However, adding L/T to the goalpost-plus-line control worsened translation p95 by 8.96%. Line utility is not proof of incremental typed-landmark utility. The frozen requirement to improve over both controls was not met; no subsequent tuning followed.

All three conditions converged in six of six trajectories and recovered the one reset. Those small denominators do not support a broad recovery claim or a narrow statistical interval.

## Engineering checks and cost

Ten focused tests passed in the original experiment. The baseline replay reproduced with zero pose difference and identical sensor hashes. Public tests retain analytic cases without opening recorded evaluation payloads.

Incidental line-potential-only p95 was 29.159 ms for goalposts+lines and 25.975 ms for lines+L/T. These are not controlled CPU benchmark results and exclude segmentation/projection. The explicit-line segmenter itself historically cost 108.223 ms four-thread p95 on NUC.

Centering the circle avoids exploiting tiny asset asymmetry. Lines cannot resolve global 180-degree symmetry after complete heading loss. See [line code](../../src/robocup_perception/localization/lines.py) and [observability](map-observability-and-symmetry.md).

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `bd92041c521696ae0154b512422c60f85f836430041581c354e7b5128e8e3500`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
