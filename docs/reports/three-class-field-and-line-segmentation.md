# Three-Class Field and Line Segmentation

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [7].

## Engineering question and contribution

Field lines are thin, easily corrupted labels and useful geometric evidence. I corrected and reviewed masks, established a 1,999-mask trusted pool, and trained a MobileNetV3-Large U-Net reference under a group-safe protocol. The semantic task distinguishes background, line and field; it is not equivalent to binary field-union segmentation.

## Frozen evaluation

Recipe, representative checkpoint and line threshold 0.45 were frozen before the single corrected-final evaluation. That surface remains closed. This public edition reads the terminal report only and performs no new evaluation.

| Metric | Value |
|---|---:|
| Aggregate line Dice | 0.8487730249 |
| Image-macro line Dice | 0.7834277820 |
| Relaxed line F1 | 0.9037141476 |
| Line precision / recall | 0.8301136603 / 0.8682905323 |
| Field Dice | 0.9779854123 |
| Background Dice | 0.9695265124 |
| No-line false-positive ratio | 0.00003059895833 |

## Interpretation

The model provides a strong explicit-line reference and a source of measured line support for simulation research. Aggregate and image-macro Dice differ because they weight frames differently. The improvement cannot be attributed to annotation correction alone without a matched intervention separating all recipe changes.

Actual NUC p95 was 181.882 ms at one thread and 108.223 ms at four threads in the historical deployment benchmark. Explicit line quality therefore comes at a substantial cost. It is not the real-time field-segmentation recommendation.

Raw mask encoding is 0 background, 128 line, 255 field. Training indices are 0 background, 1 line, 2 field. Export-specific class metadata takes precedence over informal class-name order in older prose.

See [binary field segmentation](binary-field-segmentation.md) for the distinct faster task and [measured-line localization](measured-line-localization.md) for downstream use.

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `69840a2ac35ddd4b26628d5f2574ab32ae74e3a1771cd316a86da61895f09270`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
