# Field-Map Consistency, Observability and Symmetry

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [21].

## Source audit

I audited the camera, odometry, inertial, reset and landmark interfaces in a pinned ITAndroids reference checkout. A source checkout is not proof of the deployed robot revision. The audit made no model, configuration or production-source changes.

| Interface finding | Engineering consequence |
|---|---|
| Accelerometer and bias-corrected angular rate are available | Relative rotation is plausible; field-aligned absolute heading is not established |
| Gravity observation is independent of yaw | Gravity cannot remove global yaw offset |
| Torso yaw initializes/resets to zero | It cannot silently become persistent absolute field heading |
| Camera projection uses roll/pitch | Projection does not create an independent heading measurement |
| Localizer consumes walk odometry | Synthetic heading input is a new benchmark assumption |
| Known-placement resets exist | Placement priors differ from arbitrary kidnapping recovery |
| Lines carry endpoints; L/T observations lack arm orientation | Box centers cannot be treated as orientation measurements |

## Structural symmetry

For a centrally symmetric same-type landmark map, pose (p, theta) and (-p, theta + pi) predict identical robot-frame observations when each landmark m is paired with -m:

```text
R(-(theta + pi)) [-m - (-p)] = R(-theta) [m - p]
```

Relative odometry, relative yaw and co-rotated line directions preserve this ambiguity. Better measurements alone cannot identify global side from a fully ambiguous prior. An independent absolute cue or justified placement prior can break it. Tiny simulator asymmetry is not a reliable robot sensor.

## Limits of the conclusion

Symmetry explains a structural recovery limit, not every localization failure. Sparse observations can also cause premature loss of a correct local mode without a reset. The sensor study contains such a case. Earlier incomplete landmark maps and inconsistent projection planes also qualify an association-only explanation of historical failures.

The analytic tests use hand-constructed fixtures, not frozen trajectories. The [measured-line study](measured-line-localization.md) subsequently tested one bounded development intervention; it did not establish robust global recovery.

X remains disabled for localization until detector and field-map semantics are independently reconciled. No new magnetometer, absolute yaw interface or deployed behavior is inferred from this audit.

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `698e8c2b4cc11e04fb0f8be154302d68cfcdba4ae846117bb9c00e592f59bd27`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
