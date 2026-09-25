# Sequence Leakage and Near-Duplicate Audit

Yacine Zahouani | ITA-ITAndroids | Technical report, 2026

Public edition 1.0.0. ENSEIRB reference [6].

## Engineering question

Can validation scores be trusted when adjacent soccer frames and near duplicates occur in both training and validation? I audited the historical segmentation splits and built component-aware grouping to prevent related images from crossing evaluation boundaries.

## Method

Exact pixel hashes, short same-sequence windows and perceptual-hash locality-sensitive buckets produced candidate relationships. Perceptual similarity was a grouping aid, not an automatic declaration that two green-field images were duplicates. Connected groups, rather than individual frames, define the unit of separation.

The audit organized 1,108 sequence groups and 1,958 near-duplicate clusters. There were 799 sequence singletons and 1,930 near-duplicate singletons. Median sequence-group size was 1, p90 was 4 and maximum was 14. About 23.06% of images belonged to groups larger than five.

## Results

| Historical split | Manifest rows | Mapped rows | Overlapping sequence groups | Overlapping near-duplicate clusters |
|---|---:|---:|---:|---:|
| Earlier split A | 1,160 | 1,060 | 54 | 3 |
| Earlier split B | 1,585 | 1,485 | 66 | 4 |

These are **54 and 66 overlaps in two separate splits**, not a measured “54 of 66” ratio. Both older validation protocols were potentially optimistic. The audit does not estimate the numerical size of their accuracy bias.

## Engineering consequence

I used group-safe manifests, deterministic preprocessing and explicit quarantine boundaries for subsequent studies. A hash is an identity/provenance check, not evidence that a label is correct. Trusted-mask review and duplicate grouping solve different problems.

Public utilities operate on user-provided metadata or synthetic fixtures. Protected sample identities and original manifests are not distributed. See [data eligibility](../methods/data-and-evaluation.md).

## Provenance and reuse

This is a condensed public edition of the frozen technical report. Source SHA-256: `a9aed19eb16bceb40370d2d51292c147fd7b00954b3a671f46f6f1a5b0df72fa`. The [source registry](../../reproducibility/source-provenance.json) records the original authority; [evidence definitions](../methods/data-and-evaluation.md) delimit the claims. No evaluation was rerun for publication.

Copyright 2026 Yacine Zahouani. Original report text and tables: CC BY 4.0. This is a project technical report, not a peer-reviewed publication.
