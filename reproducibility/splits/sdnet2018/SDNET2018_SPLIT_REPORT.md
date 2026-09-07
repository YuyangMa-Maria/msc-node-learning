# SDNET2018 Grouped Split Report

## Protocol

- Group key: surface type plus original-image filename prefix.
- Exact duplicates connect their parent groups before split assignment.
- Target ratios: 70% train, 15% validation, and 15% frozen test.
- The frozen test split must not be used for fine-tuning, calibration, or threshold selection.

## Audit

- Physical images scanned: 56,092
- Images retained: 56,088
- Label-conflicting files excluded: 4
- Label-conflicting exact hashes: 2
- Parent groups: 230
- Duplicate-safe components: 230
- Exact duplicate files beyond first occurrence: 13
- Cross-split component errors: 0
- Cross-split parent-group errors: 0
- Cross-split exact-hash errors: 0

## Distribution

| Split | Images | Non-cracked | Cracked | Parent groups | Components |
|---|---:|---:|---:|---:|---:|
| train | 39,258 | 33,256 | 6,002 | 161 | 161 |
| val | 8,296 | 7,106 | 1,190 | 34 | 34 |
| test | 8,534 | 7,244 | 1,290 | 35 | 35 |

## Interpretation

This split is suitable for zero-shot reporting and future target-domain transfer experiments. The complete dataset may be reported as an additional descriptive zero-shot result because no SDNET2018 sample was used to train the existing VSN, but all model selection after this point must use the grouped train/validation subsets and preserve the grouped test subset.