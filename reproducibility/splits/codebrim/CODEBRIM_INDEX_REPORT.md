# CODEBRIM Multi-Label Index Report

## Task Definition

- Binary evidence head: any annotated defect.
- Multi-label heads: Crack, Spallation, Efflorescence, ExposedBars, and CorrosionStain.
- Background is represented by binary evidence label 0.
- Defect labels are not ordinal severity or risk levels.

## Integrity

- XML records: 7,734
- Physical images joined: 7,729
- All-zero annotation files excluded: 52
- Manifest images retained: 7,677
- Metadata records without a physical image: 5
- Exact duplicate files beyond first copy: 0
- Parent groups crossing official splits: 0
- Exact hashes crossing splits after policy: 0

## Split Distribution

| Split | Images | Background | Any defect | Parent groups | Multi-defect images |
|---|---:|---:|---:|---:|---:|
| train | 6,438 | 2,185 | 4,253 | 1249 | 1,640 |
| val | 611 | 150 | 461 | 138 | 183 |
| test | 628 | 150 | 478 | 135 | 167 |

## Test Policy

The official test split remains frozen. Model selection and calibration must use validation only.