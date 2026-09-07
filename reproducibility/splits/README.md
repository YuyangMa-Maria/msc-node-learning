# Tracked Experimental Splits

This directory contains the exact relative-path manifests used by the formal
experiments. It deliberately contains no image, audio or time-series samples.
Run `python tools/restore_split_manifests.py` to copy these files to the
`experiments/` locations expected by the original training scripts.

## Formal paired VSN/ASN split

`conference_grouped/` is the final leakage-resistant split used for the main
VSN, ASN, compressed-fusion and higher-layer-sharing results.

- Seed: 2026.
- Ratios: 70% train, 15% validation and 15% test at the image-group level.
- Exact duplicates: one canonical image per SHA-256 hash.
- Near duplicates: same-label 64-bit pHash neighbours at Hamming distance <= 4
  are joined transitively and assigned to one split.
- Cross-modal rule: every ASN recording inherits the split of its paired image.
- Output: 38,402 VSN images and 4,094 ASN recordings; the paired test partition
  contains 596 events.

`vsn_group_manifest.csv` records grouping hashes and duplicate provenance.
`vsn_binary_index_grouped.csv` and `asn_audio_index_grouped.csv` are consumed by
the formal model scripts. `grouped_split_report.json` contains class counts and
sanity checks. The earlier `conference_aligned` audit was superseded by this
grouped protocol and is not included here.

## CODEBRIM

`codebrim/` retains the dataset's official train/validation/test assignment and
records parent-image identifiers, multi-label targets and SHA-256 hashes. The
index contains 7,677 retained images. The 52 all-zero annotation records are
listed separately so the exclusion can be reproduced. These metadata remain
subject to the original CODEBRIM research-only licence.

## SDNET2018

`sdnet2018/` contains the frozen grouped split used for zero-shot and transfer
experiments. Seed 2026 and 50,000 candidate assignments were used. All patches
from one parent image and components linked by exact duplicate content remain
together. Four files involved in two label-conflicting hashes were excluded.

## ORION-AE VBN proxy

`vbn_orion/` contains the 69-file subset index. Files are sorted by name within
each torque level and divided deterministically 60/20/20. The binary task uses
50 files: 5/10 cNm for class 0 and 40/50/60 cNm for class 1. The split does not
separate independent structures or measurement campaigns and therefore remains
a small structural time-series proxy rather than deployment validation.

## Path and data policy

Every dataset path starts with `dataset/` and is relative to the repository
root. Hashes and labels are supplied to identify exact experimental inputs, but
the corresponding third-party samples must be obtained from their official
sources. See `docs/DATASETS.md` for links, licences and citations.
