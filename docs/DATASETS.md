# Dataset Sources, Layout and Experimental Scope

Third-party datasets are not redistributed in this source submission. Download
each dataset from the authoritative source below, comply with its licence, and
extract it under `dataset/` using the stated local directory. Source metadata
and licences were last verified on 3 September 2026. Citation records are
provided in [`DATASET_CITATIONS.bib`](DATASET_CITATIONS.bib).

## Source summary

| Dataset | Authoritative source and version | Access | Licence | Local directory |
|---|---|---|---|---|
| Multimodal Concrete Crack Detection Dataset | [Kaggle, version 1 (8 March 2026)](https://www.kaggle.com/datasets/rupankarmajumdar/multimodal-concrete-crack-detection-dataset) | Kaggle account normally required for download | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `multimodal-concrete-crack-detection-dataset/` |
| Concrete Crack Images for Classification | [Mendeley Data, version 2](https://doi.org/10.17632/5y9wdsg2zt.2) | Public direct download | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `concrete-crack-images-for-classification-mendeley/` |
| CODEBRIM | [Zenodo, version 1.0](https://doi.org/10.5281/zenodo.2620293) | Public download subject to custom terms | [Custom research-only licence](https://zenodo.org/records/2620293/files/license.md?download=1) | `codebrim/classification_dataset/` |
| SDNET2018 | [Utah State University](https://digitalcommons.usu.edu/all_datasets/48/) | Public direct download | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `sdnet2018/` |
| ORION-AE | [Harvard Dataverse, version 1.1](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FBRDU0) | Public; select individual files where possible | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | `orion-ae-sensor-b-subset/` |

The three CC BY datasets permit reuse and redistribution with attribution, but
their raw files are omitted to keep the submission small and avoid duplicating
third-party archives. CODEBRIM is more restrictive: it is limited to
non-commercial research and education and prohibits redistribution of the
database or modified versions. Consult the original licence before reuse.

## Project-specific usage

### Paired visual and acoustic data

The Kaggle multimodal dataset supplies 4,094 image-audio pairs and the mapping
file `dataset.csv`. The project uses the images for VSN evidence classification,
the airborne recordings as an ASN proxy, and the shared pair identity for
cross-modal split alignment. The formal grouped test partition contains 596
pairs. The recordings must not be described as contact acoustic-emission
measurements.

Expected layout:

```text
multimodal-concrete-crack-detection-dataset/
  dataset.csv
  image/{positive,negative}/
  audio/{positive,negative}/
```

### Binary visual baseline

The Mendeley dataset contains 40,000 RGB patches of size 227 x 227, divided
equally between `Positive/` and `Negative/`. All images were considered when
constructing the combined formal VSN split. Exact duplicates were removed and
perceptually related samples were grouped before assignment. The images derive
from 458 high-resolution source photographs, so individual patches must not be
treated as independent acquisition scenes.

### Multi-label visual extension

Only `CODEBRIM_classification_dataset.zip` was used; the balanced archive was
not used. The expected archive MD5 is
`c1612d9674e2e628e72e7f5817c40130`. The official train/validation/test
assignment was retained. The index contains 7,677 images after excluding 52
all-zero annotation records. Its five defect labels are non-exclusive visible
evidence categories, not ordinal structural-risk levels.

### External visual domain

SDNET2018 supplies deck, pavement and wall patches labelled cracked or
non-cracked. The project scanned 56,092 files and retained 56,088 after
excluding four files involved in conflicting exact-hash labels. Patches from
the same parent image, and parent groups linked by exact duplicates, remain in
one 70/15/15 split. The frozen test split is used for external-domain and
transfer evaluation.

### Structural time-series proxy

The VBN study uses 69 ORION-AE `.mat` files from the F50A sensor (`B`) in one
measurement sequence. The binary experiment uses 50 files: 5 and 10 cNm are
mapped to the low-torque proxy class, while 40, 50 and 60 cNm are mapped to the
high-torque proxy class; 20 and 30 cNm are excluded from the binary task. This
is a controlled bolted-joint acoustic-emission benchmark, not physical
MPU6050 building-vibration validation.

## Reproduced split manifests

The exact lightweight metadata used by the formal experiments is tracked under
`reproducibility/splits/` even though raw datasets and newly generated
`experiments/` output remain ignored. The tracked files contain relative paths,
labels, split assignments and, where applicable, content hashes; they contain
no images, audio or time-series measurements.

| Manifest set | Policy |
|---|---|
| `conference_grouped/` | Seed 2026; SHA-256 exact deduplication; 64-bit pHash grouping at Hamming distance <= 4; paired ASN records inherit the split of their image group |
| `codebrim/` | Official split retained; parent identifiers and exact hashes audited; all-zero annotation exclusions recorded |
| `sdnet2018/` | Seed 2026; parent-image and exact-duplicate components assigned together using 50,000 random-search trials |
| `vbn_orion/` | Deterministic file-name order within each torque level, split 60/20/20; no random seed |

The earlier `conference_aligned` split was an intermediate audit and is not the
formal split used by the final grouped experiments. See
[`reproducibility/splits/README.md`](../reproducibility/splits/README.md) for
file-level details and limitations.

## Verification

After extraction, run:

```powershell
python scripts/scan_datasets.py
```

The generated inventory reports which directories are present, local file
counts, source URLs, versions, licence identifiers and citation keys. Exact
counts can vary only if a provider has changed a mutable archive; compare the
local files against the tracked manifests before reproducing reported results.
