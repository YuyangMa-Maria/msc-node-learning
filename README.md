# Cooperative Node Learning for Local Structural Risk Sensing

This repository is the source-code submission for an MSc Computing project on
resource-constrained, multimodal sensing after a disaster. Visual (VSN),
acoustic (ASN) and structural time-series (VBN) nodes produce calibrated local
risk proxies. The physical VSN and ASN exchange compact decisions over BLE,
while compatible higher risk layers can be transferred less frequently for
collaborative learning.

The repository contains the code needed to understand how the software was
developed: dataset checks, model selection, robustness and calibration,
compression, fusion, partial parameter sharing, ESP32-S3 firmware, and the host
receiver. Raw datasets, training checkpoints, serial logs and bulk generated
reports are deliberately excluded; compact split manifests and their integrity
reports are retained for exact evaluation reproducibility.

## Implemented system

```text
camera -> private VSN encoder --\
                                +-> shared risk head -> local summary --\
microphone -> private ASN encoder-/                                  |
                                                                    +-> VSN fusion
structural proxy -> private VBN model -------------------------------/      |
                                                                           +-> alert
PC/MARH teacher -> validated shared-head update -> VSN -> BLE -> ASN
```

| Component | Implementation status |
|---|---|
| VSN | Physical camera capture, ESP-DL inference, local status, ASN BLE client, fusion and alert forwarding |
| ASN | Physical PDM capture, log-Mel frontend, ESP-DL inference, signal-validity gate and BLE server |
| VBN | Software structural time-series proxy; not a physical accelerometer deployment |
| External receiver | Serial protocol parser, run logger and browser dashboard |
| MARH | PC-side teacher update, candidate gate, serial delivery and BLE relay to ASN |

`risk_score` is a model-derived risk proxy. It is not a probability of collapse
or a certified structural assessment.

## Repository map

- `src/nlrisk/`: small reusable Python package for dataset registration and
  reference message fusion.
- `scripts/`: chronological experiment programs, from indexing and model
  selection to robustness, compression, fusion and shared-layer evaluation.
- `tools/`: deployment preparation and PC/MARH shared-head utilities.
- `firmware/vsn/` and `firmware/asn/`: final ESP-IDF applications for two XIAO
  ESP32-S3 Sense boards.
- `host_receiver/`: serial receiver, dashboard, command plans and protocol tests.
- `configs/`: portable dataset catalogue; no local absolute paths.
- `reproducibility/splits/`: exact relative-path manifests used by the formal
  grouped experiments; no third-party samples.
- `docs/`: protocol, datasets and reproduction notes.
- `tests/`: tests for the reusable fusion and repository contract.

See [DEVELOPMENT.md](DEVELOPMENT.md) for the design history and
[FILE_GUIDE.md](FILE_GUIDE.md) for the purpose of every submitted file.

For a first reading, use this order: `README.md` for the system map,
`DEVELOPMENT.md` for the design history, `FILE_GUIDE.md` for the source
inventory, then `pyproject.toml`, `requirements.txt` and
`requirements-espdl.txt` for the two Python environments.

## Python environment

The main experiments were written for Python 3.11. Install PyTorch with the
CUDA build appropriate for the host first, then install the remaining packages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -v
Push-Location host_receiver
python -m unittest discover -s tests -v
Pop-Location
```

ESP-DL conversion used a separate Python 3.10 environment because the tested
`esp-ppq` release has a narrower dependency set. The conversion requirements
include the common experiment requirements, so a clean conversion environment
can be prepared with:

```powershell
py -3.10 -m venv .venv-espdl
.venv-espdl\Scripts\Activate.ps1
python -m pip install -r requirements-espdl.txt
```

See `docs/REPRODUCIBILITY.md` for the ordered conversion procedure.

## Data and generated files

Official download pages, versions, access requirements, licences, citations and
the exact subset used are documented in [`docs/DATASETS.md`](docs/DATASETS.md)
and machine-readable in `configs/datasets.json`. Place locally obtained data
below `dataset/`, then check it with:

```powershell
python scripts/scan_datasets.py
python tools/restore_split_manifests.py
```

Raw datasets are not redistributed. The exact formal split metadata is tracked
under `reproducibility/splits/`; the second command restores it to the
`experiments/` paths expected by the original scripts. Newly generated
experiments, model artefacts and physical logs remain ignored by Git.

## Firmware and live demonstration

The submitted firmware was developed with ESP-IDF 5.5.5, ESP-DL 3.3.8 and an
8 MB flash layout. Open an ESP-IDF terminal, then build or flash either node:

```powershell
.\firmware\build_esp_idf.ps1 -Node asn -Action build
.\firmware\build_esp_idf.ps1 -Node asn -Action flash -Port COM8
.\firmware\build_esp_idf.ps1 -Node vsn -Action build
.\firmware\build_esp_idf.ps1 -Node vsn -Action flash -Port COM7
```

With ASN on `COM8` and VSN on `COM7`, start the local dashboard:

```powershell
.\host_receiver\run_dashboard.ps1 -VsnPort COM7 -AsnPort COM8
```

If the local Windows execution policy blocks project scripts, use a process-only
bypass rather than changing the machine-wide policy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\host_receiver\run_dashboard.ps1 -VsnPort COM7 -AsnPort COM8
```

The dashboard can switch between automatic and manual capture, request VSN,
ASN or coordinated samples, inspect node availability, display the VSN's
rule-based structured warning, and exercise model push/pull and MARH join/leave
operations. The PC records and renders events but does not replace the VSN's
online fusion or warning decision.

## Submission boundary

Pre-trained ESP-DL model assets and the ASN golden PCM vector are included
because they are required to build and parity-check the final firmware. Exact
split manifests are included because they contain only derived metadata needed
to reproduce the evaluation. No third-party dataset sample, build cache or
personal path is included. Detailed numerical results belong in the project
report rather than in this source submission.
