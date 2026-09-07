# Reproduction Guide

The project is organised as a sequence of small, inspectable programs rather
than one opaque training command. Run `python <script> --help` before a long job;
most scripts expose output paths, device selection, seeds and sample limits.

## 1. Obtain and verify datasets

Download the five external datasets from the authoritative links in
`docs/DATASETS.md`, observe their individual licences, and extract them under
`dataset/`. Then run:

```powershell
python scripts/scan_datasets.py
```

The inventory includes official source links, versions, licence identifiers and
local file counts. Raw datasets are not part of this repository.

## 2. Restore the exact formal splits

For an exact reproduction of the reported grouped experiments, restore the
tracked relative-path manifests:

```powershell
python tools/restore_split_manifests.py
```

This populates the original `experiments/` locations without regenerating split
assignments. The tracked manifests and their reports are documented in
`reproducibility/splits/README.md`.

To audit the construction procedure or intentionally generate the same splits
from raw indices, run the following instead:

```powershell
python scripts/build_vsn_index.py
python scripts/build_asn_index.py
python scripts/build_vbn_orion_index.py
python scripts/audit_vsn_perceptual_leakage.py
python scripts/build_conference_grouped_splits.py
python scripts/build_conference_aligned_indices.py
python scripts/build_codebrim_index.py
python scripts/build_sdnet2018_index.py
```

The formal grouped builder uses seed 2026 and a pHash Hamming-distance threshold
of four. Compare regenerated files with `reproducibility/splits/` before
training. Do not tune models on any frozen test split.

## 3. Reproduce node baselines

```powershell
python scripts/train_vsn_binary.py
python scripts/train_asn_audio.py
python scripts/train_vbn_orion.py
python scripts/evaluate_vbn_literature_guided_baselines.py
```

These establish modality-specific baselines. The later grouped scripts should be
used for formal numerical reporting.

## 4. Train and calibrate TinyML students

```powershell
python scripts/train_vsn_student_baseline.py
python scripts/train_vsn_student_kd.py
python scripts/calibrate_vsn_student_risk.py
python scripts/train_asn_student.py
python scripts/calibrate_asn_student_risk.py
python scripts/train_asn_signal_validity.py
```

Then run the corresponding `evaluate_*robustness.py` scripts before export.
Calibration parameters are fitted on validation predictions only.

## 5. Quantisation and fusion

```powershell
python scripts/export_vsn_student_ptq.py
python scripts/export_asn_student_ptq.py
python scripts/evaluate_vsn_ptq_robustness.py
python scripts/evaluate_asn_ptq_robustness.py
python scripts/evaluate_compressed_fusion.py
```

`scripts/evaluate_vsn_asn_fusion.py` is the paired two-node study.
`scripts/evaluate_vsn_asn_vbn_fusion.py` is a label-aligned three-node stress
test and must not be described as synchronous physical sensing.

## 6. Multi-task and parameter-sharing studies

Run the CODEBRIM teacher, student and deployable multi-task scripts for the VSN
extension. The high-layer sharing sequence is:

```text
evaluate_parameter_sharing_cnn_embeddings.py
  -> evaluate_sharing_candidate_layers.py
  -> evaluate_final_sharing_candidates.py
  -> evaluate_shared_layer_robustness.py
  -> evaluate_shared_head_federated_update.py
  -> evaluate_shared_head_federated_dropout.py
  -> evaluate_personalized_federated_sharing.py
  -> evaluate_conference_statistical_validation.py
```

The shared module expects private VSN and ASN encoders to emit compatible 64-D
representations. It does not make their low-level encoders interchangeable.

## 7. ESP-DL conversion

Create a Python 3.10 conversion environment and install
`requirements-espdl.txt`; it includes the common requirements from
`requirements.txt`:

```powershell
py -3.10 -m venv .venv-espdl
.venv-espdl\Scripts\Activate.ps1
python -m pip install -r requirements-espdl.txt
```

`scripts/export_vsn_mcu_onnx.py`,
`scripts/quantize_vsn_espdl.py`, `tools/prepare_runtime_shared_models.py` and
`tools/quantize_runtime_shared_models.py` produce the deployable private
encoders and shared-head packages. `tools/export_shared_head_payloads.py`
generates `shared_head_payloads.h` for both firmware images.

The final `.espdl` files and ASN golden PCM vector are already included so the
submitted firmware can be built without rerunning training.

## 8. Build and test the physical system

Use ESP-IDF 5.5.5 or a compatible 5.x installation with ESP-DL 3.3.8. Build from
an ESP-IDF terminal:

```powershell
.\firmware\build_esp_idf.ps1 -Node asn -Action flash -Port COM8
.\firmware\build_esp_idf.ps1 -Node vsn -Action flash -Port COM7
.\host_receiver\run_receiver.ps1 -VsnPort COM7 -AsnDebugPort COM8
```

For a visual demonstration, replace the final command with
`host_receiver/run_dashboard.ps1`. Timed command plans under
`host_receiver/plans/` reproduce normal sampling, MARH/model exchange and the
full-chain control sequence. Run data are written below `runs/hardware/`.

## 9. Tests

```powershell
python -m unittest discover -s tests -v
Push-Location host_receiver
python -m unittest discover -s tests -v
Pop-Location
python -m compileall -q src scripts tools host_receiver firmware/asn/tools
```

Firmware builds provide the C/C++ compile check. Physical sensor and BLE tests
require two connected XIAO ESP32-S3 Sense boards and therefore are not part of
the host-only unit suite.
