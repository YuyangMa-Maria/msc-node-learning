# File Guide

This is an exhaustive guide to the submitted source tree. Files with the same
name under both firmware targets are listed together where they implement the
same wire contract.

## Repository-level files

| File | Purpose |
|---|---|
| `.gitignore` | Prevents datasets, generated results, firmware builds and local environments entering the submission. |
| `README.md` | Project overview, architecture, setup, build and live-demonstration entry point. |
| `DEVELOPMENT.md` | Chronological explanation of design decisions and implementation stages. |
| `FILE_GUIDE.md` | This source inventory. |
| `pyproject.toml` | Installable `nlrisk` package metadata and source-package discovery settings. |
| `requirements.txt` | Main experiment, export and host-side Python dependencies. |
| `requirements-espdl.txt` | Python 3.10 ESP-DL conversion environment, including the common requirements. |
| `configs/datasets.json` | Portable catalogue of the five datasets used by the submitted pipeline. |
| `reproducibility/splits/` | Exact relative-path split manifests and integrity reports used by the formal experiments. |

## Reusable Python package

| File | Purpose |
|---|---|
| `src/nlrisk/__init__.py` | Package declaration. |
| `src/nlrisk/data/__init__.py` | Data subpackage declaration. |
| `src/nlrisk/data/registry.py` | Loads typed dataset entries from the JSON catalogue. |
| `src/nlrisk/fusion/__init__.py` | Exposes the fusion API. |
| `src/nlrisk/fusion/message_fusion.py` | Reference status-aware fusion, risk-level, trend and structured warning policy. |

## Dataset preparation

| File | Purpose |
|---|---|
| `scripts/scan_datasets.py` | Checks which registered datasets are available locally and summarises their files. |
| `scripts/build_vsn_index.py` | Builds the initial binary VSN image index. |
| `scripts/build_asn_index.py` | Builds the paired ASN audio index. |
| `scripts/build_vbn_orion_index.py` | Converts Orion file names and torque conditions into a VBN proxy manifest. |
| `scripts/build_codebrim_index.py` | Parses CODEBRIM XML labels and creates parent-grouped binary/multi-label indices. |
| `scripts/build_sdnet2018_index.py` | Indexes SDNET2018 and joins exact duplicate and parent-image groups. |
| `scripts/audit_vsn_perceptual_leakage.py` | Finds exact and perceptually similar image leakage across proposed splits. |
| `scripts/build_conference_grouped_splits.py` | Creates the group-disjoint formal train/validation/test splits. |
| `scripts/build_conference_aligned_indices.py` | Aligns paired VSN and ASN records to the same formal split assignment. |

## Node training and model selection

| File | Purpose |
|---|---|
| `scripts/train_vsn_binary.py` | Trains and compares the initial binary visual transfer-learning models. |
| `scripts/train_vsn_student_baseline.py` | Trains the compact depthwise-separable VSN student from scratch. |
| `scripts/train_vsn_student_kd.py` | Trains the same VSN student with teacher soft targets. |
| `scripts/train_vsn_codebrim_teachers.py` | Trains larger CODEBRIM teacher models for binary and defect-label tasks. |
| `scripts/train_vsn_codebrim_students.py` | Trains compact CODEBRIM students and distillation variants. |
| `scripts/train_vsn_codebrim_deployable_multitask.py` | Trains the final lightweight damage, defect-type and quality multi-task candidate. |
| `scripts/train_vsn_sdnet2018_transfer.py` | Runs SDNET2018 fine-tuning and mixed-domain transfer experiments. |
| `scripts/train_asn_audio.py` | Trains the initial acoustic risk-model candidates on log-Mel input. |
| `scripts/train_asn_student.py` | Trains compact ASN students from scratch and with distillation. |
| `scripts/train_asn_signal_validity.py` | Trains the auxiliary clean/degraded/invalid acoustic-input classifier. |
| `scripts/train_vbn_orion.py` | Trains the initial neural VBN proxy model. |
| `scripts/evaluate_vbn_literature_guided_baselines.py` | Compares statistical/time-frequency features, shallow classifiers and temporal neural baselines. |

## Calibration, robustness and compression

| File | Purpose |
|---|---|
| `scripts/calibrate_vsn_student_risk.py` | Fits and evaluates held-out calibration for VSN risk scores. |
| `scripts/calibrate_asn_student_risk.py` | Fits and evaluates held-out calibration for ASN risk scores. |
| `scripts/evaluate_vsn_student_robustness.py` | Applies controlled low-light, blur and corruption to the FP32 VSN student. |
| `scripts/evaluate_vsn_ptq_robustness.py` | Compares VSN FP32 and PTQ behaviour under the same corruptions. |
| `scripts/evaluate_vsn_codebrim_ptq_robustness.py` | Evaluates quantisation and degradation for the multi-task VSN extension. |
| `scripts/evaluate_vsn_sdnet_zero_shot.py` | Measures external-domain VSN performance before adaptation. |
| `scripts/evaluate_asn_robustness.py` | Applies SNR noise and temporal dropout to ASN inputs. |
| `scripts/evaluate_asn_ptq_robustness.py` | Checks FP32/INT8 ASN robustness consistency. |
| `scripts/evaluate_asn_signal_validity_ptq.py` | Checks quantised validity-gate performance and parity. |
| `scripts/evaluate_tinyml_compression_sweep.py` | Sweeps compact architectures and compression combinations under the memory constraint. |
| `scripts/tinyml_memory_audit.py` | Estimates model parameters, activations and working-memory requirements. |
| `scripts/export_vsn_student_ptq.py` | Produces the PTQ VSN export and prediction-parity artefacts. |
| `scripts/export_asn_student_ptq.py` | Produces the PTQ ASN export and prediction-parity artefacts. |
| `scripts/export_vsn_mcu_onnx.py` | Exports a static-shape VSN graph and deterministic parity vectors. |
| `scripts/quantize_vsn_espdl.py` | Converts the VSN ONNX graph into an ESP-DL INT8 model. |
| `scripts/package_asn_signal_validity_predeploy.py` | Packages ASN validity exports and golden inputs for firmware work. |

## Fusion, alerting and workflow simulation

| File | Purpose |
|---|---|
| `scripts/evaluate_vsn_asn_fusion.py` | Compares two-node score-fusion rules on paired VSN/ASN records. |
| `scripts/evaluate_vsn_asn_vbn_fusion.py` | Runs the label-aligned three-node fusion stress test. |
| `scripts/evaluate_compressed_fusion.py` | Repeats fusion with calibrated compressed-node predictions. |
| `scripts/evaluate_asn_status_gated_fusion.py` | Measures the effect of ASN validity/status gating on fused decisions. |
| `scripts/simulate_message_fusion.py` | Small scenario driver for the reusable message-fusion API. |
| `scripts/simulate_representative_alert_marh.py` | Simulates risk trend, representative alerts and optional helper intervention. |

## Higher-layer sharing and federated simulation

| File | Purpose |
|---|---|
| `scripts/evaluate_parameter_sharing_cnn_embeddings.py` | Produces private CNN embeddings and learns a compatible shared risk head. |
| `scripts/evaluate_sharing_candidate_layers.py` | Sweeps eight possible sharing boundaries. |
| `scripts/evaluate_final_sharing_candidates.py` | Repeats the selected boundaries across formal grouped data. |
| `scripts/evaluate_shared_layer_robustness.py` | Tests whether shared candidates remain reliable under sensor degradation. |
| `scripts/evaluate_shared_head_federated_update.py` | Simulates round-based updates of only the shared higher layer. |
| `scripts/evaluate_shared_head_federated_dropout.py` | Adds rotating, intermittent and one-node participation schedules. |
| `scripts/evaluate_personalized_federated_sharing.py` | Compares a global head with small node-specific calibrators. |
| `scripts/evaluate_conference_statistical_validation.py` | Runs the final multi-seed confidence-interval and significance analysis. |

## Deployment preparation and MARH tools

| File | Purpose |
|---|---|
| `tools/prepare_runtime_shared_models.py` | Splits private encoders from the compatible 64-D shared-head interface and exports ONNX graphs. |
| `tools/quantize_runtime_shared_models.py` | Quantises private representations and checks shared-head output parity. |
| `tools/evaluate_asn_mixed_precision_shared_model.py` | Evaluates the ASN mixed-precision representation used by the physical firmware. |
| `tools/export_shared_head_payloads.py` | Converts trained shared heads into versioned FP32/INT8 C++ payloads. |
| `tools/train_marh_teacher_update.py` | Trains a PC teacher, distils a candidate runtime head and applies the offline deployment gate. |
| `tools/restore_split_manifests.py` | Restores the tracked formal manifests to the `experiments/` paths expected by historical scripts. |

## Firmware build and shared contracts

| File | Purpose |
|---|---|
| `firmware/build_esp_idf.ps1` | Portable build, flash and monitor wrapper for either ESP-IDF target. |
| `firmware/{vsn,asn}/CMakeLists.txt` | Declares each ESP-IDF project. |
| `firmware/{vsn,asn}/sdkconfig.defaults` | Reproducible ESP32-S3, PSRAM, USB console and NimBLE defaults. |
| `firmware/{vsn,asn}/partitions.csv` | 8 MB flash layouts for applications, models and golden data. |
| `firmware/{vsn,asn}/main/idf_component.yml` | ESP-IDF component dependencies; the VSN additionally requests camera and NimBLE central utilities. |
| `firmware/{vsn,asn}/main/model_exchange_protocol.h` | Identical packed model-control, chunk and status records used on both boards. |
| `firmware/{vsn,asn}/main/node_summary_protocol.h` | Identical packed 20-byte node-summary record and CRC helpers. |
| `firmware/{vsn,asn}/main/sampling_control_protocol.h` | Identical coordinated-sampling command record. |
| `firmware/{vsn,asn}/main/shared_head_payloads.h` | Generated, versioned base/federated shared-head arrays and golden outputs. |

## ASN firmware

| File | Purpose |
|---|---|
| `firmware/asn/main/CMakeLists.txt` | Builds the live ASN and flashes encoder, validity and golden partitions. |
| `firmware/asn/main/asn_live.cpp` | Final microphone-to-summary runtime, including log-Mel extraction, two models and memory/latency logging. |
| `firmware/asn/main/asn_ble_server.cpp` | BLE peripheral, GATT characteristics, summary notification and guarded model receiver. |
| `firmware/asn/main/asn_ble_server.h` | BLE service API used by the live runtime. |
| `firmware/asn/main/asn_sampling_control.cpp` | Synchronises automatic and host-triggered five-second captures. |
| `firmware/asn/main/asn_sampling_control.h` | ASN sampling-control declarations. |
| `firmware/asn/main/runtime_shared_head.cpp` | Evaluates, validates, atomically replaces and rolls back the ASN higher layer. |
| `firmware/asn/main/runtime_shared_head.h` | Shared-head snapshots, transitions and public runtime API. |
| `firmware/asn/main/asn_model_contract.h` | Audio window, log-Mel and model tensor constants. |
| `firmware/asn/main/asn_bringup.cpp` | Retained early diagnostic for PDM capture, model parity and benchmarking; not in the final target. |
| `firmware/asn/tools/compare_frontends.py` | Compares desktop and ESP-DL log-Mel tensors before firmware integration. |
| `firmware/asn/main/models/asn_private_encoder_projection64_esp32s3_int16.espdl` | Final private ASN encoder/projection model flashed by CMake. |
| `firmware/asn/main/models/asn_signal_validity_seed99_esp32s3_int8.espdl` | Final acoustic validity model flashed by CMake. |
| `firmware/asn/main/golden/golden_pcm16.bin` | Deterministic PCM data used for board-side frontend and model parity checks. |

## VSN firmware

| File | Purpose |
|---|---|
| `firmware/vsn/main/CMakeLists.txt` | Builds the representative VSN and flashes its private encoder/projection. |
| `firmware/vsn/main/vsn_camera_bringup.c` | Application entry point, OV3660 camera initialisation and runtime start-up. |
| `firmware/vsn/main/vsn_model_parity.cpp` | Image preprocessing, INT8 inference, status-aware VSN-ASN fusion, trend and alerts. |
| `firmware/vsn/main/vsn_model_runtime.h` | Camera/model runtime boundary shared by C and C++ sources. |
| `firmware/vsn/main/vsn_ble_client.cpp` | BLE central discovery, summary validation, acknowledgements and reconnection. |
| `firmware/vsn/main/vsn_ble_client.h` | VSN BLE state, snapshot and control API. |
| `firmware/vsn/main/vsn_sampling_control.cpp` | Coordinates manual visual capture after a fresh ASN summary. |
| `firmware/vsn/main/vsn_sampling_control.h` | VSN sampling-control declarations. |
| `firmware/vsn/main/vsn_model_exchange.cpp` | Push/pull state machine plus corruption, ordering, timeout and rollback fault tests. |
| `firmware/vsn/main/vsn_model_exchange.h` | Model-exchange commands exposed to the host protocol. |
| `firmware/vsn/main/runtime_shared_head.cpp` | Evaluates, validates, atomically replaces and rolls back the VSN higher layer. |
| `firmware/vsn/main/runtime_shared_head.h` | Shared-head snapshots, transitions and public runtime API. |
| `firmware/vsn/main/host_model_update.cpp` | Receives a PC/MARH candidate in serial chunks and stages it for activation/relay. |
| `firmware/vsn/main/host_model_update.h` | MARH candidate staging declarations. |
| `firmware/vsn/main/system_host_protocol.cpp` | Parses manual commands and emits schema-versioned `NLJSON` events, including VSN-generated structured warnings. |
| `firmware/vsn/main/system_host_protocol.h` | Structured host-reporting API used by VSN subsystems. |
| `firmware/vsn/main/models/vsn_private_encoder_projection64_esp32s3_int8.espdl` | Final private VSN encoder/projection model flashed by CMake. |

## External receiver and dashboard

| File | Purpose |
|---|---|
| `host_receiver/protocol.py` | Validates `NLJSON` events, maintains receiver state and writes fusion CSV rows. |
| `host_receiver/receiver.py` | Captures VSN serial output, sends commands and records raw/structured run data. |
| `host_receiver/demo_dashboard.py` | Hosts the local dashboard and bridges browser commands to the VSN serial link. |
| `host_receiver/dashboard_static/index.html` | Dashboard information hierarchy and controls. |
| `host_receiver/dashboard_static/styles.css` | Responsive, restrained styling for the physical demonstration. |
| `host_receiver/dashboard_static/app.js` | Live event stream, node/fusion rendering, warning presentation and command interactions. |
| `host_receiver/run_receiver.ps1` | Convenience launcher for the terminal receiver. |
| `host_receiver/run_dashboard.ps1` | Convenience launcher for the browser dashboard. |
| `host_receiver/requirements.txt` | Minimal dependency list for a host-only installation. |
| `host_receiver/extract_detailed_run.py` | Extracts selected node, fusion and model events from a physical run. |
| `host_receiver/summarise_run.py` | Calculates availability, outage and recovery summaries from recorded events. |
| `host_receiver/upload_teacher_update.py` | Sends an accepted PC teacher update to VSN and verifies BLE relay to ASN. |
| `host_receiver/plans/manual_sampling_plan.json` | Timed switch to manual mode and coordinated node sampling. |
| `host_receiver/plans/marh_model_exchange_plan.json` | Timed MARH join, model push/pull and leave sequence. |
| `host_receiver/plans/full_chain_plan.json` | Combined end-to-end command sequence for a reproducible demonstration run. |
| `host_receiver/tests/test_protocol.py` | Unit tests for event parsing, state updates and command-plan validation. |
| `host_receiver/tests/test_summary.py` | Unit test for BLE disconnect/reconnect timing calculations. |

## Documentation and tests

| File | Purpose |
|---|---|
| `docs/WIRE_PROTOCOL.md` | Fast decision path, slow model-update path, host events and command semantics. |
| `docs/DATASETS.md` | Official sources, licences, citations, access notes, subsets, local layouts and validity boundaries. |
| `docs/DATASET_CITATIONS.bib` | BibTeX records for every dataset and its recommended companion publication. |
| `docs/REPRODUCIBILITY.md` | Ordered host, training, conversion and firmware reproduction procedure. |
| `reproducibility/splits/README.md` | Split policies, file roles, counts and known limitations. |
| `reproducibility/splits/conference_grouped/vsn_binary_index_grouped.csv` | Exact formal VSN paths, labels, sources and grouped split assignments. |
| `reproducibility/splits/conference_grouped/asn_audio_index_grouped.csv` | Exact formal ASN paths and split assignments inherited from paired VSN groups. |
| `reproducibility/splits/conference_grouped/vsn_group_manifest.csv` | SHA-256/pHash groups, canonical paths and duplicate provenance for the formal VSN split. |
| `reproducibility/splits/conference_grouped/{grouped_split_report.json,CONFERENCE_GROUPED_SPLIT_REPORT.md}` | Machine-readable and concise summaries of the formal grouped split. |
| `reproducibility/splits/codebrim/codebrim_multitask_index.csv` | Official-split CODEBRIM paths, parent groups, multi-label targets and hashes. |
| `reproducibility/splits/codebrim/codebrim_all_zero_annotation_audit.csv` | Annotation records excluded because every supplied label was zero. |
| `reproducibility/splits/codebrim/{codebrim_index_report.json,CODEBRIM_INDEX_REPORT.md}` | CODEBRIM integrity and class-distribution reports. |
| `reproducibility/splits/sdnet2018/sdnet2018_index_grouped.csv` | Frozen SDNET2018 paths, labels, parent groups, hashes and split assignments. |
| `reproducibility/splits/sdnet2018/sdnet2018_group_manifest.csv` | Component-level SDNET2018 split and surface/label counts. |
| `reproducibility/splits/sdnet2018/sdnet2018_label_conflicts.csv` | Four files excluded because identical content had conflicting labels. |
| `reproducibility/splits/sdnet2018/{sdnet2018_split_report.json,SDNET2018_SPLIT_REPORT.md}` | SDNET2018 grouping, distribution and sanity reports. |
| `reproducibility/splits/vbn_orion/vbn_orion_index.csv` | Exact ORION-AE proxy subset, labels and deterministic 60/20/20 assignments. |
| `reproducibility/splits/vbn_orion/vbn_orion_split_report.json` | VBN subset counts, binary mapping and evidence limitations. |
| `tests/test_dataset_registry.py` | Ensures every dataset has a source, version, citation, licence and tracked split set. |
| `tests/test_message_fusion.py` | Behavioural tests for offline gating, critical evidence, trend and risk bands. |
| `tests/test_repository_hygiene.py` | Guards against personal paths and committed generated-result directories. |
