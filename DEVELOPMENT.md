# Development Record

This file gives a compact account of how the implementation evolved. It is not
an experiment report: numerical tables and figures are kept in the dissertation.
The intention is to make the relationship between research decisions and source
files visible to a marker.

## 1. From sensing concept to node interface

The first design question was how heterogeneous, intermittently connected nodes
could collaborate without streaming raw images, audio or high-rate time series.
The common interface was therefore limited to `node_id`, timestamp,
`risk_score`, confidence and status. `src/nlrisk/fusion/message_fusion.py`
implemented this contract before it was moved into firmware.

The score is intentionally a risk proxy. Sensor quality and node availability
are carried separately because a model score alone cannot indicate that a camera
is obscured or a microphone input is invalid.

## 2. Dataset indexing and leakage control

Initial VSN and ASN indices were built from the paired crack dataset and a clean
visual crack dataset. Later inspection found that random file-level splitting
was not sufficient for cropped or perceptually similar images. The data pipeline
was extended with exact hashes, perceptual similarity checks, parent-image
groups and aligned cross-modal splits.

The submission retains the exact relative-path manifests and split integrity
reports used by the formal experiments. Dataset source, version, licence,
citation, access and subset metadata are kept separately from local paths in a
portable registry; no third-party samples are redistributed.

Relevant files:

- `scripts/build_vsn_index.py`, `scripts/build_asn_index.py` and
  `scripts/build_vbn_orion_index.py`: initial indices;
- `scripts/audit_vsn_perceptual_leakage.py`: exact and near-duplicate audit;
- `scripts/build_codebrim_index.py` and `scripts/build_sdnet2018_index.py`:
  dataset-specific grouping;
- `scripts/build_conference_grouped_splits.py` and
  `scripts/build_conference_aligned_indices.py`: formal split construction.

## 3. Node model selection

Each modality was first treated independently so that failure could be traced to
a local model rather than the fusion rule. The VSN study compared conventional
transfer-learning baselines before introducing a small depthwise-separable
student. The ASN followed the same teacher/student pattern on log-Mel features.
The VBN compared time/frequency feature baselines with temporal neural models,
but remains a structural acoustic-emission proxy rather than a deployed IMU.

The final node design keeps a private modality encoder. This was preferred to a
single cross-modal network because the sensor frontends, sampling rates and
embedded operators are not interchangeable.

## 4. Robustness, calibration and abstention

Clean accuracy did not answer whether a node remained trustworthy after sensor
degradation. Controlled low-light, blur, image corruption, audio noise and
dropout tests were added, followed by held-out confidence calibration. The ASN
also gained a signal-validity model so invalid input can be marked before it
enters fusion. A later VSN study added multi-label damage and quality heads to
test whether multi-task learning improved severe-degradation behaviour without
substantially enlarging the private encoder.

This stage led to an important negative decision: a shared model layer does not
replace status communication. Parameter sharing transfers learned knowledge;
status and confidence describe the current observation.

## 5. Decision fusion

Pairwise VSN-ASN fusion was implemented before the three-node stress test. The
main rule is status-aware and confidence-weighted, with explicit single-node
fallback. Alternative max-risk and filtered rules were retained for ablation.
Fusion uses calibrated scores, and compressed-model scores are evaluated again
rather than assumed to match FP32 behaviour.

The VSN acts as the representative for external forwarding in the physical
prototype. This is a routing responsibility, not a permanent central learning
server.

The same node derives a fixed-schema warning from the fused risk level, trend,
available evidence and MARH recommendation. The warning contains a headline,
severity, evidence set and recommended action. It is deliberately rule based:
the external receiver displays the decision made on the VSN and does not use a
language model or silently reinterpret the risk score.

## 6. TinyML compression

Memory auditing was followed by student-from-scratch and knowledge-distillation
baselines. Post-training INT8 quantisation was evaluated before considering more
expensive quantisation-aware training. Export scripts check FP32/quantised
prediction parity and package deterministic test vectors for firmware.

The two-megabyte project constraint concerns working memory, not only the model
file. Firmware therefore records internal RAM, PSRAM and stage latency in
addition to static model size.

## 7. Higher-layer parameter sharing

The supervisor's multi-task and parameter-sharing suggestion changed the
software from independent classifiers into heterogeneous learning nodes. Each
private encoder maps its input into a compatible 64-dimensional risk space. A
small `64-32-1` higher risk head can then be shared without exchanging either
the raw data or the full modality network.

Eight sharing boundaries were explored, followed by multi-seed comparisons,
personalised variants, non-IID simulations and partial participation. A small
fully shared head was retained because extra private calibration did not provide
a consistent enough benefit to justify a more complex runtime contract.

Two collaboration paths are therefore kept:

- frequent 20-byte risk summaries for online decisions;
- infrequent compatible-head updates for collaborative learning.

## 8. Physical implementation

The final software was split across two XIAO ESP32-S3 Sense boards. The ASN is a
BLE peripheral and publishes each completed microphone inference with a
validity-derived status. The VSN is a BLE central, performs camera inference,
fuses fresh admissible ASN evidence, generates the structured warning and
forwards the event over USB serial. The receiver records those decisions and
provides a browser dashboard; it does not recompute the fused result or alert.

Model transfer was implemented as a versioned, chunked protocol with record
CRC, payload CRC, ordering checks, timeout, golden-vector validation, atomic
activation and rollback. Fault injection covers malformed, incomplete,
corrupted and interrupted transfers. Reconnection returns the VSN to dual-node
operation after ASN availability is restored.

## 9. Optional MARH teacher update

The PC can train a larger teacher over stored private-node representations,
distil it into the runtime-compatible shared head and apply an offline acceptance
gate. An accepted candidate is sent to the VSN over USB and relayed to the ASN
over BLE. The helper remains optional: local sensing and fusion continue without
it, and a recommendation to intervene is distinct from automatic takeover.

The physical update path includes post-installation sampling so that both nodes
report the activated version. Invalid or interrupted candidates remain staged
and cannot replace the last validated runtime head.

## Decisions deliberately not taken

- Raw multimodal streaming was not used because it contradicts the communication
  and privacy objective.
- A single shared low-level encoder was not used because physical modalities and
  embedded frontends are incompatible.
- Risk maps, VAE-based generation and synthetic crack generation were excluded
  from the final pipeline because they did not support the local risk-message
  contract.
- The VBN proxy is not presented as physical building-vibration validation.
- Software estimates are not substituted for measured hardware latency or
  runtime memory where physical measurements are available.
