"""Package the ASN validity model and golden vectors for firmware integration.

The export separates the desktop log-Mel frontend from the quantised feature
model and records deterministic tensors for checking that the firmware frontend
preserves the same input contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from train_asn_audio import PROJECT_ROOT, read_samples
from train_asn_signal_validity import AsnValidityModel, ValidityDataset


class ExportFeatureModel(nn.Module):
    """Feature-domain wrapper matching the tensor consumed by ESP-DL."""

    def __init__(self, source: AsnValidityModel) -> None:
        super().__init__()
        self.encoder = source.encoder
        self.risk_head = source.risk_head
        self.validity_head = source.validity_head

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(features)
        risk = self.risk_head(embedding)
        validity = self.validity_head(embedding)
        return torch.cat((risk, validity), dim=1)


def namespace(values: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate(formal_dir: Path) -> tuple[Path, dict[str, object]]:
    candidates = []
    for result_path in formal_dir.glob("asn_signal_validity_*_seed*/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        candidates.append((float(result["training"]["best_validation_selection_score"]), result_path.parent, result))
    if not candidates:
        raise RuntimeError("No formal ASN signal-validity runs found")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, run_dir, result = candidates[0]
    return run_dir, result


def load_model(checkpoint_path: Path) -> tuple[AsnValidityModel, dict[str, object], SimpleNamespace]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = namespace(checkpoint["args"])
    model = AsnValidityModel(
        int(args.sample_rate), int(args.n_fft), int(args.hop_length), int(args.n_mels), float(args.width), float(args.dropout)
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, args


def confidence(probability: float) -> float:
    probability = min(max(probability, 1e-7), 1 - 1e-7)
    entropy = -(probability * math.log2(probability) + (1 - probability) * math.log2(1 - probability))
    return 1 - entropy


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    values = values - values.max()
    exponent = np.exp(values)
    return exponent / exponent.sum()


def export_onnx(model: nn.Module, example: torch.Tensor, path: Path) -> dict[str, object]:
    torch.onnx.export(
        model,
        example,
        path,
        input_names=["logmel"],
        output_names=["node_logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    verification: dict[str, object] = {"onnx_exported": True, "onnxruntime_checked": False}
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        observed = session.run(None, {"logmel": example.numpy().astype(np.float32)})[0]
        with torch.no_grad():
            expected = model(example).numpy()
        verification.update(
            {
                "onnxruntime_checked": True,
                "max_abs_logit_error": float(np.max(np.abs(observed - expected))),
            }
        )
    except Exception as exc:  # noqa: BLE001
        verification["onnxruntime_error"] = str(exc)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_dir, formal_result = load_candidate(args.formal_dir)
    model, checkpoint, train_args = load_model(run_dir / "deployed_state_dict.pt")
    calibration = checkpoint["calibration"]
    index = Path(train_args.index)
    if not index.is_absolute():
        index = PROJECT_ROOT / index
    samples = read_samples(index, "val", None)
    negative = next(sample for sample in samples if sample.label == 0)
    positive = next(sample for sample in samples if sample.label == 1)
    vector_specs = (
        ("usable_clean_negative", negative, "clean", 0),
        ("degraded_snr10_positive", positive, "snr_noise", 2),
        ("invalid_dropout30_positive", positive, "dropout", 3),
    )

    export_model = ExportFeatureModel(model).eval()
    features_all = []
    logits_all = []
    pcm16_all = []
    vector_records = []
    with torch.no_grad():
        for name, sample, condition, severity in vector_specs:
            dataset = ValidityDataset(
                [sample], int(train_args.sample_rate), float(train_args.duration), int(train_args.seed), (condition, severity)
            )
            waveform, label, target_validity = dataset[0]
            pcm16 = torch.round(waveform.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
            reference_waveform = pcm16.to(torch.float32) / 32768.0
            features = model.frontend(reference_waveform.unsqueeze(0))
            logits = export_model(features)
            pcm16_all.append(pcm16.numpy()[None, ...])
            features_all.append(features.numpy().astype(np.float32))
            logits_all.append(logits.numpy().astype(np.float32))
            raw = logits.numpy().reshape(-1)
            risk_probability = 1.0 / (1.0 + math.exp(-float(raw[0]) / float(calibration["risk_temperature"])))
            validity_probabilities = softmax(raw[1:] / float(calibration["validity_temperature"]))
            validity_index = int(np.argmax(validity_probabilities))
            vector_records.append(
                {
                    "name": name,
                    "source_path": str(sample.path),
                    "label": int(label.item()),
                    "controlled_condition": condition,
                    "severity": severity,
                    "target_validity": int(target_validity.item()),
                    "expected_logits": raw.tolist(),
                    "expected_risk_score": risk_probability,
                    "expected_confidence": confidence(risk_probability),
                    "expected_validity_probabilities": validity_probabilities.tolist(),
                    "expected_validity_index": validity_index,
                    "expected_operational_status": "invalid" if validity_index == 2 else "accepted",
                }
            )

    feature_array = np.concatenate(features_all, axis=0)
    logits_array = np.concatenate(logits_all, axis=0)
    pcm16_array = np.concatenate(pcm16_all, axis=0)
    np.savez_compressed(
        args.output_dir / "golden_vectors.npz",
        pcm16=pcm16_array,
        logmel=feature_array,
        expected_logits=logits_array,
    )
    (args.output_dir / "golden_vectors.json").write_text(json.dumps(vector_records, indent=2), encoding="utf-8")
    np.save(args.output_dir / "onnx_test_input.npy", feature_array[:1])

    representative_conditions = (
        ("clean", 0),
        ("snr_noise", 1),
        ("snr_noise", 2),
        ("snr_noise", 3),
        ("clipping", 3),
        ("dropout", 1),
        ("dropout", 2),
        ("dropout", 3),
    )
    calibration_features = []
    with torch.no_grad():
        for sample_index, sample in enumerate(samples[:128]):
            condition, severity = representative_conditions[sample_index % len(representative_conditions)]
            dataset = ValidityDataset(
                [sample], int(train_args.sample_rate), float(train_args.duration), int(train_args.seed) + sample_index, (condition, severity)
            )
            waveform, _, _ = dataset[0]
            pcm16 = torch.round(waveform.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
            reference_waveform = pcm16.to(torch.float32) / 32768.0
            calibration_features.append(model.frontend(reference_waveform.unsqueeze(0)).numpy().astype(np.float32))
    np.save(args.output_dir / "espdl_calibration_logmel.npy", np.concatenate(calibration_features, axis=0))

    onnx_path = args.output_dir / "asn_signal_validity_feature_model_fp32.onnx"
    verification = export_onnx(export_model, torch.from_numpy(feature_array[:1]), onnx_path)
    channels = int(model.encoder.output_channels)
    params = sum(parameter.numel() for parameter in model.parameters())
    raw_audio_bytes = int(train_args.sample_rate * train_args.duration) * 2
    fp32_feature_bytes = int(np.prod(feature_array.shape[1:])) * 4
    int8_feature_bytes = int(np.prod(feature_array.shape[1:]))
    int8_weight_bytes = params
    manifest = {
        "artifact": "ASN pre-deployment package",
        "selection": {
            "rule": "highest validation selection score among the three locked formal seeds; test metrics were not used",
            "run_id": formal_result["run_id"],
            "seed": formal_result["protocol"]["seed"],
            "validation_selection_score": formal_result["training"]["best_validation_selection_score"],
        },
        "input_contract": {
            "microphone_pcm": "mono signed PCM16",
            "sample_rate_hz": int(train_args.sample_rate),
            "window_seconds": float(train_args.duration),
            "window_samples": int(train_args.sample_rate * train_args.duration),
            "n_fft": int(train_args.n_fft),
            "hop_length": int(train_args.hop_length),
            "n_mels": int(train_args.n_mels),
            "reference_frontend": "power mel spectrogram, AmplitudeToDB(power), per-window mean/std normalisation",
            "feature_shape_nchw": list(feature_array.shape[1:]),
        },
        "output_contract": {
            "shape": [1, 4],
            "order": ["risk_logit", "validity_usable_logit", "validity_degraded_logit", "validity_invalid_logit"],
            "risk_temperature": calibration["risk_temperature"],
            "validity_temperature": calibration["validity_temperature"],
            "risk_threshold": calibration["risk_threshold"],
            "operational_gate": "argmax(validity logits / temperature) == 2 means invalid; otherwise accepted",
            "note": "degraded remains diagnostic until physical microphone validation",
        },
        "model": {
            "shared_embedding_channels": channels,
            "parameters": params,
            "estimated_int8_weight_kib": int8_weight_bytes / 1024.0,
            "onnx_file": onnx_path.name,
            "onnx_sha256": sha256(onnx_path),
        },
        "static_buffer_budget": {
            "pcm16_window_kib": raw_audio_bytes / 1024.0,
            "fp32_logmel_kib": fp32_feature_bytes / 1024.0,
            "int8_logmel_kib": int8_feature_bytes / 1024.0,
            "int8_weights_kib": int8_weight_bytes / 1024.0,
            "scope": "Static input/feature/weight accounting only; tensor arena, DSP workspace, runtime, stack, and communication buffers require board profiling.",
        },
        "verification": verification,
        "claims_boundary": "No board latency, peak SRAM, Flash, or energy claim is made by this package.",
    }
    (args.output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    header = f"""#pragma once

// Generated ASN model contract. Do not hand-edit calibration constants.
#define ASN_SAMPLE_RATE_HZ {int(train_args.sample_rate)}
#define ASN_WINDOW_SAMPLES {int(train_args.sample_rate * train_args.duration)}
#define ASN_N_FFT {int(train_args.n_fft)}
#define ASN_HOP_LENGTH {int(train_args.hop_length)}
#define ASN_N_MELS {int(train_args.n_mels)}
#define ASN_FEATURE_FRAMES {int(feature_array.shape[-1])}
#define ASN_OUTPUT_COUNT 4
#define ASN_RISK_TEMPERATURE {float(calibration['risk_temperature']):.9f}f
#define ASN_VALIDITY_TEMPERATURE {float(calibration['validity_temperature']):.9f}f
#define ASN_RISK_THRESHOLD {float(calibration['risk_threshold']):.9f}f
#define ASN_VALIDITY_INVALID_INDEX 2
"""
    (args.output_dir / "asn_model_contract.h").write_text(header, encoding="utf-8")
    readme = f"""# ASN Signal-Validity Pre-deployment Package

Selected candidate: `{formal_result['run_id']}` using the locked validation score only.

## Package contents

- `asn_signal_validity_feature_model_fp32.onnx`: fixed-shape CNN and dual heads.
- `golden_vectors.npz`: three PCM16 windows, corresponding log-mel tensors, and expected raw logits.
- `golden_vectors.json`: calibrated semantic expectations for readable parity checks.
- `onnx_test_input.npy`: one tensor for ESP-DL conversion smoke testing.
- `espdl_calibration_logmel.npy`: 128 validation-derived representative tensors for INT8 calibration.
- `model_manifest.json`: signal, model, calibration, memory, and claims contracts.
- `asn_model_contract.h`: constants required by firmware post-processing.

## Board boundary

The board implementation must reproduce the reference log-mel frontend before model parity can be claimed. First compare raw logits against the golden vectors, then connect live microphone capture. Do not tune thresholds on hardware test observations.

This directory is pre-deployment evidence. It contains no ESP32-S3 latency, peak SRAM, Flash, or energy measurement.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"selected_run": formal_result["run_id"], "onnx": str(onnx_path), "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
