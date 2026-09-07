"""Evaluate an INT8 ASN encoder with a native FP32 private projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.executor import TorchExecutor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_balanced(samples: list[Any], total: int, seed: int) -> list[Any]:
    rng = np.random.default_rng(seed)
    result: list[Any] = []
    for label in (0, 1):
        candidates = [sample for sample in samples if int(sample.label) == label]
        count = min(total // 2, len(candidates))
        indices = rng.choice(len(candidates), size=count, replace=False)
        result.extend(candidates[int(index)] for index in indices)
    result.sort(key=lambda sample: (int(sample.label), str(sample.path)))
    return result


def folded_projection(
    formal_state: dict[str, torch.Tensor], scaler: StandardScaler
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weight1 = formal_state["projections.asn.0.weight"].detach().cpu().numpy().astype(np.float32)
    bias1 = formal_state["projections.asn.0.bias"].detach().cpu().numpy().astype(np.float32)
    weight2 = formal_state["projections.asn.3.weight"].detach().cpu().numpy().astype(np.float32)
    bias2 = formal_state["projections.asn.3.bias"].detach().cpu().numpy().astype(np.float32)
    weight1 = weight1 / scaler.scale_.astype(np.float32)[None, :]
    bias1 = bias1 - weight1 @ scaler.mean_.astype(np.float32)
    return weight1, bias1, weight2, bias2


def project(
    embeddings: np.ndarray,
    parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    weight1, bias1, weight2, bias2 = parameters
    hidden = np.maximum(embeddings @ weight1.T + bias1, 0.0)
    return np.maximum(hidden @ weight2.T + bias2, 0.0).astype(np.float32)


def run_head(representations: np.ndarray, state: dict[str, torch.Tensor]) -> np.ndarray:
    weight1 = state["0.weight"].detach().cpu().numpy()
    bias1 = state["0.bias"].detach().cpu().numpy()
    weight2 = state["3.weight"].detach().cpu().numpy()
    bias2 = state["3.bias"].detach().cpu().numpy()
    hidden = np.maximum(representations @ weight1.T + bias1, 0.0)
    return (hidden @ weight2.T + bias2).reshape(-1)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def format_float_array(name: str, values: np.ndarray) -> str:
    flat = values.reshape(-1)
    rows = []
    for offset in range(0, len(flat), 8):
        rows.append("    " + ", ".join(f"{float(value):.9e}f" for value in flat[offset : offset + 8]) + ",")
    return f"inline constexpr float {name}[{len(flat)}] = {{\n" + "\n".join(rows) + "\n};\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    prepared_dir = args.prepared_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root / "scripts"))
    from train_asn_audio import read_samples
    from train_asn_student import AsnStudentDataset, AsnStudentModel

    device = torch.device(args.device)
    checkpoint = torch.load(
        project_root / "outputs/conference_grouped_asn_student/asn_student_w0p5_d5p0s_scratch/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = vars(checkpoint_args)
    student = AsnStudentModel(
        sample_rate=int(checkpoint_args.get("sample_rate", 16000)),
        n_fft=int(checkpoint_args.get("n_fft", 1024)),
        hop_length=int(checkpoint_args.get("hop_length", 320)),
        n_mels=int(checkpoint_args.get("n_mels", 64)),
        width=float(checkpoint_args.get("width", 0.5)),
        dropout=float(checkpoint_args.get("dropout", 0.15)),
    )
    student.load_state_dict(checkpoint["model"])
    student.to(device).eval()

    index_path = project_root / "experiments/conference_grouped/asn_audio_index_grouped.csv"
    train_samples = read_samples(index_path, "train", None)
    test_samples = read_samples(index_path, "test", None)
    calibration_samples = select_balanced(train_samples, args.calibration_samples, 2040)

    def features(samples: list[Any], seed: int) -> list[tuple[torch.Tensor, int, str]]:
        dataset = AsnStudentDataset(
            samples,
            sample_rate=int(checkpoint_args.get("sample_rate", 16000)),
            student_duration=float(checkpoint_args.get("duration", 5.0)),
            teacher_duration=float(checkpoint_args.get("teacher_duration", 10.0)),
            train=False,
            seed=seed,
        )
        rows = []
        with torch.no_grad():
            for index, (waveform, _teacher, label) in enumerate(dataset):
                log_mel = student.frontend(waveform.unsqueeze(0).to(device)).detach().cpu()
                rows.append((log_mel, int(label.item()), str(samples[index].path)))
        return rows

    calibration_rows = features(calibration_samples, 2041)
    test_rows = features(test_samples, 2042)
    calibration = [row[0] for row in calibration_rows]

    source_onnx = prepared_dir / "onnx/asn_private_encoder48_fp32.onnx"
    espdl_path = output_dir / "asn_private_encoder48_esp32s3_int8.espdl"
    graph = espdl_quantize_onnx(
        onnx_import_file=str(source_onnx),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calibration,
        calib_steps=len(calibration),
        input_shape=list(calibration[0].shape),
        inputs=[calibration[0].to(device)],
        target="esp32s3",
        num_of_bits=8,
        device=str(device),
        error_report=True,
        export_config=True,
        export_test_values=True,
        verbose=1,
        metadata_props={
            "project": "post-disaster-node-learning",
            "node": "ASN",
            "stage": "runtime-shared-head-mixed-precision",
            "output": "48-D private embedding",
        },
    )
    executor = TorchExecutor(graph=graph, device=str(device))
    session = ort.InferenceSession(str(source_onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    fp32_embeddings = []
    int8_embeddings = []
    labels = []
    paths = []
    for index, (tensor, label, path) in enumerate(test_rows, start=1):
        array = tensor.numpy().astype(np.float32, copy=False)
        fp32_embeddings.append(session.run([output_name], {input_name: array})[0].reshape(-1))
        int8_embeddings.append(executor.forward(inputs=[tensor.to(device)])[0].detach().cpu().numpy().reshape(-1))
        labels.append(label)
        paths.append(path)
        if index % 100 == 0 or index == len(test_rows):
            print(f"[ASN mixed] evaluated {index}/{len(test_rows)}")
    fp32_embedding = np.asarray(fp32_embeddings, dtype=np.float32)
    int8_embedding = np.asarray(int8_embeddings, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)

    cache = np.load(
        project_root / "outputs/conference_grouped_parameter_sharing_models/embedding_cache/asn_cnn_embeddings_t1600_e500_s42.npz",
        allow_pickle=True,
    )
    scaler = StandardScaler().fit(cache["train_x"].astype(np.float32))
    formal = torch.load(
        project_root / "outputs/conference_grouped_parameter_sharing_models/shared_risk_head.pt",
        map_location="cpu",
        weights_only=False,
    )
    federated = torch.load(
        project_root / "outputs/conference_grouped_shared_head_federated_update/final_shared_head_states.pt",
        map_location="cpu",
        weights_only=False,
    )
    projection_parameters = folded_projection(formal["model_state"], scaler)
    fp32_representation = project(fp32_embedding, projection_parameters)
    int8_representation = project(int8_embedding, projection_parameters)
    heads = {
        "base_shared_head": {
            key.removeprefix("shared_head."): value
            for key, value in formal["model_state"].items()
            if key.startswith("shared_head.")
        },
        "federated_updated_head": federated["final_shared_head_states"]["continual_pretrained_sample_weighted"],
    }
    thresholds = {"base_shared_head": 0.84, "federated_updated_head": 0.945}
    head_results: dict[str, Any] = {}
    for name, state in heads.items():
        fp32_probability = sigmoid(run_head(fp32_representation, state))
        int8_probability = sigmoid(run_head(int8_representation, state))
        threshold = thresholds[name]
        fp32_prediction = fp32_probability >= threshold
        int8_prediction = int8_probability >= threshold
        head_results[name] = {
            "threshold": threshold,
            "fp32_f1": float(f1_score(labels_array, fp32_prediction, zero_division=0)),
            "int8_encoder_fp32_projection_f1": float(f1_score(labels_array, int8_prediction, zero_division=0)),
            "f1_delta": float(
                f1_score(labels_array, int8_prediction, zero_division=0)
                - f1_score(labels_array, fp32_prediction, zero_division=0)
            ),
            "fp32_accuracy": float(accuracy_score(labels_array, fp32_prediction)),
            "int8_encoder_fp32_projection_accuracy": float(accuracy_score(labels_array, int8_prediction)),
            "prediction_disagreement_rate": float(np.mean(fp32_prediction != int8_prediction)),
            "risk_score_mae": float(np.mean(np.abs(fp32_probability - int8_probability))),
        }

    denominator = np.linalg.norm(fp32_embedding, axis=1) * np.linalg.norm(int8_embedding, axis=1)
    embedding_cosine = np.sum(fp32_embedding * int8_embedding, axis=1) / np.maximum(denominator, 1.0e-12)
    denominator = np.linalg.norm(fp32_representation, axis=1) * np.linalg.norm(int8_representation, axis=1)
    representation_cosine = np.sum(fp32_representation * int8_representation, axis=1) / np.maximum(denominator, 1.0e-12)
    summary = {
        "stage": "ASN mixed-precision runtime-sharing optimisation",
        "design": "INT8 private CNN encoder (48-D) + native FP32 folded private projection (48-96-64)",
        "samples": len(labels_array),
        "espdl": str(espdl_path),
        "espdl_sha256": sha256(espdl_path),
        "espdl_bytes": espdl_path.stat().st_size,
        "private_projection_parameters": int(sum(array.size for array in projection_parameters)),
        "private_projection_fp32_bytes": int(sum(array.nbytes for array in projection_parameters)),
        "encoder_embedding": {
            "mae": float(np.mean(np.abs(fp32_embedding - int8_embedding))),
            "mean_cosine_similarity": float(np.mean(embedding_cosine)),
            "minimum_cosine_similarity": float(np.min(embedding_cosine)),
        },
        "projected_representation": {
            "mae": float(np.mean(np.abs(fp32_representation - int8_representation))),
            "mean_cosine_similarity": float(np.mean(representation_cosine)),
            "minimum_cosine_similarity": float(np.min(representation_cosine)),
        },
        "head_output_parity": head_results,
    }
    np.savez_compressed(
        output_dir / "asn_mixed_precision_predictions.npz",
        path=np.asarray(paths),
        label=labels_array,
        fp32_embedding=fp32_embedding,
        int8_embedding=int8_embedding,
        fp32_representation=fp32_representation,
        int8_representation=int8_representation,
    )
    (output_dir / "asn_mixed_precision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    weight1, bias1, weight2, bias2 = projection_parameters
    header = """// Generated by evaluate_asn_mixed_precision_shared_model.py.
#pragma once

namespace runtime_shared_model {
inline constexpr int kAsnEmbeddingDim = 48;
inline constexpr int kProjectionHiddenDim = 96;
inline constexpr int kRiskRepresentationDim = 64;
"""
    header += format_float_array("kAsnProjectionWeight1", weight1)
    header += format_float_array("kAsnProjectionBias1", bias1)
    header += format_float_array("kAsnProjectionWeight2", weight2)
    header += format_float_array("kAsnProjectionBias2", bias2)
    header += "}  // namespace runtime_shared_model\n"
    (output_dir / "asn_folded_projection.h").write_text(header, encoding="ascii")

    lines = [
        "# ASN Mixed-Precision Runtime-Sharing Optimisation",
        "",
        f"The private ASN CNN is INT8 and produces 48 values. The {summary['private_projection_parameters']:,}-parameter node-specific projection remains FP32 and occupies {summary['private_projection_fp32_bytes'] / 1024.0:.2f} KiB in Flash.",
        "",
        "| Head | FP32 F1 | Mixed F1 | Delta | Disagreement | Risk MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in head_results.items():
        lines.append(
            f"| {name} | {row['fp32_f1']:.4f} | {row['int8_encoder_fp32_projection_f1']:.4f} | "
            f"{row['f1_delta']:+.4f} | {row['prediction_disagreement_rate']:.4f} | {row['risk_score_mae']:.5f} |"
        )
    lines.extend(
        [
            "",
            "This ablation isolates whether the ASN loss came from quantising its private projection. The mixed-precision route is selected for hardware only if it materially reduces downstream decision drift while remaining inside the memory budget.",
        ]
    )
    (output_dir / "ASN_MIXED_PRECISION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
