"""Prepare split node models for runtime shared-head activation.

The formal parameter-sharing experiment standardises frozen node embeddings,
projects them into a common 64-D space, and applies a shared 64-32-1 risk head.
This script folds each fitted StandardScaler into its first projection layer,
exports the private encoder plus projection to ONNX, and verifies that splitting
the model does not change its predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn


MODALITIES = ("vsn", "asn")
FEDERATED_VARIANT = "continual_pretrained_sample_weighted"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    best_value = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(labels, probabilities >= threshold, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_value = float(threshold)
    return best_value


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    try:
        auroc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auroc = float("nan")
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": auroc,
    }


class VsnPrivateRepresentation(nn.Module):
    def __init__(self, features: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = projection

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embedding = self.pool(self.features(image)).flatten(1)
        return self.projection(embedding)


class AsnPrivateRepresentation(nn.Module):
    def __init__(self, encoder_net: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.encoder_net = encoder_net
        self.projection = projection

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder_net(log_mel).flatten(1)
        return self.projection(embedding)


class AsnPrivateEncoder(nn.Module):
    def __init__(self, encoder_net: nn.Module) -> None:
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(log_mel).flatten(1)


def make_folded_projection(
    state: dict[str, torch.Tensor], modality: str, scaler: StandardScaler
) -> nn.Sequential:
    first_weight = state[f"projections.{modality}.0.weight"].detach().cpu().float()
    first_bias = state[f"projections.{modality}.0.bias"].detach().cpu().float()
    second_weight = state[f"projections.{modality}.3.weight"].detach().cpu().float()
    second_bias = state[f"projections.{modality}.3.bias"].detach().cpu().float()

    scale = torch.from_numpy(scaler.scale_.astype(np.float32))
    mean = torch.from_numpy(scaler.mean_.astype(np.float32))
    folded_weight = first_weight / scale.unsqueeze(0)
    folded_bias = first_bias - folded_weight @ mean

    projection = nn.Sequential(
        nn.Linear(first_weight.shape[1], first_weight.shape[0]),
        nn.ReLU(),
        nn.Linear(second_weight.shape[1], second_weight.shape[0]),
        nn.ReLU(),
    )
    with torch.no_grad():
        projection[0].weight.copy_(folded_weight)
        projection[0].bias.copy_(folded_bias)
        projection[2].weight.copy_(second_weight)
        projection[2].bias.copy_(second_bias)
    projection.eval()
    return projection


def extract_head_state(checkpoint: dict[str, Any], prefix: str = "shared_head.") -> dict[str, torch.Tensor]:
    return {
        key.removeprefix(prefix): value.detach().cpu().float()
        for key, value in checkpoint["model_state"].items()
        if key.startswith(prefix)
    }


def run_head(representations: np.ndarray, state: dict[str, torch.Tensor]) -> np.ndarray:
    weight1 = state["0.weight"].numpy()
    bias1 = state["0.bias"].numpy()
    weight2 = state["3.weight"].numpy()
    bias2 = state["3.bias"].numpy()
    hidden = np.maximum(representations @ weight1.T + bias1, 0.0)
    return (hidden @ weight2.T + bias2).reshape(-1).astype(np.float32)


def project_numpy(
    embeddings: np.ndarray, state: dict[str, torch.Tensor], modality: str, scaler: StandardScaler
) -> np.ndarray:
    standardised = scaler.transform(embeddings).astype(np.float32)
    weight1 = state[f"projections.{modality}.0.weight"].detach().cpu().numpy()
    bias1 = state[f"projections.{modality}.0.bias"].detach().cpu().numpy()
    weight2 = state[f"projections.{modality}.3.weight"].detach().cpu().numpy()
    bias2 = state[f"projections.{modality}.3.bias"].detach().cpu().numpy()
    hidden = np.maximum(standardised @ weight1.T + bias1, 0.0)
    return np.maximum(hidden @ weight2.T + bias2, 0.0).astype(np.float32)


def onnx_export_and_check(
    model: nn.Module,
    example: torch.Tensor,
    path: Path,
    input_name: str,
    output_name: str,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            str(path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=None,
            dynamo=False,
        )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    graph = onnx.shape_inference.infer_shapes(graph)
    onnx.save(graph, str(path))

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    rng = np.random.default_rng(seed)
    errors: list[float] = []
    arrays = [example.detach().cpu().numpy().astype(np.float32)]
    arrays.extend(rng.standard_normal(tuple(example.shape)).astype(np.float32) for _ in range(7))
    for array in arrays:
        with torch.no_grad():
            expected = model(torch.from_numpy(array)).detach().cpu().numpy()
        observed = session.run([output_meta.name], {input_meta.name: array})[0]
        errors.append(float(np.max(np.abs(expected - observed))))
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "input_name": input_meta.name,
        "input_shape": list(input_meta.shape),
        "output_name": output_meta.name,
        "output_shape": list(output_meta.shape),
        "max_abs_error": max(errors),
        "mean_max_abs_error": float(np.mean(errors)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root / "scripts"))

    from train_asn_student import AsnStudentModel
    from train_vsn_student_baseline import VsnStudentDwCnn

    formal_path = project_root / "outputs/conference_grouped_parameter_sharing_models/shared_risk_head.pt"
    federated_path = project_root / "outputs/conference_grouped_shared_head_federated_update/final_shared_head_states.pt"
    cache_dir = project_root / "outputs/conference_grouped_parameter_sharing_models/embedding_cache"
    cache_paths = {
        "vsn": cache_dir / "vsn_cnn_embeddings_t3000_e1000_s42.npz",
        "asn": cache_dir / "asn_cnn_embeddings_t1600_e500_s42.npz",
    }
    student_paths = {
        "vsn": project_root / "outputs/conference_grouped_vsn_student_nw0/vsn_student_dwcnn_s128_w1_scratch/best.pt",
        "asn": project_root / "outputs/conference_grouped_asn_student/asn_student_w0p5_d5p0s_scratch/best.pt",
    }

    formal = torch.load(formal_path, map_location="cpu", weights_only=False)
    federated = torch.load(federated_path, map_location="cpu", weights_only=False)
    formal_state = formal["model_state"]
    base_head = extract_head_state(formal)
    updated_head = {
        key: value.detach().cpu().float()
        for key, value in federated["final_shared_head_states"][FEDERATED_VARIANT].items()
    }

    caches: dict[str, dict[str, np.ndarray]] = {}
    scalers: dict[str, StandardScaler] = {}
    folded_projections: dict[str, nn.Sequential] = {}
    split_equivalence: dict[str, Any] = {}
    for modality in MODALITIES:
        loaded = np.load(cache_paths[modality], allow_pickle=True)
        caches[modality] = {
            key: loaded[key].astype(np.float32 if key.endswith("_x") else np.int64)
            for key in ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y")
        }
        scaler = StandardScaler().fit(caches[modality]["train_x"])
        scalers[modality] = scaler
        folded = make_folded_projection(formal_state, modality, scaler)
        folded_projections[modality] = folded

        raw = caches[modality]["test_x"]
        expected = project_numpy(raw, formal_state, modality, scaler)
        with torch.no_grad():
            observed = folded(torch.from_numpy(raw)).numpy()
        base_expected = run_head(expected, base_head)
        base_observed = run_head(observed, base_head)
        split_equivalence[modality] = {
            "samples": int(len(raw)),
            "representation_max_abs_error": float(np.max(np.abs(expected - observed))),
            "base_logit_max_abs_error": float(np.max(np.abs(base_expected - base_observed))),
        }

    vsn_checkpoint = torch.load(student_paths["vsn"], map_location="cpu", weights_only=False)
    vsn_args = vsn_checkpoint.get("args", {})
    if not isinstance(vsn_args, dict):
        vsn_args = vars(vsn_args)
    vsn_student = VsnStudentDwCnn(
        width=float(vsn_args.get("width", 1.0)), dropout=float(vsn_args.get("dropout", 0.1))
    )
    vsn_student.load_state_dict(vsn_checkpoint["model"])
    vsn_model = VsnPrivateRepresentation(vsn_student.features, folded_projections["vsn"]).eval()

    asn_checkpoint = torch.load(student_paths["asn"], map_location="cpu", weights_only=False)
    asn_args = asn_checkpoint.get("args", {})
    if not isinstance(asn_args, dict):
        asn_args = vars(asn_args)
    asn_student = AsnStudentModel(
        sample_rate=int(asn_args.get("sample_rate", 16000)),
        n_fft=int(asn_args.get("n_fft", 1024)),
        hop_length=int(asn_args.get("hop_length", 320)),
        n_mels=int(asn_args.get("n_mels", 64)),
        width=float(asn_args.get("width", 0.5)),
        dropout=float(asn_args.get("dropout", 0.15)),
    )
    asn_student.load_state_dict(asn_checkpoint["model"])
    asn_student.eval()
    with torch.no_grad():
        asn_example = asn_student.frontend(torch.zeros(1, int(16000 * 5.0), dtype=torch.float32))
    asn_model = AsnPrivateRepresentation(asn_student.encoder.net, folded_projections["asn"]).eval()
    asn_encoder_model = AsnPrivateEncoder(asn_student.encoder.net).eval()

    export_dir = output_dir / "onnx"
    onnx_results = {
        "vsn": onnx_export_and_check(
            vsn_model,
            torch.zeros(1, 3, 128, 128),
            export_dir / "vsn_private_encoder_projection64_fp32.onnx",
            "image",
            "risk_representation",
            2026,
        ),
        "asn": onnx_export_and_check(
            asn_model,
            asn_example,
            export_dir / "asn_private_encoder_projection64_fp32.onnx",
            "log_mel",
            "risk_representation",
            2027,
        ),
        "asn_encoder_only": onnx_export_and_check(
            asn_encoder_model,
            asn_example,
            export_dir / "asn_private_encoder48_fp32.onnx",
            "log_mel",
            "private_embedding",
            2028,
        ),
    }
    np.save(export_dir / "asn_log_mel_zero_example.npy", asn_example.numpy().astype(np.float32))

    metric_rows: list[dict[str, Any]] = []
    head_states = {"base_shared_head": base_head, "federated_updated_head": updated_head}
    for head_name, head_state in head_states.items():
        for modality in MODALITIES:
            cache = caches[modality]
            val_rep = project_numpy(cache["val_x"], formal_state, modality, scalers[modality])
            test_rep = project_numpy(cache["test_x"], formal_state, modality, scalers[modality])
            val_prob = sigmoid(run_head(val_rep, head_state))
            test_prob = sigmoid(run_head(test_rep, head_state))
            threshold = best_threshold(val_prob, cache["val_y"])
            metric_rows.append(
                {
                    "head": head_name,
                    "modality": modality,
                    **metrics(cache["test_y"], test_prob, threshold),
                }
            )

    deltas: dict[str, Any] = {}
    for modality in MODALITIES:
        base = next(row for row in metric_rows if row["head"] == "base_shared_head" and row["modality"] == modality)
        updated = next(row for row in metric_rows if row["head"] == "federated_updated_head" and row["modality"] == modality)
        deltas[modality] = {
            "f1_delta_updated_minus_base": float(updated["f1"] - base["f1"]),
            "accuracy_delta_updated_minus_base": float(updated["accuracy"] - base["accuracy"]),
            "auroc_delta_updated_minus_base": float(updated["auroc"] - base["auroc"]),
        }

    manifest = {
        "stage": "runtime shared-head split-model preparation",
        "formal_checkpoint": str(formal_path),
        "federated_checkpoint": str(federated_path),
        "federated_variant": FEDERATED_VARIANT,
        "architecture": {
            "private_path": "modality encoder -> folded node-specific projection -> 64-D representation",
            "replaceable_path": "Linear(64,32) -> ReLU -> Linear(32,1)",
            "shared_head_parameters": 2113,
        },
        "split_equivalence": split_equivalence,
        "onnx": onnx_results,
        "metrics": metric_rows,
        "head_update_deltas": deltas,
        "claim_boundary": (
            "This stage verifies software splitting and ONNX export. It does not yet establish "
            "INT8 representation parity or live hardware hot activation."
        ),
    }
    (output_dir / "runtime_shared_model_manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "shared_head_update_metrics.csv", metric_rows)

    lines = [
        "# Runtime Shared-Head Split-Model Preparation",
        "",
        "## Purpose",
        "",
        "The formal grouped-split parameter-sharing model was decomposed into a private node encoder/projection and a replaceable 2,113-parameter shared risk head. Training-time standardisation was algebraically folded into each node projection, avoiding an additional runtime preprocessing stage.",
        "",
        "## Equivalence gates",
        "",
        "| Node | Held-out samples | 64-D max abs error | Base-logit max abs error | ONNX max abs error |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for modality in MODALITIES:
        split = split_equivalence[modality]
        onnx_row = onnx_results[modality]
        lines.append(
            f"| {modality.upper()} | {split['samples']} | {split['representation_max_abs_error']:.3e} | "
            f"{split['base_logit_max_abs_error']:.3e} | {onnx_row['max_abs_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Shared-head update effect",
            "",
            "Thresholds were selected independently on each validation split, then frozen for the held-out test split.",
            "",
            "| Head | Node | Accuracy | Precision | Recall | F1 | AUROC | Threshold |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metric_rows:
        lines.append(
            f"| {row['head']} | {str(row['modality']).upper()} | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['auroc']:.4f} | {row['threshold']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Passing these checks establishes that the formal sharing architecture can be split at the 64-D interface without changing its FP32 behaviour. The next gate is INT8 representation parity, followed by atomic activation and rollback on both ESP32-S3 boards.",
            "",
            "## Claim boundary",
            "",
            "This report does not yet claim that the received parameters are active in live inference. That claim requires the subsequent quantised hardware hot-swap experiment.",
        ]
    )
    (output_dir / "RUNTIME_SHARED_MODEL_PREPARATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(jsonable(manifest), indent=2))


if __name__ == "__main__":
    main()
