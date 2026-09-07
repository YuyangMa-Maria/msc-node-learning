"""Quantise private 64-D representation models for ESP32-S3 deployment.

Quantisation is evaluated at both representation and final shared-head output
levels. This catches a private encoder whose tensor error looks small but still
changes the downstream risk decision.
"""

from __future__ import annotations

import argparse
import csv
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
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def run_head(representations: np.ndarray, state: dict[str, torch.Tensor]) -> np.ndarray:
    weight1 = state["0.weight"].detach().cpu().numpy()
    bias1 = state["0.bias"].detach().cpu().numpy()
    weight2 = state["3.weight"].detach().cpu().numpy()
    bias2 = state["3.bias"].detach().cpu().numpy()
    hidden = np.maximum(representations @ weight1.T + bias1, 0.0)
    return (hidden @ weight2.T + bias2).reshape(-1).astype(np.float32)


def select_balanced(samples: list[Any], total: int, seed: int) -> list[Any]:
    rng = np.random.default_rng(seed)
    selected: list[Any] = []
    per_class = total // 2
    for label in (0, 1):
        candidates = [sample for sample in samples if int(sample.label) == label]
        indices = rng.choice(len(candidates), size=min(per_class, len(candidates)), replace=False)
        selected.extend(candidates[int(index)] for index in indices)
    selected.sort(key=lambda sample: (int(sample.label), str(sample.path)))
    return selected


def representation_metrics(fp32: np.ndarray, int8: np.ndarray) -> dict[str, float]:
    """Measure element-wise parity between desktop and quantised representations."""
    difference = np.abs(fp32 - int8)
    denominator = np.linalg.norm(fp32, axis=1) * np.linalg.norm(int8, axis=1)
    cosine = np.sum(fp32 * int8, axis=1) / np.maximum(denominator, 1.0e-12)
    return {
        "mae": float(np.mean(difference)),
        "max_abs_error": float(np.max(difference)),
        "mean_cosine_similarity": float(np.mean(cosine)),
        "minimum_cosine_similarity": float(np.min(cosine)),
    }


def compare_head_outputs(
    labels: np.ndarray,
    fp32_representation: np.ndarray,
    int8_representation: np.ndarray,
    head_state: dict[str, torch.Tensor],
    threshold: float,
) -> dict[str, float]:
    """Propagate both representations through one fixed shared risk head."""
    fp32_logits = run_head(fp32_representation, head_state)
    int8_logits = run_head(int8_representation, head_state)
    fp32_probability = sigmoid(fp32_logits)
    int8_probability = sigmoid(int8_logits)
    fp32_prediction = fp32_probability >= threshold
    int8_prediction = int8_probability >= threshold
    return {
        "threshold": threshold,
        "fp32_accuracy": float(accuracy_score(labels, fp32_prediction)),
        "int8_accuracy": float(accuracy_score(labels, int8_prediction)),
        "fp32_f1": float(f1_score(labels, fp32_prediction, zero_division=0)),
        "int8_f1": float(f1_score(labels, int8_prediction, zero_division=0)),
        "f1_delta_int8_minus_fp32": float(
            f1_score(labels, int8_prediction, zero_division=0)
            - f1_score(labels, fp32_prediction, zero_division=0)
        ),
        "prediction_disagreement_rate": float(np.mean(fp32_prediction != int8_prediction)),
        "logit_mae": float(np.mean(np.abs(fp32_logits - int8_logits))),
        "risk_score_mae": float(np.mean(np.abs(fp32_probability - int8_probability))),
    }


def load_head_states(project_root: Path) -> dict[str, dict[str, torch.Tensor]]:
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
    base = {
        key.removeprefix("shared_head."): value.detach().cpu().float()
        for key, value in formal["model_state"].items()
        if key.startswith("shared_head.")
    }
    updated = {
        key: value.detach().cpu().float()
        for key, value in federated["final_shared_head_states"]["continual_pretrained_sample_weighted"].items()
    }
    return {"base_shared_head": base, "federated_updated_head": updated}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--evaluation-samples", type=int, default=500)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--bits", type=int, choices=(8, 16), default=8)
    parser.add_argument("--modalities", nargs="+", choices=("vsn", "asn"), default=["vsn", "asn"])
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    prepared_dir = args.prepared_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root / "scripts"))

    from train_asn_audio import read_samples as read_asn_samples
    from train_asn_student import AsnStudentDataset, AsnStudentModel
    from train_vsn_binary import build_transforms, read_samples as read_vsn_samples

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    prep_manifest = json.loads((prepared_dir / "runtime_shared_model_manifest.json").read_text(encoding="utf-8"))
    threshold_lookup = {
        (row["head"], row["modality"]): float(row["threshold"])
        for row in prep_manifest["metrics"]
    }
    head_states = load_head_states(project_root)

    vsn_index = project_root / "experiments/conference_grouped/vsn_binary_index_grouped.csv"
    _, vsn_transform = build_transforms(128)
    vsn_train = read_vsn_samples(vsn_index, "train", None)
    vsn_test = read_vsn_samples(vsn_index, "test", None)
    vsn_calibration_samples = select_balanced(vsn_train, args.calibration_samples, 2026)
    vsn_evaluation_samples = select_balanced(vsn_test, args.evaluation_samples, 2027)
    vsn_calibration = [
        vsn_transform(Image.open(sample.path).convert("RGB")).unsqueeze(0)
        for sample in vsn_calibration_samples
    ]
    vsn_evaluation = [
        (
            vsn_transform(Image.open(sample.path).convert("RGB")).unsqueeze(0),
            int(sample.label),
            str(sample.path),
        )
        for sample in vsn_evaluation_samples
    ]

    asn_index = project_root / "experiments/conference_grouped/asn_audio_index_grouped.csv"
    asn_checkpoint = torch.load(
        project_root / "outputs/conference_grouped_asn_student/asn_student_w0p5_d5p0s_scratch/best.pt",
        map_location="cpu",
        weights_only=False,
    )
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
    asn_student.to(device).eval()
    asn_train = read_asn_samples(asn_index, "train", None)
    asn_test = read_asn_samples(asn_index, "test", None)
    asn_calibration_samples = select_balanced(asn_train, args.calibration_samples, 2028)
    asn_evaluation_samples = select_balanced(asn_test, args.evaluation_samples, 2029)

    def audio_features(samples: list[Any], seed: int) -> list[tuple[torch.Tensor, int, str]]:
        dataset = AsnStudentDataset(
            samples,
            sample_rate=int(asn_args.get("sample_rate", 16000)),
            student_duration=float(asn_args.get("duration", 5.0)),
            teacher_duration=float(asn_args.get("teacher_duration", 10.0)),
            train=False,
            seed=seed,
        )
        rows: list[tuple[torch.Tensor, int, str]] = []
        with torch.no_grad():
            for index, (waveform, _teacher, label) in enumerate(dataset):
                feature = asn_student.frontend(waveform.unsqueeze(0).to(device)).detach().cpu()
                rows.append((feature, int(label.item()), str(samples[index].path)))
        return rows

    asn_calibration_rows = audio_features(asn_calibration_samples, 2030)
    asn_evaluation = audio_features(asn_evaluation_samples, 2031)
    asn_calibration = [row[0] for row in asn_calibration_rows]

    inputs = {
        "vsn": {
            "onnx": prepared_dir / "onnx/vsn_private_encoder_projection64_fp32.onnx",
            "shape": [1, 3, 128, 128],
            "calibration": vsn_calibration,
            "evaluation": vsn_evaluation,
        },
        "asn": {
            "onnx": prepared_dir / "onnx/asn_private_encoder_projection64_fp32.onnx",
            "shape": list(asn_calibration[0].shape),
            "calibration": asn_calibration,
            "evaluation": asn_evaluation,
        },
    }

    all_results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for modality, config in inputs.items():
        if modality not in args.modalities:
            continue
        modality_dir = output_dir / modality
        modality_dir.mkdir(parents=True, exist_ok=True)
        source_onnx = config["onnx"]
        espdl_path = modality_dir / f"{modality}_private_encoder_projection64_esp32s3_int{args.bits}.espdl"
        calibration = config["calibration"]
        graph = espdl_quantize_onnx(
            onnx_import_file=str(source_onnx),
            espdl_export_file=str(espdl_path),
            calib_dataloader=calibration,
            calib_steps=len(calibration),
            input_shape=config["shape"],
            inputs=[calibration[0].to(device)],
            target="esp32s3",
            num_of_bits=args.bits,
            device=str(device),
            error_report=True,
            export_config=True,
            export_test_values=True,
            verbose=1,
            metadata_props={
                "project": "post-disaster-node-learning",
                "stage": "runtime-shared-head",
                "node": modality.upper(),
                "output": "64-D risk representation",
            },
        )
        executor = TorchExecutor(graph=graph, device=str(device))
        session = ort.InferenceSession(str(source_onnx), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        fp32_outputs: list[np.ndarray] = []
        int8_outputs: list[np.ndarray] = []
        labels: list[int] = []
        paths: list[str] = []
        for index, (tensor, label, path) in enumerate(config["evaluation"], start=1):
            array = tensor.numpy().astype(np.float32, copy=False)
            fp32 = session.run([output_name], {input_name: array})[0].reshape(-1)
            int8 = executor.forward(inputs=[tensor.to(device)])[0].detach().cpu().numpy().reshape(-1)
            fp32_outputs.append(fp32.astype(np.float32))
            int8_outputs.append(int8.astype(np.float32))
            labels.append(label)
            paths.append(path)
            if index % 100 == 0 or index == len(config["evaluation"]):
                print(f"[{modality.upper()}] evaluated {index}/{len(config['evaluation'])}")

        fp32_array = np.stack(fp32_outputs)
        int8_array = np.stack(int8_outputs)
        label_array = np.asarray(labels, dtype=np.int64)
        rep_metrics = representation_metrics(fp32_array, int8_array)
        head_results: dict[str, Any] = {}
        for head_name, head_state in head_states.items():
            row = compare_head_outputs(
                label_array,
                fp32_array,
                int8_array,
                head_state,
                threshold_lookup[(head_name, modality)],
            )
            head_results[head_name] = row
            comparison_rows.append({"modality": modality, "head": head_name, **row})

        predictions_path = modality_dir / "representation_predictions.npz"
        np.savez_compressed(
            predictions_path,
            path=np.asarray(paths),
            label=label_array,
            fp32_representation=fp32_array,
            int8_representation=int8_array,
        )
        all_results[modality] = {
            "calibration_samples": len(calibration),
            "evaluation_samples": len(label_array),
            "source_onnx": str(source_onnx),
            "source_onnx_sha256": sha256(source_onnx),
            "espdl": str(espdl_path),
            "espdl_sha256": sha256(espdl_path),
            "espdl_bytes": espdl_path.stat().st_size,
            "representation_parity": rep_metrics,
            "head_output_parity": head_results,
            "predictions": str(predictions_path),
        }

    summary = {
        "stage": "INT8 split-representation parity before runtime shared-head activation",
        "target": "ESP32-S3",
        "quantisation": f"ESP-DL PTQ INT{args.bits}",
        "nodes": all_results,
        "claim_boundary": (
            "This establishes offline ESP-DL quantisation behaviour. Live on-device representation "
            "execution and atomic head activation are evaluated in the next hardware stage."
        ),
    }
    (output_dir / "runtime_shared_model_int8_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_dir / "head_output_parity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    lines = [
        "# Runtime Shared-Model INT8 Parity",
        "",
        "| Node | ESP-DL size (KiB) | Representation MAE | Mean cosine | Min cosine |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for modality, row in all_results.items():
        parity = row["representation_parity"]
        lines.append(
            f"| {modality.upper()} | {row['espdl_bytes'] / 1024.0:.2f} | {parity['mae']:.5f} | "
            f"{parity['mean_cosine_similarity']:.6f} | {parity['minimum_cosine_similarity']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Node | Head | FP32 F1 | INT8 F1 | Delta F1 | Disagreement | Risk MAE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {str(row['modality']).upper()} | {row['head']} | {row['fp32_f1']:.4f} | "
            f"{row['int8_f1']:.4f} | {row['f1_delta_int8_minus_fp32']:+.4f} | "
            f"{row['prediction_disagreement_rate']:.4f} | {row['risk_score_mae']:.5f} |"
        )
    lines.extend(
        [
            "",
            "The INT8 representation model is accepted for hardware hot-swap testing only if its downstream shared-head decisions remain close to the FP32 split reference. Hardware execution remains a separate gate.",
        ]
    )
    (output_dir / "RUNTIME_SHARED_MODEL_INT8_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
