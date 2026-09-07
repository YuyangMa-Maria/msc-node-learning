"""Convert the grouped VSN student to the ESP-DL INT8 format.

Balanced validation images calibrate activation ranges. The generated model is
accepted only after ONNX/ESP-DL parity, classification and calibration metrics
have been recorded against the unchanged evaluation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.executor import TorchExecutor
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class CalibrationRecord:
    path: str
    label: int
    source: str


class CalibrationDataset(Dataset[torch.Tensor]):
    """Provide deterministic validation tensors to the ESP-PPQ calibrator."""

    def __init__(self, samples: list[Any], transform: Any) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.samples[index].path).convert("RGB")
        return self.transform(image)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid_array(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits.astype(np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == 0:
            selected = (probabilities >= edges[index]) & (
                probabilities <= edges[index + 1]
            )
        else:
            selected = (probabilities > edges[index]) & (
                probabilities <= edges[index + 1]
            )
        if not np.any(selected):
            continue
        result += float(np.mean(selected)) * abs(
            float(np.mean(probabilities[selected])) - float(np.mean(labels[selected]))
        )
    return result


def classification_metrics(
    labels: np.ndarray, logits: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    probabilities = sigmoid_array(logits)
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "ece_15_bins": expected_calibration_error(labels, probabilities, bins=15),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def select_balanced_calibration_samples(
    samples: list[Any], total: int, seed: int
) -> list[Any]:
    if total < 2 or total % 2:
        raise ValueError("--calibration-samples must be an even integer of at least 2")

    rng = np.random.default_rng(seed)
    selected: list[Any] = []
    per_class = total // 2
    for label in (0, 1):
        candidates = [sample for sample in samples if int(sample.label) == label]
        if len(candidates) < per_class:
            raise RuntimeError(
                f"Only {len(candidates)} train samples are available for label {label}"
            )
        indices = rng.choice(len(candidates), size=per_class, replace=False)
        selected.extend(candidates[int(index)] for index in indices)

    selected.sort(key=lambda sample: (int(sample.label), str(sample.path)))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize the grouped VSN student to ESP32-S3 INT8 ESP-DL format."
    )
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--evaluation-samples",
        type=int,
        default=0,
        help="Number of grouped test images to evaluate; 0 evaluates the full test split.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = project_root / "scripts"
    sys.path.insert(0, str(scripts_dir))

    from train_vsn_binary import build_transforms, read_samples

    source_onnx = (
        project_root
        / "outputs"
        / "hardware_deployment"
        / "vsn_esp32s3"
        / "export"
        / "vsn_student_dwcnn_s128_fp32.onnx"
    )
    test_vector_path = (
        project_root
        / "outputs"
        / "hardware_deployment"
        / "vsn_esp32s3"
        / "export"
        / "test_vector_nchw_fp32.npy"
    )
    index_path = (
        project_root
        / "experiments"
        / "conference_grouped"
        / "vsn_binary_index_grouped.csv"
    )
    output_dir = args.output_dir or (
        project_root
        / "outputs"
        / "hardware_deployment"
        / "vsn_esp32s3"
        / "espdl_ptq_int8"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    image_size = 128
    _, eval_transform = build_transforms(image_size)
    train_samples = read_samples(index_path, "train", None)
    test_samples = read_samples(index_path, "test", None)
    calibration_samples = select_balanced_calibration_samples(
        train_samples, args.calibration_samples, args.seed
    )

    if args.evaluation_samples > 0:
        test_samples = test_samples[: args.evaluation_samples]
    if not test_samples:
        raise RuntimeError("No grouped test samples were available")

    calibration_loader = DataLoader(
        CalibrationDataset(calibration_samples, eval_transform),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    quant_input_onnx = output_dir / "vsn_student_dwcnn_s128_espdl_input.onnx"
    shutil.copy2(source_onnx, quant_input_onnx)
    espdl_path = output_dir / "vsn_student_dwcnn_s128_esp32s3_int8.espdl"
    test_tensor = torch.from_numpy(np.load(test_vector_path)).to(device=device)

    quantized_graph = espdl_quantize_onnx(
        onnx_import_file=str(quant_input_onnx),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calibration_loader,
        calib_steps=len(calibration_samples),
        input_shape=[1, 3, image_size, image_size],
        inputs=[test_tensor],
        target="esp32s3",
        num_of_bits=8,
        device=device,
        error_report=True,
        export_config=True,
        export_test_values=True,
        verbose=1,
        metadata_props={
            "project": "post-disaster-node-learning",
            "node": "VSN",
            "quantization": "INT8 PTQ",
            "split": "conference_grouped",
        },
    )

    quantized_executor = TorchExecutor(graph=quantized_graph, device=device)
    fp32_session = ort.InferenceSession(
        str(source_onnx), providers=["CPUExecutionProvider"]
    )
    fp32_input_name = fp32_session.get_inputs()[0].name
    fp32_output_name = fp32_session.get_outputs()[0].name

    labels: list[int] = []
    paths: list[str] = []
    sources: list[str] = []
    fp32_logits: list[float] = []
    int8_logits: list[float] = []

    with torch.no_grad():
        for index, sample in enumerate(test_samples, start=1):
            image = Image.open(sample.path).convert("RGB")
            tensor = eval_transform(image).unsqueeze(0)
            array = tensor.numpy().astype(np.float32, copy=False)
            fp32_logit = float(
                fp32_session.run(
                    [fp32_output_name], {fp32_input_name: array}
                )[0].reshape(-1)[0]
            )
            quantized_output = quantized_executor.forward(
                inputs=[tensor.to(device=device)]
            )[0]
            int8_logit = float(quantized_output.detach().cpu().reshape(-1)[0])

            labels.append(int(sample.label))
            paths.append(str(sample.path))
            sources.append(str(sample.source))
            fp32_logits.append(fp32_logit)
            int8_logits.append(int8_logit)

            if index % 500 == 0 or index == len(test_samples):
                print(f"Evaluated {index}/{len(test_samples)} grouped test images")

    label_array = np.asarray(labels, dtype=np.int64)
    fp32_logit_array = np.asarray(fp32_logits, dtype=np.float32)
    int8_logit_array = np.asarray(int8_logits, dtype=np.float32)
    fp32_risk = sigmoid_array(fp32_logit_array)
    int8_risk = sigmoid_array(int8_logit_array)
    fp32_predictions = fp32_risk >= 0.5
    int8_predictions = int8_risk >= 0.5

    fp32_metrics = classification_metrics(label_array, fp32_logit_array)
    int8_metrics = classification_metrics(label_array, int8_logit_array)
    comparison = {
        "prediction_disagreement_rate": float(
            np.mean(fp32_predictions != int8_predictions)
        ),
        "logit_mae": float(np.mean(np.abs(fp32_logit_array - int8_logit_array))),
        "logit_max_abs_error": float(
            np.max(np.abs(fp32_logit_array - int8_logit_array))
        ),
        "risk_score_mae": float(np.mean(np.abs(fp32_risk - int8_risk))),
        "risk_score_max_abs_error": float(np.max(np.abs(fp32_risk - int8_risk))),
        "f1_delta_int8_minus_fp32": int8_metrics["f1"] - fp32_metrics["f1"],
        "ece_delta_int8_minus_fp32": (
            int8_metrics["ece_15_bins"] - fp32_metrics["ece_15_bins"]
        ),
    }

    predictions_path = output_dir / "grouped_test_predictions.npz"
    np.savez_compressed(
        predictions_path,
        path=np.asarray(paths),
        source=np.asarray(sources),
        label=label_array,
        fp32_logit=fp32_logit_array,
        int8_logit=int8_logit_array,
        fp32_risk_score=fp32_risk.astype(np.float32),
        int8_risk_score=int8_risk.astype(np.float32),
    )

    calibration_records = [
        CalibrationRecord(
            path=str(sample.path),
            label=int(sample.label),
            source=str(sample.source),
        )
        for sample in calibration_samples
    ]
    manifest = {
        "stage": "VSN ESP32-S3 INT8 PTQ",
        "source_onnx": str(source_onnx),
        "source_onnx_sha256": sha256(source_onnx),
        "espdl": {
            "path": str(espdl_path),
            "sha256": sha256(espdl_path),
            "size_bytes": espdl_path.stat().st_size,
            "target": "esp32s3",
            "bits": 8,
            "embedded_test_values": True,
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "device": device,
            "esp_ppq": "1.3.6",
        },
        "calibration": {
            "split": "train",
            "seed": args.seed,
            "samples": len(calibration_records),
            "label_0": sum(record.label == 0 for record in calibration_records),
            "label_1": sum(record.label == 1 for record in calibration_records),
            "records": [asdict(record) for record in calibration_records],
        },
        "evaluation": {
            "split": "test",
            "samples": len(test_samples),
            "threshold": 0.5,
            "fp32": fp32_metrics,
            "int8": int8_metrics,
            "comparison": comparison,
            "predictions": str(predictions_path),
        },
        "caution": [
            "ESP-PPQ results are simulated target-quantized inference until board-side parity passes.",
            "The calibration set contains only grouped training images.",
            "Live-camera preprocessing and end-to-end latency are not assessed by this script.",
        ],
    }
    manifest_path = output_dir / "quantization_report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = [
        "# VSN ESP32-S3 INT8 PTQ",
        "",
        "## Artifact",
        "",
        f"- ESP-DL model: `{espdl_path.name}`",
        f"- Model size: {espdl_path.stat().st_size / 1024:.2f} KiB",
        f"- Calibration: {len(calibration_records)} grouped-train images "
        f"({manifest['calibration']['label_0']} normal, "
        f"{manifest['calibration']['label_1']} crack)",
        f"- Evaluation: {len(test_samples)} grouped-test images",
        "",
        "## Grouped-Test Metrics",
        "",
        "| Model | Accuracy | F1 | AUROC | ECE | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| FP32 ONNX | {fp32_metrics['accuracy']:.6f} | "
            f"{fp32_metrics['f1']:.6f} | {fp32_metrics['auroc']:.6f} | "
            f"{fp32_metrics['ece_15_bins']:.6f} | {fp32_metrics['brier']:.6f} |"
        ),
        (
            f"| ESP32-S3 INT8 simulation | {int8_metrics['accuracy']:.6f} | "
            f"{int8_metrics['f1']:.6f} | {int8_metrics['auroc']:.6f} | "
            f"{int8_metrics['ece_15_bins']:.6f} | {int8_metrics['brier']:.6f} |"
        ),
        "",
        "## Quantization Difference",
        "",
        f"- Prediction disagreement: {comparison['prediction_disagreement_rate']:.6f}",
        f"- Risk-score MAE: {comparison['risk_score_mae']:.6f}",
        f"- Maximum risk-score error: {comparison['risk_score_max_abs_error']:.6f}",
        f"- INT8 minus FP32 F1: {comparison['f1_delta_int8_minus_fp32']:+.6f}",
        "",
        "## Next Gate",
        "",
        "Embed this `.espdl` artifact in the ESP-IDF project and verify its embedded "
        "test vector on the XIAO ESP32S3 before enabling live-camera inference.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps(manifest["evaluation"], indent=2))
    print(espdl_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
