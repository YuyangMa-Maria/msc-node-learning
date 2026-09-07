"""Measure VSN cross-dataset transfer to the held-out SDNET2018 domain."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_vsn_binary import build_model, build_transforms  # noqa: E402
from train_vsn_student_baseline import VsnStudentDwCnn  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_kind: str
    checkpoint: Path
    image_size: int
    source_temperature: float
    evidence_tier: str
    note: str


class SdnetDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: Callable) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[index]
        path = PROJECT_ROOT / row["path"]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), index


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "label", "surface", "parent_group", "component_id", "split"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise RuntimeError(f"Index is missing required fields: {sorted(missing)}")
    return rows


def load_model(spec: ModelSpec, device: torch.device) -> nn.Module:
    checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint does not contain a model state dict: {spec.checkpoint}")
    if spec.model_kind == "mobilenet_v3_small":
        model = build_model("mobilenet_v3_small", pretrained=False)
    elif spec.model_kind == "student_dwcnn":
        args = checkpoint.get("args", {})
        model = VsnStudentDwCnn(
            width=float(args.get("width", 1.0)),
            dropout=float(args.get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unsupported model kind: {spec.model_kind}")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def predict(
    model: nn.Module,
    rows: list[dict[str, str]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    _, eval_transform = build_transforms(image_size)
    loader = DataLoader(
        SdnetDataset(rows, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    logits = np.empty(len(rows), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    with torch.inference_mode():
        for images, batch_labels, indices in loader:
            images = images.to(device, non_blocking=True)
            batch_logits = model(images).flatten()
            logits[indices.numpy()] = batch_logits.detach().cpu().numpy()
            labels[indices.numpy()] = batch_labels.numpy()
    return logits, labels


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    ece = 0.0
    for bin_index in range(bins):
        if bin_index == bins - 1:
            mask = (probabilities >= edges[bin_index]) & (probabilities <= edges[bin_index + 1])
        else:
            mask = (probabilities >= edges[bin_index]) & (probabilities < edges[bin_index + 1])
        count = int(mask.sum())
        if count:
            ece += count / total * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return ece


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.int64)
    positives = labels == 1
    negatives = labels == 0
    tp = int(((predictions == 1) & positives).sum())
    tn = int(((predictions == 0) & negatives).sum())
    fp = int(((predictions == 1) & negatives).sum())
    fn = int(((predictions == 0) & positives).sum())
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    try:
        roc_auc = float(roc_auc_score(labels, probabilities))
        average_precision = float(average_precision_score(labels, probabilities))
    except ValueError:
        roc_auc = float("nan")
        average_precision = float("nan")
    return {
        "samples": len(labels),
        "positive_prevalence": float(labels.mean()),
        "threshold": threshold,
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "precision_positive": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall_positive": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "specificity": tn / max(tn + fp, 1),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "nll": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        "ece_15": expected_calibration_error(labels, probabilities, bins=15),
        "mean_probability_positive": float(probabilities[positives].mean()),
        "mean_probability_negative": float(probabilities[negatives].mean()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def slice_metrics(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    masks: dict[str, np.ndarray] = {
        "all": np.ones(len(rows), dtype=bool),
        "frozen_test": np.array([row["split"] == "test" for row in rows]),
    }
    for surface in sorted({row["surface"] for row in rows}):
        masks[f"surface_all:{surface}"] = np.array([row["surface"] == surface for row in rows])
        masks[f"surface_test:{surface}"] = np.array(
            [row["surface"] == surface and row["split"] == "test" for row in rows]
        )
    return {
        name: binary_metrics(labels[mask], probabilities[mask])
        for name, mask in masks.items()
        if int(mask.sum()) > 0
    }


def group_bootstrap_ci(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    test_indices = np.array([index for index, row in enumerate(rows) if row["split"] == "test"])
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index in test_indices:
        row = rows[int(index)]
        group_id = row.get("component_id") or row.get("parent_group")
        if not group_id:
            raise RuntimeError("Bootstrap rows require component_id or parent_group")
        group_to_indices[group_id].append(int(index))
    groups = sorted(group_to_indices)
    rng = random.Random(seed)
    metric_names = ("balanced_accuracy", "macro_f1", "f1_positive", "roc_auc", "average_precision")
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(repetitions):
        selected_groups = [groups[rng.randrange(len(groups))] for _ in groups]
        sampled_indices = np.concatenate(
            [np.asarray(group_to_indices[group], dtype=np.int64) for group in selected_groups]
        )
        metrics = binary_metrics(labels[sampled_indices], probabilities[sampled_indices])
        for name in metric_names:
            value = float(metrics[name])
            if math.isfinite(value):
                samples[name].append(value)
    return {
        name: {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for name, values in samples.items()
        if values
    }


def threshold_sensitivity(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, float | int]]:
    test_mask = np.array([row["split"] == "test" for row in rows])
    return [
        binary_metrics(labels[test_mask], probabilities[test_mask], float(threshold))
        for threshold in np.arange(0.1, 1.0, 0.1)
    ]


def write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    logits: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
) -> None:
    fields = [
        "path",
        "label",
        "label_name",
        "surface",
        "parent_group",
        "component_id",
        "split",
        "logit",
        "probability_raw",
        "probability_source_calibrated",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    **{field: row[field] for field in fields[:7]},
                    "logit": f"{float(logits[index]):.9g}",
                    "probability_raw": f"{float(raw_probabilities[index]):.9g}",
                    "probability_source_calibrated": f"{float(calibrated_probabilities[index]):.9g}",
                }
            )


def write_summary_csv(path: Path, results: list[dict[str, object]]) -> None:
    fields = [
        "model_id",
        "probability_variant",
        "slice",
        "samples",
        "positive_prevalence",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "f1_positive",
        "precision_positive",
        "recall_positive",
        "specificity",
        "mcc",
        "roc_auc",
        "average_precision",
        "brier",
        "nll",
        "ece_15",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for variant in ("raw", "source_calibrated"):
                for slice_name, metrics in result["metrics"][variant].items():
                    writer.writerow(
                        {
                            "model_id": result["model_id"],
                            "probability_variant": variant,
                            "slice": slice_name,
                            **{field: metrics[field] for field in fields[3:]},
                        }
                    )


def write_report(path: Path, results: list[dict[str, object]]) -> None:
    lines = [
        "# VSN SDNET2018 Zero-Shot Evaluation",
        "",
        "## Protocol",
        "",
        "- Neither model was trained, calibrated, threshold-tuned, or selected using SDNET2018.",
        "- The decision threshold is fixed at 0.5.",
        "- The primary result is the frozen grouped test split; the complete dataset is an additional descriptive result.",
        "- Confidence calibration uses temperatures fitted on each model's source validation data only.",
        "- Confidence intervals use component-level bootstrap resampling of the frozen test split.",
        "",
        "## Frozen Grouped Test Results",
        "",
        "| Model | Variant | Balanced acc. | Macro-F1 | Positive F1 | AUROC | AUPRC | ECE | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for variant in ("raw", "source_calibrated"):
            metrics = result["metrics"][variant]["frozen_test"]
            lines.append(
                f"| {result['model_id']} | {variant} | "
                f"{metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
                f"{metrics['f1_positive']:.4f} | {metrics['roc_auc']:.4f} | "
                f"{metrics['average_precision']:.4f} | {metrics['ece_15']:.4f} | "
                f"{metrics['brier']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Surface Breakdown",
            "",
            "| Model | Surface | Balanced acc. | Macro-F1 | Positive F1 | AUROC |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        metrics_by_slice = result["metrics"]["source_calibrated"]
        for slice_name in sorted(name for name in metrics_by_slice if name.startswith("surface_test:")):
            surface = slice_name.split(":", maxsplit=1)[1]
            metrics = metrics_by_slice[slice_name]
            lines.append(
                f"| {result['model_id']} | {surface} | "
                f"{metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
                f"{metrics['f1_positive']:.4f} | {metrics['roc_auc']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Accuracy alone is not a primary metric because SDNET2018 is class-imbalanced.",
            "- Source-domain temperature scaling is not target-domain calibration.",
            "- A low zero-shot score indicates domain shift; it does not invalidate the source-domain model.",
            "- The legacy mixed-source teacher used an earlier source split with known leakage risk. Its SDNET2018 "
            "evaluation is independent, but its source-domain score must not be used as a publication-grade comparison.",
            "- SDNET2018 labels visual cracks, not structural safety or collapse probability.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate current VSN models on unseen SDNET2018.")
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "vsn_sdnet2018" / "sdnet2018_index_grouped.csv",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "vsn_binary_mixed_full"
        / "mobilenet_v3_small_pretrained"
        / "best.pt",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "conference_grouped_vsn_student_nw0"
        / "vsn_student_dwcnn_s128_w1_scratch"
        / "best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vsn_sdnet2018_zero_shot",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    specs = [
        ModelSpec(
            model_id="legacy_mixed_teacher_mobilenetv3_small_fp32",
            model_kind="mobilenet_v3_small",
            checkpoint=args.teacher_checkpoint,
            image_size=224,
            source_temperature=0.9976220726966858,
            evidence_tier="legacy_teacher_independent_target_test",
            note="Mixed-source teacher; source split predates grouped leakage-resistant protocol.",
        ),
        ModelSpec(
            model_id="grouped_student_dwcnn_s128_fp32",
            model_kind="student_dwcnn",
            checkpoint=args.student_checkpoint,
            image_size=128,
            source_temperature=0.9334065318107605,
            evidence_tier="conference_grouped_student",
            note="15,017-parameter from-scratch student trained on the grouped source split.",
        ),
    ]
    rows = read_rows(args.index)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"SDNET2018 images: {len(rows)}")

    all_results: list[dict[str, object]] = []
    for spec in specs:
        print(f"Evaluating {spec.model_id}")
        model = load_model(spec, device)
        logits, labels = predict(
            model,
            rows,
            spec.image_size,
            args.batch_size,
            args.num_workers,
            device,
        )
        raw_probabilities = sigmoid(logits)
        calibrated_probabilities = sigmoid(logits / spec.source_temperature)
        result = {
            "model_id": spec.model_id,
            "model_kind": spec.model_kind,
            "checkpoint": str(spec.checkpoint),
            "image_size": spec.image_size,
            "source_temperature": spec.source_temperature,
            "evidence_tier": spec.evidence_tier,
            "note": spec.note,
            "protocol": {
                "decision_threshold": 0.5,
                "target_domain_used_for_calibration": False,
                "target_domain_used_for_threshold_selection": False,
                "bootstrap_unit": "duplicate-safe original-image component",
                "bootstrap_repetitions": args.bootstrap_repetitions,
                "seed": args.seed,
            },
            "metrics": {
                "raw": slice_metrics(rows, labels, raw_probabilities),
                "source_calibrated": slice_metrics(rows, labels, calibrated_probabilities),
            },
            "frozen_test_group_bootstrap_95ci": {
                "raw": group_bootstrap_ci(
                    rows,
                    labels,
                    raw_probabilities,
                    args.bootstrap_repetitions,
                    args.seed,
                ),
                "source_calibrated": group_bootstrap_ci(
                    rows,
                    labels,
                    calibrated_probabilities,
                    args.bootstrap_repetitions,
                    args.seed,
                ),
            },
            "frozen_test_threshold_sensitivity": {
                "raw": threshold_sensitivity(rows, labels, raw_probabilities),
                "source_calibrated": threshold_sensitivity(
                    rows,
                    labels,
                    calibrated_probabilities,
                ),
            },
        }
        all_results.append(result)
        write_predictions(
            output_dir / f"{spec.model_id}_predictions.csv",
            rows,
            logits,
            raw_probabilities,
            calibrated_probabilities,
        )
        (output_dir / f"{spec.model_id}_result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summary_csv(output_dir / "summary_metrics.csv", all_results)
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(all_results, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_report(output_dir / "REPORT.md", all_results)
    print(f"Wrote zero-shot results to {output_dir}")


if __name__ == "__main__":
    main()
