"""Train CODEBRIM teacher models used in the VSN extension study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_vsn_sdnet_zero_shot import (  # noqa: E402
    binary_metrics,
    group_bootstrap_ci,
    sigmoid,
)
from train_vsn_binary import build_transforms  # noqa: E402


DEFECT_LABELS = (
    "Crack",
    "Spallation",
    "Efflorescence",
    "ExposedBars",
    "CorrosionStain",
)
RUN_KINDS = ("binary_only", "hierarchical_multitask")


class CodebrimDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: Callable) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(PROJECT_ROOT / row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        damage = torch.tensor(float(row["damage"]), dtype=torch.float32)
        defects = torch.tensor(
            [float(row[label]) for label in DEFECT_LABELS],
            dtype=torch.float32,
        )
        return tensor, damage, defects, index


class CodebrimTeacher(nn.Module):
    def __init__(self, multitask: bool, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        embedding_dim = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Identity()
        self.binary_head = nn.Linear(embedding_dim, 1)
        self.defect_head = nn.Linear(embedding_dim, len(DEFECT_LABELS)) if multitask else None

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedding = self.backbone(images)
        binary_logits = self.binary_head(embedding).flatten()
        defect_logits = self.defect_head(embedding) if self.defect_head is not None else None
        return binary_logits, defect_logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No CODEBRIM rows found in {path}")
    return rows


def make_loader(
    rows: list[dict[str, str]],
    transform: Callable,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        CodebrimDataset(rows, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def multilabel_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    per_label: dict[str, dict[str, float]] = {}
    average_precisions: list[float] = []
    roc_aucs: list[float] = []
    f1_values: list[float] = []
    for index, label_name in enumerate(DEFECT_LABELS):
        y_true = labels[:, index]
        y_prob = probabilities[:, index]
        y_pred = predictions[:, index]
        average_precision = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        average_precisions.append(average_precision)
        roc_aucs.append(roc_auc)
        f1_values.append(f1)
        per_label[label_name] = {
            "prevalence": float(y_true.mean()),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": f1,
            "average_precision": average_precision,
            "roc_auc": roc_auc,
        }
    return {
        "threshold": threshold,
        "samples": len(labels),
        "macro_f1": float(np.mean(f1_values)),
        "micro_f1": float(f1_score(labels.ravel(), predictions.ravel(), zero_division=0)),
        "mean_average_precision": float(np.mean(average_precisions)),
        "macro_roc_auc": float(np.mean(roc_aucs)),
        "exact_match_ratio": float(np.all(labels == predictions, axis=1).mean()),
        "per_label": per_label,
    }


def collect_outputs(
    model: CodebrimTeacher,
    loader: DataLoader,
    device: torch.device,
    binary_criterion: nn.Module,
    defect_criterion: nn.Module | None,
    defect_loss_weight: float,
) -> dict[str, object]:
    model.eval()
    binary_logits = np.empty(len(loader.dataset), dtype=np.float32)
    damage_labels = np.empty(len(loader.dataset), dtype=np.int64)
    defect_logits = np.empty((len(loader.dataset), len(DEFECT_LABELS)), dtype=np.float32)
    defect_labels = np.empty((len(loader.dataset), len(DEFECT_LABELS)), dtype=np.int64)
    total_loss = 0.0
    with torch.inference_mode():
        for images, damage, defects, indices in loader:
            images = images.to(device, non_blocking=True)
            damage = damage.to(device, non_blocking=True)
            defects = defects.to(device, non_blocking=True)
            batch_binary_logits, batch_defect_logits = model(images)
            loss = binary_criterion(batch_binary_logits, damage)
            if batch_defect_logits is not None and defect_criterion is not None:
                loss = loss + defect_loss_weight * defect_criterion(batch_defect_logits, defects)
            index_values = indices.numpy()
            binary_logits[index_values] = batch_binary_logits.detach().cpu().numpy()
            damage_labels[index_values] = damage.detach().cpu().numpy().astype(np.int64)
            defect_labels[index_values] = defects.detach().cpu().numpy().astype(np.int64)
            if batch_defect_logits is not None:
                defect_logits[index_values] = batch_defect_logits.detach().cpu().numpy()
            total_loss += float(loss.item()) * len(indices)
    return {
        "binary_logits": binary_logits,
        "damage_labels": damage_labels,
        "defect_logits": defect_logits if model.defect_head is not None else None,
        "defect_labels": defect_labels,
        "loss": total_loss / max(len(loader.dataset), 1),
    }


def train_epoch(
    model: CodebrimTeacher,
    loader: DataLoader,
    device: torch.device,
    binary_criterion: nn.Module,
    defect_criterion: nn.Module | None,
    defect_loss_weight: float,
    optimiser: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    total_binary_loss = 0.0
    total_defect_loss = 0.0
    total_loss = 0.0
    for images, damage, defects, _ in loader:
        images = images.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True)
        defects = defects.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        binary_logits, defect_logits = model(images)
        binary_loss = binary_criterion(binary_logits, damage)
        defect_loss = torch.zeros((), device=device)
        if defect_logits is not None and defect_criterion is not None:
            defect_loss = defect_criterion(defect_logits, defects)
        loss = binary_loss + defect_loss_weight * defect_loss
        loss.backward()
        optimiser.step()
        batch_size = images.size(0)
        total_binary_loss += float(binary_loss.item()) * batch_size
        total_defect_loss += float(defect_loss.item()) * batch_size
        total_loss += float(loss.item()) * batch_size
    sample_count = max(len(loader.dataset), 1)
    return {
        "total": total_loss / sample_count,
        "binary": total_binary_loss / sample_count,
        "defect": total_defect_loss / sample_count,
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimiser = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    criterion = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = criterion(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    binary_logits: np.ndarray,
    raw_binary_probabilities: np.ndarray,
    calibrated_binary_probabilities: np.ndarray,
    defect_probabilities: np.ndarray | None,
) -> None:
    fields = [
        "path",
        "filename",
        "split",
        "parent_group",
        "damage",
        *DEFECT_LABELS,
        "binary_logit",
        "binary_probability_raw",
        "binary_probability_val_calibrated",
    ]
    if defect_probabilities is not None:
        fields.extend(f"{label}_probability" for label in DEFECT_LABELS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = {
                **{field: row[field] for field in fields[: 5 + len(DEFECT_LABELS)]},
                "binary_logit": f"{float(binary_logits[index]):.9g}",
                "binary_probability_raw": f"{float(raw_binary_probabilities[index]):.9g}",
                "binary_probability_val_calibrated": (
                    f"{float(calibrated_binary_probabilities[index]):.9g}"
                ),
            }
            if defect_probabilities is not None:
                output.update(
                    {
                        f"{label}_probability": f"{float(defect_probabilities[index, label_index]):.9g}"
                        for label_index, label in enumerate(DEFECT_LABELS)
                    }
                )
            writer.writerow(output)


def train_run(args: argparse.Namespace, run_kind: str) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    rows = read_rows(args.index)
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    train_transform, eval_transform = build_transforms(args.image_size)
    loaders = {
        "train": make_loader(
            split_rows["train"],
            train_transform,
            args.batch_size,
            args.num_workers,
            device,
            True,
        ),
        "val": make_loader(
            split_rows["val"],
            eval_transform,
            args.batch_size,
            args.num_workers,
            device,
            False,
        ),
        "test": make_loader(
            split_rows["test"],
            eval_transform,
            args.batch_size,
            args.num_workers,
            device,
            False,
        ),
    }

    multitask = run_kind == "hierarchical_multitask"
    model = CodebrimTeacher(multitask=multitask, pretrained=True).to(device)
    train_damage = np.asarray([int(row["damage"]) for row in split_rows["train"]])
    binary_positive_weight = float((train_damage == 0).sum() / max((train_damage == 1).sum(), 1))
    train_defects = np.asarray(
        [[int(row[label]) for label in DEFECT_LABELS] for row in split_rows["train"]],
        dtype=np.int64,
    )
    defect_positive_weights = (
        (len(train_defects) - train_defects.sum(axis=0))
        / np.maximum(train_defects.sum(axis=0), 1)
    ).astype(np.float32)
    binary_train_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(binary_positive_weight, dtype=torch.float32, device=device)
    )
    binary_eval_criterion = nn.BCEWithLogitsLoss()
    defect_train_criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(defect_positive_weights, dtype=torch.float32, device=device)
        )
        if multitask
        else None
    )
    defect_eval_criterion = nn.BCEWithLogitsLoss() if multitask else None
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=max(args.epochs, 1),
    )

    run_id = f"codebrim_{run_kind}_seed{args.seed}"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    best_selection_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    start_time = time.perf_counter()

    print(f"Run={run_id} device={device} multitask={multitask}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(
            model,
            loaders["train"],
            device,
            binary_train_criterion,
            defect_train_criterion,
            args.defect_loss_weight,
            optimiser,
        )
        val_outputs = collect_outputs(
            model,
            loaders["val"],
            device,
            binary_eval_criterion,
            defect_eval_criterion,
            args.defect_loss_weight,
        )
        val_binary_probabilities = sigmoid(val_outputs["binary_logits"])
        val_binary_metrics = binary_metrics(
            val_outputs["damage_labels"],
            val_binary_probabilities,
        )
        val_multilabel_metrics = None
        if val_outputs["defect_logits"] is not None:
            val_multilabel_metrics = multilabel_metrics(
                val_outputs["defect_labels"],
                sigmoid(val_outputs["defect_logits"]),
            )
            selection_score = 0.5 * float(val_binary_metrics["macro_f1"]) + 0.5 * float(
                val_multilabel_metrics["mean_average_precision"]
            )
        else:
            selection_score = float(val_binary_metrics["macro_f1"])
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_losses,
                "val_loss": val_outputs["loss"],
                "val_binary": val_binary_metrics,
                "val_multilabel": val_multilabel_metrics,
                "selection_score": selection_score,
                "learning_rate": optimiser.param_groups[0]["lr"],
            }
        )
        label_summary = (
            f" val_mAP={val_multilabel_metrics['mean_average_precision']:.4f}"
            if val_multilabel_metrics is not None
            else ""
        )
        print(
            f"{run_id} epoch={epoch:02d} train_loss={train_losses['total']:.4f} "
            f"val_binary_macro_f1={val_binary_metrics['macro_f1']:.4f}"
            f"{label_summary} selection={selection_score:.4f}",
            flush=True,
        )
        if selection_score > best_selection_score + args.min_delta:
            best_selection_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "run_kind": run_kind,
                    "multitask": multitask,
                    "defect_labels": DEFECT_LABELS,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "args": vars(args),
                    "val_binary_metrics": val_binary_metrics,
                    "val_multilabel_metrics": val_multilabel_metrics,
                    "selection_score": selection_score,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_outputs = collect_outputs(
        model,
        loaders["val"],
        device,
        binary_eval_criterion,
        defect_eval_criterion,
        args.defect_loss_weight,
    )
    temperature = fit_temperature(
        val_outputs["binary_logits"],
        val_outputs["damage_labels"],
    )
    test_outputs = collect_outputs(
        model,
        loaders["test"],
        device,
        binary_eval_criterion,
        defect_eval_criterion,
        args.defect_loss_weight,
    )
    raw_binary_probabilities = sigmoid(test_outputs["binary_logits"])
    calibrated_binary_probabilities = sigmoid(test_outputs["binary_logits"] / temperature)
    defect_probabilities = (
        sigmoid(test_outputs["defect_logits"])
        if test_outputs["defect_logits"] is not None
        else None
    )
    binary_raw_metrics = binary_metrics(
        test_outputs["damage_labels"],
        raw_binary_probabilities,
    )
    binary_calibrated_metrics = binary_metrics(
        test_outputs["damage_labels"],
        calibrated_binary_probabilities,
    )
    test_multilabel_metrics = (
        multilabel_metrics(test_outputs["defect_labels"], defect_probabilities)
        if defect_probabilities is not None
        else None
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result: dict[str, object] = {
        "run_id": run_id,
        "run_kind": run_kind,
        "architecture": "MobileNetV3-Small shared encoder",
        "initialisation": "ImageNet",
        "task": {
            "binary": "any annotated defect",
            "defect_labels": list(DEFECT_LABELS) if multitask else [],
            "multitask": multitask,
            "defect_loss_weight": args.defect_loss_weight if multitask else 0.0,
        },
        "protocol": {
            "index": str(args.index),
            "official_parent_disjoint_splits": True,
            "test_used_during_training": False,
            "selection_metric": (
                "0.5 * validation binary macro-F1 + 0.5 * validation defect mAP"
                if multitask
                else "validation binary macro-F1"
            ),
            "decision_threshold": 0.5,
            "calibration_data": "official validation split",
            "threshold_tuning": None,
            "seed": args.seed,
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_selection_score": best_selection_score,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "binary_positive_weight": binary_positive_weight,
            "defect_positive_weights": {
                label: float(value)
                for label, value in zip(DEFECT_LABELS, defect_positive_weights)
            },
            "sample_counts": {split: len(values) for split, values in split_rows.items()},
            "elapsed_seconds": time.perf_counter() - start_time,
        },
        "model": {
            "parameters": parameter_count,
            "fp32_weight_mb_estimate": parameter_count * 4 / (1024 * 1024),
        },
        "calibration": {
            "binary_temperature": temperature,
            "note": "Only the binary evidence head is temperature-scaled.",
        },
        "test": {
            "loss": test_outputs["loss"],
            "binary_raw": binary_raw_metrics,
            "binary_val_calibrated": binary_calibrated_metrics,
            "binary_group_bootstrap_95ci": {
                "raw": group_bootstrap_ci(
                    split_rows["test"],
                    test_outputs["damage_labels"],
                    raw_binary_probabilities,
                    args.bootstrap_repetitions,
                    args.seed,
                ),
                "val_calibrated": group_bootstrap_ci(
                    split_rows["test"],
                    test_outputs["damage_labels"],
                    calibrated_binary_probabilities,
                    args.bootstrap_repetitions,
                    args.seed,
                ),
            },
            "multilabel_raw": test_multilabel_metrics,
        },
        "claims_boundary": (
            "The labels describe visible defect evidence. They are not ordinal severity grades "
            "or certified structural risk levels."
        ),
    }
    (output_dir / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history),
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_predictions(
        output_dir / "test_predictions.csv",
        split_rows["test"],
        test_outputs["binary_logits"],
        raw_binary_probabilities,
        calibrated_binary_probabilities,
        defect_probabilities,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def write_comparison(output_dir: Path, results: list[dict[str, object]]) -> None:
    lines = [
        "# CODEBRIM Binary-Only and Hierarchical Multi-Task Teachers",
        "",
        "## Protocol",
        "",
        "- Both Teachers use the same MobileNetV3-Small encoder, ImageNet initialisation, seed, and official splits.",
        "- The multi-task variant adds five visual defect heads to the shared encoder.",
        "- The binary evidence label is any annotated defect; it is not a structural risk level.",
        "- Test thresholds remain fixed at 0.5.",
        "",
        "## Binary Evidence Results",
        "",
        "| Model | Params | Balanced acc. | Macro-F1 | Positive F1 | Precision | Recall | AUROC | AUPRC | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result["test"]["binary_val_calibrated"]
        lines.append(
            f"| {result['run_kind']} | {result['model']['parameters']:,} | "
            f"{metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['f1_positive']:.4f} | {metrics['precision_positive']:.4f} | "
            f"{metrics['recall_positive']:.4f} | {metrics['roc_auc']:.4f} | "
            f"{metrics['average_precision']:.4f} | {metrics['ece_15']:.4f} |"
        )
    multitask_results = [
        result for result in results if result["test"]["multilabel_raw"] is not None
    ]
    if multitask_results:
        metrics = multitask_results[0]["test"]["multilabel_raw"]
        lines.extend(
            [
                "",
                "## Multi-Label Defect Results",
                "",
                f"- Macro-F1: {metrics['macro_f1']:.4f}",
                f"- Micro-F1: {metrics['micro_f1']:.4f}",
                f"- Mean average precision: {metrics['mean_average_precision']:.4f}",
                f"- Macro AUROC: {metrics['macro_roc_auc']:.4f}",
                f"- Exact-match ratio: {metrics['exact_match_ratio']:.4f}",
                "",
                "| Label | Prevalence | Precision | Recall | F1 | AP | AUROC |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label in DEFECT_LABELS:
            values = metrics["per_label"][label]
            lines.append(
                f"| {label} | {values['prevalence']:.4f} | {values['precision']:.4f} | "
                f"{values['recall']:.4f} | {values['f1']:.4f} | "
                f"{values['average_precision']:.4f} | {values['roc_auc']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "The auxiliary defect heads are useful for the deployed binary task only if the multi-task "
            "Teacher improves binary evidence performance or provides a better distillation target under "
            "the same Student architecture. Multi-label accuracy alone does not establish structural risk.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CODEBRIM binary and multi-task VSN Teachers.")
    parser.add_argument("--run-kinds", nargs="+", choices=RUN_KINDS, default=list(RUN_KINDS))
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "vsn_codebrim"
        / "codebrim_multitask_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vsn_codebrim_teachers",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--defect-loss-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [train_run(args, run_kind) for run_kind in args.run_kinds]
    write_comparison(args.output_dir, results)
    print(f"Wrote CODEBRIM Teacher comparison to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
