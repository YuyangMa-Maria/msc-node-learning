"""Compare private and shared risk modules over frozen node representations.

VSN and ASN retain their modality-specific CNNs; VBN supplies engineered proxy
features. Node-specific projections map these incompatible inputs to a common
64-D risk space, after which only structurally compatible higher layers are
allowed to share parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_asn_audio import read_samples as read_asn_samples
from train_asn_student import AsnStudentDataset, AsnStudentModel, limit_samples as limit_asn_samples
from train_vsn_binary import VsnBinaryDataset, build_transforms, limit_samples as limit_vsn_samples, read_samples as read_vsn_samples
from train_vsn_student_baseline import VsnStudentDwCnn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VSN_INDEX = PROJECT_ROOT / "experiments" / "vsn_binary" / "vsn_binary_index.csv"
DEFAULT_ASN_INDEX = PROJECT_ROOT / "experiments" / "asn_audio" / "asn_audio_index.csv"
DEFAULT_VBN_FEATURES = PROJECT_ROOT / "outputs" / "vbn_literature_guided_baselines" / "literature_features_max200000.json"
DEFAULT_VSN_CHECKPOINT = PROJECT_ROOT / "outputs" / "vsn_student_kd" / "vsn_student_dwcnn_s128_w1_kd_t4_a0.5" / "best.pt"
DEFAULT_ASN_CHECKPOINT = PROJECT_ROOT / "outputs" / "asn_student_baselines" / "asn_student_w0p5_d5p0s_scratch" / "best.pt"


@dataclass
class ModalityData:
    """Standardised train/validation/test arrays for one private node path."""

    name: str
    source_representation: str
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    scaler: StandardScaler


def json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def checkpoint_args(checkpoint: dict[str, object]) -> dict[str, object]:
    args = checkpoint.get("args", {})
    if hasattr(args, "__dict__"):
        return json_safe(dict(vars(args)))  # type: ignore[return-value]
    if isinstance(args, dict):
        return json_safe(args)  # type: ignore[return-value]
    return {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_modality_data(
    name: str,
    source_representation: str,
    split_x: dict[str, np.ndarray],
    split_y: dict[str, np.ndarray],
) -> ModalityData:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(split_x["train"]).astype(np.float32)
    return ModalityData(
        name=name,
        source_representation=source_representation,
        train_x=train_x,
        train_y=split_y["train"].astype(np.int64),
        val_x=scaler.transform(split_x["val"]).astype(np.float32),
        val_y=split_y["val"].astype(np.int64),
        test_x=scaler.transform(split_x["test"]).astype(np.float32),
        test_y=split_y["test"].astype(np.int64),
        scaler=scaler,
    )


def load_vsn_model(checkpoint_path: Path, device: torch.device) -> tuple[VsnStudentDwCnn, int, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint_args(checkpoint)
    image_size = int(checkpoint.get("image_size") or args.get("image_size", 128))
    model = VsnStudentDwCnn(width=float(args.get("width", 1.0)), dropout=float(args.get("dropout", 0.1)))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, image_size, args


def load_asn_model(checkpoint_path: Path, device: torch.device) -> tuple[AsnStudentModel, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint_args(checkpoint)
    model = AsnStudentModel(
        sample_rate=int(args.get("sample_rate", 16_000)),
        n_fft=int(args.get("n_fft", 1024)),
        hop_length=int(args.get("hop_length", 320)),
        n_mels=int(args.get("n_mels", 64)),
        width=float(args.get("width", 0.5)),
        dropout=float(args.get("dropout", 0.15)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, args


def student_training_mode(checkpoint_args: dict[str, object]) -> str:
    teacher = checkpoint_args.get("teacher_run_dir") or checkpoint_args.get("teacher_checkpoint")
    return "knowledge-distilled" if teacher else "from-scratch"


def extract_vsn_embeddings(
    index_path: Path,
    checkpoint_path: Path,
    cache_path: Path,
    train_limit: int,
    eval_limit: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        split_x = {split: cached[f"{split}_x"].astype(np.float32) for split in ["train", "val", "test"]}
        split_y = {split: cached[f"{split}_y"].astype(np.int64) for split in ["train", "val", "test"]}
        meta = json.loads(str(cached["meta"].item()))
        ckpt_args = meta.get("checkpoint_args", {})
        if isinstance(ckpt_args, dict):
            meta["source_representation"] = (
                f"frozen VSN {student_training_mode(ckpt_args)} student CNN penultimate embedding"
            )
        return split_x, split_y, meta

    model, image_size, ckpt_args = load_vsn_model(checkpoint_path, device)
    _, eval_tf = build_transforms(image_size)
    split_x: dict[str, np.ndarray] = {}
    split_y: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    with torch.no_grad():
        for split, limit, split_seed in [
            ("train", train_limit, seed),
            ("val", eval_limit, seed + 1),
            ("test", eval_limit, seed + 2),
        ]:
            samples = limit_vsn_samples(read_vsn_samples(index_path, split, None), limit, split_seed)
            loader = DataLoader(
                VsnBinaryDataset(samples, eval_tf),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )
            embeddings: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            for images, y in loader:
                images = images.to(device, non_blocking=True)
                feats = model.features(images)
                emb = F.adaptive_avg_pool2d(feats, 1).flatten(1)
                embeddings.append(emb.detach().cpu().numpy().astype(np.float32))
                labels.append(y.numpy().astype(np.int64))
            split_x[split] = np.concatenate(embeddings, axis=0)
            split_y[split] = np.concatenate(labels, axis=0)
            counts[split] = int(split_x[split].shape[0])
            print(f"[VSN] extracted {split}: {split_x[split].shape}")

    training_mode = student_training_mode(ckpt_args)
    meta = {
        "checkpoint": str(checkpoint_path),
        "image_size": image_size,
        "checkpoint_args": ckpt_args,
        "embedding_dim": int(split_x["train"].shape[1]),
        "sample_counts": counts,
        "source_representation": f"frozen VSN {training_mode} student CNN penultimate embedding",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{f"{s}_x": split_x[s] for s in split_x}, **{f"{s}_y": split_y[s] for s in split_y}, meta=json.dumps(meta))
    return split_x, split_y, meta


def extract_asn_embeddings(
    index_path: Path,
    checkpoint_path: Path,
    cache_path: Path,
    train_limit: int,
    eval_limit: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        split_x = {split: cached[f"{split}_x"].astype(np.float32) for split in ["train", "val", "test"]}
        split_y = {split: cached[f"{split}_y"].astype(np.int64) for split in ["train", "val", "test"]}
        meta = json.loads(str(cached["meta"].item()))
        ckpt_args = meta.get("checkpoint_args", {})
        if isinstance(ckpt_args, dict):
            meta["source_representation"] = (
                f"frozen ASN {student_training_mode(ckpt_args)} student CNN penultimate embedding"
            )
        return split_x, split_y, meta

    model, ckpt_args = load_asn_model(checkpoint_path, device)
    sample_rate = int(ckpt_args.get("sample_rate", 16_000))
    duration = float(ckpt_args.get("duration", 5.0))
    teacher_duration = float(ckpt_args.get("teacher_duration", duration))
    split_x: dict[str, np.ndarray] = {}
    split_y: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    with torch.no_grad():
        for split, limit, split_seed in [
            ("train", train_limit, seed + 3),
            ("val", eval_limit, seed + 4),
            ("test", eval_limit, seed + 5),
        ]:
            samples = limit_asn_samples(read_asn_samples(index_path, split, None), limit, split_seed)
            dataset = AsnStudentDataset(
                samples,
                sample_rate=sample_rate,
                student_duration=duration,
                teacher_duration=teacher_duration,
                train=False,
                seed=split_seed,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )
            embeddings: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            for student_waveform, _teacher_waveform, y in loader:
                student_waveform = student_waveform.to(device, non_blocking=True)
                spec = model.frontend(student_waveform)
                emb = model.encoder.net(spec).flatten(1)
                embeddings.append(emb.detach().cpu().numpy().astype(np.float32))
                labels.append(y.numpy().astype(np.int64))
            split_x[split] = np.concatenate(embeddings, axis=0)
            split_y[split] = np.concatenate(labels, axis=0)
            counts[split] = int(split_x[split].shape[0])
            print(f"[ASN] extracted {split}: {split_x[split].shape}")

    training_mode = student_training_mode(ckpt_args)
    meta = {
        "checkpoint": str(checkpoint_path),
        "duration": duration,
        "sample_rate": sample_rate,
        "checkpoint_args": ckpt_args,
        "embedding_dim": int(split_x["train"].shape[1]),
        "sample_counts": counts,
        "source_representation": f"frozen ASN {training_mode} student CNN penultimate embedding",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{f"{s}_x": split_x[s] for s in split_x}, **{f"{s}_y": split_y[s] for s in split_y}, meta=json.dumps(meta))
    return split_x, split_y, meta


def load_vbn_feature_embeddings(features_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    with features_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    X = np.asarray(payload["X"], dtype=np.float32)
    y = np.asarray(payload["y"], dtype=np.int64)
    rows = payload["rows"]
    split_x: dict[str, np.ndarray] = {}
    split_y: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        mask = np.asarray([row["split"] == split for row in rows], dtype=bool)
        split_x[split] = X[mask]
        split_y[split] = y[mask]
        counts[split] = int(np.sum(mask))
    meta = {
        "features_path": str(features_path),
        "embedding_dim": int(X.shape[1]),
        "feature_count": int(X.shape[1]),
        "sample_counts": counts,
        "source_representation": "VBN literature-guided time/frequency features from ORION AE proxy",
        "limitation": "proxy structural AE/time-series dataset; not true MPU6050 building vibration validation",
    }
    return split_x, split_y, meta


class SharedRiskHeadModel(nn.Module):
    """Private projections followed by one fully shared higher risk head."""

    def __init__(self, input_dims: dict[str, int], embedding_dim: int, projection_hidden: int, head_hidden: int, dropout: float) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, projection_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(projection_hidden, embedding_dim),
                    nn.ReLU(inplace=True),
                )
                for name, dim in input_dims.items()
            }
        )
        self.shared_head = nn.Sequential(
            nn.Linear(embedding_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        return self.shared_head(self.projections[modality](x)).flatten()


class SeparateRiskHeadsModel(nn.Module):
    """Parameter-matched comparison with no model-level collaboration."""

    def __init__(self, input_dims: dict[str, int], embedding_dim: int, projection_hidden: int, head_hidden: int, dropout: float) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, projection_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(projection_hidden, embedding_dim),
                    nn.ReLU(inplace=True),
                )
                for name, dim in input_dims.items()
            }
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(embedding_dim, head_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, 1),
                )
                for name in input_dims
            }
        )

    def forward(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        return self.heads[modality](self.projections[modality](x)).flatten()


class SharedTrunkNodeCalibratedModel(nn.Module):
    """Shared hidden risk features with a small private output calibrator."""

    def __init__(self, input_dims: dict[str, int], embedding_dim: int, projection_hidden: int, head_hidden: int, dropout: float) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, projection_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(projection_hidden, embedding_dim),
                    nn.ReLU(inplace=True),
                )
                for name, dim in input_dims.items()
            }
        )
        self.shared_trunk = nn.Sequential(
            nn.Linear(embedding_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.node_calibrators = nn.ModuleDict({name: nn.Linear(head_hidden, 1) for name in input_dims})

    def forward(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_trunk(self.projections[modality](x))
        return self.node_calibrators[modality](shared).flatten()


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def positive_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    positives = float(np.sum(y == 1))
    negatives = float(np.sum(y == 0))
    return torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32, device=device)


def best_threshold(probs: np.ndarray, y: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        f1 = f1_score(y, (probs >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(threshold)
    return best_t


def metric_row(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_positive": float(precision_score(y, pred, zero_division=0)),
        "recall_positive": float(recall_score(y, pred, zero_division=0)),
        "f1_positive": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": auc,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def predict_probs(model: nn.Module, modality: str, x: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            logits = model(modality, xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            chunks.append(probs.astype(np.float32))
    return np.concatenate(chunks, axis=0)


def tune_thresholds(model: nn.Module, data: dict[str, ModalityData], device: torch.device) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for name, modality_data in data.items():
        probs = predict_probs(model, name, modality_data.val_x, device)
        thresholds[name] = best_threshold(probs, modality_data.val_y)
    return thresholds


def evaluate_model(
    model: nn.Module,
    data: dict[str, ModalityData],
    split: str,
    device: torch.device,
    thresholds: dict[str, float],
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for name, modality_data in data.items():
        x = getattr(modality_data, f"{split}_x")
        y = getattr(modality_data, f"{split}_y")
        probs = predict_probs(model, name, x, device)
        results[name] = metric_row(y, probs, thresholds[name])
    return results


def macro_summary(metrics: dict[str, dict[str, float | int]]) -> dict[str, float]:
    return {
        "macro_accuracy": float(np.mean([float(row["accuracy"]) for row in metrics.values()])),
        "macro_f1_positive": float(np.mean([float(row["f1_positive"]) for row in metrics.values()])),
        "macro_recall_positive": float(np.mean([float(row["recall_positive"]) for row in metrics.values()])),
        "macro_precision_positive": float(np.mean([float(row["precision_positive"]) for row in metrics.values()])),
    }


def train_model(
    model: nn.Module,
    data: dict[str, ModalityData],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> tuple[nn.Module, list[dict[str, object]], dict[str, float]]:
    """Train each modality in turn and select the model by validation macro-F1."""
    model.to(device)
    loaders = {name: make_loader(d.train_x, d.train_y, batch_size, shuffle=True) for name, d in data.items()}
    criteria = {name: nn.BCEWithLogitsLoss(pos_weight=positive_weight(d.train_y, device)) for name, d in data.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, object]] = []
    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_thresholds: dict[str, float] = {}
    modality_names = list(data)
    # Cycling shorter loaders prevents the largest modality from being silently
    # under-used while still giving each node one update per optimisation step.
    steps_per_epoch = max(len(loader) for loader in loaders.values())
    for epoch in range(1, epochs + 1):
        model.train()
        loader_iters = {name: cycle(loaders[name]) for name in modality_names}
        total_loss = 0.0
        total_batches = 0
        for _ in range(steps_per_epoch):
            for name in modality_names:
                xb, yb = next(loader_iters[name])
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(name, xb)
                loss = criteria[name](logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                total_batches += 1
        thresholds = tune_thresholds(model, data, device)
        val_metrics = evaluate_model(model, data, "val", device, thresholds)
        val_macro = macro_summary(val_metrics)
        row: dict[str, object] = {
            "epoch": epoch,
            "loss": total_loss / max(total_batches, 1),
            "thresholds": thresholds,
            "val_macro": val_macro,
            "val": val_metrics,
        }
        history.append(row)
        if val_macro["macro_f1_positive"] > best_macro_f1:
            best_macro_f1 = val_macro["macro_f1_positive"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_thresholds = thresholds
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_thresholds


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def shared_parameter_count(model: nn.Module) -> int:
    if hasattr(model, "shared_head"):
        return sum(p.numel() for p in model.shared_head.parameters())  # type: ignore[attr-defined]
    if hasattr(model, "shared_trunk"):
        return sum(p.numel() for p in model.shared_trunk.parameters())  # type: ignore[attr-defined]
    return 0


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    test_rows = [row for row in rows if row["split"] == "test"]
    labels = [f"{row['method']}\n{row['modality']}" for row in test_rows]
    f1_values = [float(row["f1_positive"]) for row in test_rows]
    recall_values = [float(row["recall_positive"]) for row in test_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.5, 5.4), dpi=140)
    ax.bar(x - 0.18, f1_values, width=0.36, label="F1")
    ax.bar(x + 0.18, recall_values, width=0.36, label="Recall")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("CNN-Embedding Parameter Sharing: Held-out Test Results")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def method_description(method: str) -> str:
    descriptions = {
        "separate_heads": "Each node has its own projection and risk head after the frozen local representation.",
        "shared_risk_head": "Each node has a modality-specific projection, then all nodes use the same higher risk head.",
        "shared_trunk_node_calibrated": "Each node projects into a shared trunk, then uses a small node-specific calibration layer.",
    }
    return descriptions[method]


def write_report(
    path: Path,
    args: argparse.Namespace,
    metadata: dict[str, dict[str, object]],
    result_rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
    elapsed_seconds: float,
) -> None:
    test_rows = [row for row in result_rows if row["split"] == "test"]
    macro_rows = []
    for method in sorted({str(row["method"]) for row in test_rows}):
        rows = [row for row in test_rows if row["method"] == method]
        macro_rows.append(
            {
                "method": method,
                "macro_f1": float(np.mean([float(row["f1_positive"]) for row in rows])),
                "macro_recall": float(np.mean([float(row["recall_positive"]) for row in rows])),
                "macro_accuracy": float(np.mean([float(row["accuracy"]) for row in rows])),
            }
        )
    best = max(macro_rows, key=lambda row: row["macro_f1"])

    def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = []
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            body.append("| " + " | ".join(values) + " |")
        return "\n".join([header, sep, *body])

    per_node_rows = [
        {
            "method": row["method"],
            "node": row["modality"],
            "f1": row["f1_positive"],
            "recall": row["recall_positive"],
            "accuracy": row["accuracy"],
            "threshold": row["threshold"],
        }
        for row in test_rows
    ]
    memory_rows = [
        {
            "method": row["method"],
            "params": row["total_parameters"],
            "shared_params": row["shared_parameters"],
            "fp32_mb": row["fp32_mb"],
            "int8_est_mb": row["int8_weight_est_mb"],
        }
        for row in model_rows
    ]

    lines = [
        "# Parameter-Sharing Node Learning with Frozen CNN Embeddings",
        "",
        "## Purpose",
        "",
        "This experiment responds to the supervisor suggestion that the project should move beyond independent node models plus score-level fusion. It tests whether heterogeneous nodes can keep local lower encoders while sharing a higher risk-decision layer.",
        "",
        "## Representation Used",
        "",
        markdown_table(
            [
                {
                    "node": name,
                    "representation": meta["source_representation"],
                    "dim": meta["embedding_dim"],
                    "train/val/test": f"{meta['sample_counts']['train']}/{meta['sample_counts']['val']}/{meta['sample_counts']['test']}",
                }
                for name, meta in metadata.items()
            ],
            ["node", "representation", "dim", "train/val/test"],
        ),
        "",
        "## Compared Sharing Strategies",
        "",
        markdown_table(
            [{"method": method, "description": method_description(method)} for method in ["separate_heads", "shared_risk_head", "shared_trunk_node_calibrated"]],
            ["method", "description"],
        ),
        "",
        "## Held-Out Test Macro Results",
        "",
        markdown_table(macro_rows, ["method", "macro_f1", "macro_recall", "macro_accuracy"]),
        "",
        "## Held-Out Test Per-Node Results",
        "",
        markdown_table(per_node_rows, ["method", "node", "f1", "recall", "accuracy", "threshold"]),
        "",
        "## Trainable High-Layer Memory Estimate",
        "",
        markdown_table(memory_rows, ["method", "params", "shared_params", "fp32_mb", "int8_est_mb"]),
        "",
        "## Communication / Sharing Interpretation",
        "",
        f"- The common projected embedding is {args.embedding_dim} values: {args.embedding_dim * 4} bytes in FP32 or about {args.embedding_dim} bytes if quantized to INT8.",
        "- VSN and ASN lower CNN encoders remain local to their nodes; only higher risk-decision parameters are compared as shared components.",
        "- This is closer to model-level collaboration than previous score fusion because the experiment shares a trainable decision layer rather than only averaging final risk scores.",
        "",
        "## Main Finding",
        "",
        f"- Best held-out macro F1: `{best['method']}` with macro F1 {best['macro_f1']:.4f}, macro recall {best['macro_recall']:.4f}, and macro accuracy {best['macro_accuracy']:.4f}.",
        "- The result should be interpreted as a software feasibility test for higher-layer parameter sharing, not as a physical distributed deployment.",
        "",
        "## Claims to Avoid / Cautious Wording",
        "",
        "- Do not claim that low-level modality encoders are shared across VSN, ASN, and VBN; their raw inputs are incompatible.",
        "- Do not claim physical cross-device layer execution yet; this is an offline software simulation of shared higher layers.",
        "- Do not claim real vibration validation; VBN still uses an ORION AE proxy with literature-guided features.",
        "- Do not claim real collapse probability; the output remains a structural risk proxy.",
        "- TinyML memory values are parameter-size estimates for the high-layer sharing module, not measured MCU SRAM/tensor-arena profiling.",
        "",
        "## Recommended Next Step",
        "",
        "- Keep independent node models and score fusion as baselines.",
        "- Add this parameter-sharing experiment as an intermediate contribution: local encoders plus shared higher risk layer.",
        "- If this remains stable, the next research upgrade is a small federated or continual update experiment on the shared head only, while keeping raw data local.",
        "",
        f"Runtime: {elapsed_seconds:.1f} seconds.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate higher-layer parameter sharing using frozen VSN/ASN CNN embeddings and VBN proxy features.")
    parser.add_argument("--vsn-index", type=Path, default=DEFAULT_VSN_INDEX)
    parser.add_argument("--asn-index", type=Path, default=DEFAULT_ASN_INDEX)
    parser.add_argument("--vbn-features", type=Path, default=DEFAULT_VBN_FEATURES)
    parser.add_argument("--vsn-checkpoint", type=Path, default=DEFAULT_VSN_CHECKPOINT)
    parser.add_argument("--asn-checkpoint", type=Path, default=DEFAULT_ASN_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "parameter_sharing_cnn_embeddings")
    parser.add_argument("--vsn-train-limit", type=int, default=3000)
    parser.add_argument("--vsn-eval-limit", type=int, default=1000)
    parser.add_argument("--asn-train-limit", type=int, default=1600)
    parser.add_argument("--asn-eval-limit", type=int, default=500)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--projection-hidden", type=int, default=96)
    parser.add_argument("--head-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "embedding_cache"
    device = torch.device(args.device)
    start = time.perf_counter()

    vsn_x, vsn_y, vsn_meta = extract_vsn_embeddings(
        args.vsn_index,
        args.vsn_checkpoint,
        cache_dir / f"vsn_cnn_embeddings_t{args.vsn_train_limit}_e{args.vsn_eval_limit}_s{args.seed}.npz",
        args.vsn_train_limit,
        args.vsn_eval_limit,
        args.extract_batch_size,
        args.seed,
        device,
        args.num_workers,
    )
    asn_x, asn_y, asn_meta = extract_asn_embeddings(
        args.asn_index,
        args.asn_checkpoint,
        cache_dir / f"asn_cnn_embeddings_t{args.asn_train_limit}_e{args.asn_eval_limit}_s{args.seed}.npz",
        args.asn_train_limit,
        args.asn_eval_limit,
        args.extract_batch_size,
        args.seed,
        device,
        args.num_workers,
    )
    vbn_x, vbn_y, vbn_meta = load_vbn_feature_embeddings(args.vbn_features)

    data = {
        "vsn": make_modality_data("vsn", str(vsn_meta["source_representation"]), vsn_x, vsn_y),
        "asn": make_modality_data("asn", str(asn_meta["source_representation"]), asn_x, asn_y),
        "vbn": make_modality_data("vbn", str(vbn_meta["source_representation"]), vbn_x, vbn_y),
    }
    metadata = {"vsn": vsn_meta, "asn": asn_meta, "vbn": vbn_meta}
    input_dims = {name: d.train_x.shape[1] for name, d in data.items()}
    methods = {
        "separate_heads": SeparateRiskHeadsModel(input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout),
        "shared_risk_head": SharedRiskHeadModel(input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout),
        "shared_trunk_node_calibrated": SharedTrunkNodeCalibratedModel(input_dims, args.embedding_dim, args.projection_hidden, args.head_hidden, args.dropout),
    }

    result_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, object]]] = {}
    for method, model in methods.items():
        print(f"[train] {method}")
        trained, history, thresholds = train_model(model, data, device, args.epochs, args.batch_size, args.lr, args.weight_decay)
        histories[method] = history
        for split in ["val", "test"]:
            metrics = evaluate_model(trained, data, split, device, thresholds)
            for modality, row in metrics.items():
                result_rows.append({"method": method, "split": split, "modality": modality, **row})
        params = parameter_count(trained)
        shared_params = shared_parameter_count(trained)
        model_rows.append(
            {
                "method": method,
                "total_parameters": params,
                "shared_parameters": shared_params,
                "fp32_mb": params * 4 / (1024 * 1024),
                "int8_weight_est_mb": params / (1024 * 1024),
                "embedding_payload": f"{args.embedding_dim} floats = {args.embedding_dim * 4} bytes FP32, {args.embedding_dim} bytes INT8",
            }
        )
        torch.save({"model_state": trained.state_dict(), "thresholds": thresholds, "args": vars(args)}, args.output_dir / f"{method}.pt")
        write_jsonl(args.output_dir / f"history_{method}.jsonl", history)

    elapsed = time.perf_counter() - start
    write_csv(args.output_dir / "metrics.csv", result_rows)
    write_csv(args.output_dir / "model_sizes.csv", model_rows)
    plot_results(result_rows, args.output_dir / "cnn_embedding_parameter_sharing_results.png")
    summary = {
        "task": "parameter_sharing_node_learning_with_frozen_cnn_embeddings",
        "metadata": metadata,
        "metrics": result_rows,
        "model_sizes": model_rows,
        "elapsed_seconds": elapsed,
        "cautions": [
            "VSN/ASN use frozen trained encoders, but VBN still uses proxy literature-guided features.",
            "The experiment shares higher risk-decision layers in software; it is not physical split inference across hardware yet.",
            "Risk score remains a proxy, not certified collapse probability.",
        ],
    }
    save_json(args.output_dir / "summary.json", summary)
    report_path = args.output_dir / "PARAMETER_SHARING_CNN_EMBEDDING_REPORT.md"
    write_report(report_path, args, metadata, result_rows, model_rows, elapsed)

    final_package = PROJECT_ROOT / "outputs" / "reports" / "final_results_package"
    if final_package.exists():
        shutil.copy2(report_path, final_package / report_path.name)
        shutil.copy2(args.output_dir / "metrics.csv", final_package / "parameter_sharing_cnn_embedding_metrics.csv")
        shutil.copy2(args.output_dir / "model_sizes.csv", final_package / "parameter_sharing_cnn_embedding_model_sizes.csv")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
