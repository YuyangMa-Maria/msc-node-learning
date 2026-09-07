"""Compare literature-guided feature and temporal baselines for the VBN proxy.

Time statistics, spectral descriptors and shallow classifiers provide an
interpretable low-cost reference before 1-D neural models are considered. All
methods use the same deterministic ORION split and binary torque mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - reported at runtime
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "experiments" / "vbn_orion" / "vbn_orion_index.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def read_rows(index_path: Path, split: str | None = None, binary_only: bool = True) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with index_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split and row["split"] != split:
                continue
            if binary_only and row["use_binary"] != "1":
                continue
            rows.append(row)
    return rows


def load_channels(path: Path, max_points: int) -> dict[str, np.ndarray]:
    data = loadmat(path)
    channels: dict[str, np.ndarray] = {}
    for channel in ["A", "B", "C", "D"]:
        x = np.asarray(data[channel]).reshape(-1).astype(np.float32)
        if x.size > max_points:
            step = max(1, x.size // max_points)
            x = x[::step][:max_points]
        channels[channel] = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return channels


def safe_float(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(value)


def spectral_features(x: np.ndarray, prefix: str, sample_rate: float) -> dict[str, float]:
    x = x.astype(np.float64)
    x = x - float(np.mean(x))
    if x.size < 4:
        return {}
    spectrum = np.abs(np.fft.rfft(x))
    power = spectrum * spectrum
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    total_power = float(np.sum(power)) + 1e-12
    probs = power / total_power
    dominant_idx = int(np.argmax(power[1:]) + 1) if power.size > 1 else 0
    centroid = float(np.sum(freqs * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / math.log(max(len(probs), 2)))
    cumulative = np.cumsum(power) / total_power

    def rolloff(q: float) -> float:
        idx = int(np.searchsorted(cumulative, q, side="left"))
        idx = min(idx, freqs.size - 1)
        return float(freqs[idx])

    bands = [
        (0, 25_000),
        (25_000, 50_000),
        (50_000, 100_000),
        (100_000, 200_000),
        (200_000, 400_000),
        (400_000, 800_000),
        (800_000, 1_500_000),
        (1_500_000, sample_rate / 2),
    ]
    feats = {
        f"{prefix}_fft_dominant_freq": safe_float(freqs[dominant_idx]),
        f"{prefix}_fft_dominant_power_ratio": safe_float(power[dominant_idx] / total_power),
        f"{prefix}_fft_centroid": safe_float(centroid),
        f"{prefix}_fft_bandwidth": safe_float(bandwidth),
        f"{prefix}_fft_entropy": safe_float(entropy),
        f"{prefix}_fft_rolloff_50": safe_float(rolloff(0.50)),
        f"{prefix}_fft_rolloff_85": safe_float(rolloff(0.85)),
        f"{prefix}_fft_rolloff_95": safe_float(rolloff(0.95)),
    }
    for idx, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        feats[f"{prefix}_fft_band_{idx}_ratio"] = safe_float(float(np.sum(power[mask]) / total_power))
    return feats


def autocorr_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = x.astype(np.float64)
    x = x - float(np.mean(x))
    denom = float(np.sum(x * x)) + 1e-12
    feats: dict[str, float] = {}
    for lag in [1, 5, 10, 25, 50, 100]:
        if x.size <= lag:
            feats[f"{prefix}_autocorr_lag_{lag}"] = 0.0
        else:
            feats[f"{prefix}_autocorr_lag_{lag}"] = safe_float(float(np.sum(x[:-lag] * x[lag:]) / denom))
    return feats


def time_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = x.astype(np.float64)
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    peak = float(np.max(abs_x) + 1e-12)
    mean_abs = float(np.mean(abs_x) + 1e-12)
    sqrt_abs_mean = float(np.mean(np.sqrt(abs_x)) + 1e-12)
    segments = np.array_split(x, 10)
    segment_rms = np.asarray([np.sqrt(np.mean(seg * seg)) for seg in segments], dtype=np.float64)
    zcr = float(np.mean(np.diff(np.signbit(x)) != 0)) if x.size > 1 else 0.0
    feats = {
        f"{prefix}_mean": safe_float(float(np.mean(x))),
        f"{prefix}_median": safe_float(float(np.median(x))),
        f"{prefix}_std": safe_float(float(np.std(x))),
        f"{prefix}_var": safe_float(float(np.var(x))),
        f"{prefix}_rms": safe_float(rms),
        f"{prefix}_abs_mean": safe_float(mean_abs),
        f"{prefix}_energy": safe_float(float(np.mean(x * x))),
        f"{prefix}_peak": safe_float(peak),
        f"{prefix}_peak_to_peak": safe_float(float(np.ptp(x))),
        f"{prefix}_p75_abs": safe_float(float(np.percentile(abs_x, 75))),
        f"{prefix}_p90_abs": safe_float(float(np.percentile(abs_x, 90))),
        f"{prefix}_p95_abs": safe_float(float(np.percentile(abs_x, 95))),
        f"{prefix}_p99_abs": safe_float(float(np.percentile(abs_x, 99))),
        f"{prefix}_iqr": safe_float(float(np.percentile(x, 75) - np.percentile(x, 25))),
        f"{prefix}_skew": safe_float(float(skew(x, bias=False))),
        f"{prefix}_kurtosis": safe_float(float(kurtosis(x, fisher=True, bias=False))),
        f"{prefix}_zero_crossing_rate": safe_float(zcr),
        f"{prefix}_crest_factor": safe_float(peak / rms),
        f"{prefix}_shape_factor": safe_float(rms / mean_abs),
        f"{prefix}_impulse_factor": safe_float(peak / mean_abs),
        f"{prefix}_clearance_factor": safe_float(peak / (sqrt_abs_mean * sqrt_abs_mean)),
        f"{prefix}_segment_rms_mean": safe_float(float(np.mean(segment_rms))),
        f"{prefix}_segment_rms_std": safe_float(float(np.std(segment_rms))),
        f"{prefix}_segment_rms_max": safe_float(float(np.max(segment_rms))),
        f"{prefix}_segment_rms_slope": safe_float(float(np.polyfit(np.arange(segment_rms.size), segment_rms, 1)[0])),
    }
    feats.update(autocorr_features(x, prefix))
    return feats


def extract_literature_features(path: Path, max_points: int, sample_rate: float) -> dict[str, float]:
    channels = load_channels(path, max_points=max_points)
    feats: dict[str, float] = {}
    rms_values = []
    peak_values = []
    entropy_values = []
    for channel, x in channels.items():
        t = time_features(x, channel)
        s = spectral_features(x, channel, sample_rate)
        feats.update(t)
        feats.update(s)
        rms_values.append(t[f"{channel}_rms"])
        peak_values.append(t[f"{channel}_peak"])
        entropy_values.append(s[f"{channel}_fft_entropy"])
    feats["cross_channel_rms_mean"] = safe_float(float(np.mean(rms_values)))
    feats["cross_channel_rms_std"] = safe_float(float(np.std(rms_values)))
    feats["cross_channel_peak_mean"] = safe_float(float(np.mean(peak_values)))
    feats["cross_channel_peak_std"] = safe_float(float(np.std(peak_values)))
    feats["cross_channel_entropy_mean"] = safe_float(float(np.mean(entropy_values)))
    feats["cross_channel_entropy_std"] = safe_float(float(np.std(entropy_values)))
    for first, second in [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]:
        x = channels[first].astype(np.float64)
        y = channels[second].astype(np.float64)
        n = min(x.size, y.size)
        if n < 4:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x[:n], y[:n])[0, 1])
        feats[f"cross_corr_{first}_{second}"] = safe_float(corr)
    return feats


def build_feature_table(rows: list[dict[str, str]], cache_path: Path, max_points: int, sample_rate: float) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return (
            np.asarray(cached["X"], dtype=np.float32),
            np.asarray(cached["y"], dtype=np.int64),
            list(cached["feature_names"]),
            list(cached["rows"]),
        )

    feature_rows: list[dict[str, float]] = []
    labels: list[int] = []
    for idx, row in enumerate(rows, 1):
        path = PROJECT_ROOT / row["path"]
        feats = extract_literature_features(path, max_points=max_points, sample_rate=sample_rate)
        feature_rows.append(feats)
        labels.append(int(row["label"]))
        print(f"feature extraction {idx}/{len(rows)} {path.name}", flush=True)
    feature_names = sorted(feature_rows[0])
    X = np.asarray([[feature_row.get(name, 0.0) for name in feature_names] for feature_row in feature_rows], dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"feature_names": feature_names, "X": X.tolist(), "y": y.tolist(), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return X, y, feature_names, rows


def envelope_sequence(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    if values.size < sequence_length:
        padded = np.pad(values, (0, sequence_length - values.size))
        rms = np.abs(padded)
        peak = np.abs(padded)
        return rms.astype(np.float32), peak.astype(np.float32)
    block = max(1, values.size // sequence_length)
    usable = block * sequence_length
    blocks = values[:usable].reshape(sequence_length, block).astype(np.float32)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    peak = np.max(np.abs(blocks), axis=1)
    return rms.astype(np.float32), peak.astype(np.float32)


def build_sequence_table(rows: list[dict[str, str]], cache_path: Path, sequence_length: int, mode: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["X"].astype(np.float32), cached["y"].astype(np.int64), list(cached["rows"].tolist())

    X = []
    y = []
    for idx, row in enumerate(rows, 1):
        path = PROJECT_ROOT / row["path"]
        data = loadmat(path)
        channels = []
        for channel in ["A", "B", "C", "D"]:
            values = np.asarray(data[channel]).reshape(-1).astype(np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            if mode == "uniform":
                if values.size > sequence_length:
                    positions = np.linspace(0, values.size - 1, sequence_length).astype(np.int64)
                    values = values[positions]
                elif values.size < sequence_length:
                    values = np.pad(values, (0, sequence_length - values.size))
                std = float(np.std(values)) + 1e-6
                values = (values - float(np.mean(values))) / std
                channels.append(values)
            elif mode == "envelope":
                rms, peak = envelope_sequence(values, sequence_length)
                for sequence in [rms, peak]:
                    sequence = np.log1p(sequence)
                    std = float(np.std(sequence)) + 1e-6
                    sequence = (sequence - float(np.mean(sequence))) / std
                    channels.append(sequence)
            else:
                raise ValueError(f"Unsupported sequence mode: {mode}")
        X.append(np.stack(channels, axis=0))
        y.append(int(row["label"]))
        print(f"sequence extraction {idx}/{len(rows)} {path.name}", flush=True)
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X_arr, y=y_arr, rows=np.asarray(rows, dtype=object))
    return X_arr, y_arr, rows


def positive_probs(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"Model {model} does not expose probabilities or decision scores.")


def best_threshold_from_probs(probs: np.ndarray, y: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (probs >= threshold).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


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


def summarize(values: list[float]) -> tuple[float, float]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        return float(clean[0]), 0.0
    return float(clean.mean()), float(clean.std(ddof=1))


def make_models(seed: int) -> dict[str, object]:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed),
        ),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        "random_forest": RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=2, class_weight="balanced", random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, max_depth=6, min_samples_leaf=1, class_weight="balanced", random_state=seed),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, learning_rate_init=0.001, max_iter=2000, early_stopping=False, random_state=seed),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, min_samples_leaf=2, max_leaf_nodes=5, l2_regularization=0.05, random_state=seed),
    }


def pickle_size_mb(model: object) -> float:
    return len(pickle.dumps(model)) / (1024 * 1024)


def run_heldout_baselines(X: np.ndarray, y: np.ndarray, rows: list[dict[str, str]], seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    split_indices = {split: [idx for idx, row in enumerate(rows) if row["split"] == split] for split in ["train", "val", "test"]}
    X_train, y_train = X[split_indices["train"]], y[split_indices["train"]]
    X_val, y_val = X[split_indices["val"]], y[split_indices["val"]]
    X_test, y_test = X[split_indices["test"]], y[split_indices["test"]]
    results: list[dict[str, object]] = []
    fitted: dict[str, object] = {}
    for name, model in make_models(seed).items():
        model.fit(X_train, y_train)
        val_probs = positive_probs(model, X_val)
        threshold = best_threshold_from_probs(val_probs, y_val)
        for split, split_X, split_y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            probs = positive_probs(model, split_X)
            m = metric_row(split_y, probs, threshold)
            results.append({"model": name, "split": split, "model_size_mb_pickle": pickle_size_mb(model), **m})
        fitted[name] = model
        print(f"held-out {name} done")
    return results, fitted


def run_cv_baselines(X: np.ndarray, y: np.ndarray, seed: int, folds: int, repeats: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    fold_rows: list[dict[str, object]] = []
    for model_name in make_models(seed):
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), 1):
            model = make_models(seed + fold_idx)[model_name]
            model.fit(X[train_idx], y[train_idx])
            train_probs = positive_probs(model, X[train_idx])
            threshold = best_threshold_from_probs(train_probs, y[train_idx])
            probs = positive_probs(model, X[test_idx])
            m = metric_row(y[test_idx], probs, threshold)
            fold_rows.append({"model": model_name, "fold": fold_idx, **m})
        print(f"cv {model_name} done")
    summary_rows: list[dict[str, object]] = []
    for model_name in make_models(seed):
        model_folds = [row for row in fold_rows if row["model"] == model_name]
        summary: dict[str, object] = {"model": model_name, "folds": len(model_folds)}
        for metric_name in ["accuracy", "precision_positive", "recall_positive", "f1_positive", "roc_auc"]:
            mean, std = summarize([float(row[metric_name]) for row in model_folds])
            summary[f"{metric_name}_mean"] = mean
            summary[f"{metric_name}_std"] = std
        summary_rows.append(summary)
    return fold_rows, summary_rows


class VBNSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False) -> None:
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self) -> int:
        return int(self.y.size)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()
        if self.augment:
            scale = np.random.uniform(0.9, 1.1, size=(x.shape[0], 1)).astype(np.float32)
            noise = np.random.normal(0.0, 0.015, size=x.shape).astype(np.float32)
            x = x * scale + noise
        return torch.from_numpy(x), torch.tensor(float(self.y[idx]), dtype=torch.float32)


class TinyVBN1DCNN(nn.Module):
    def __init__(self, input_channels: int = 4, width: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, width, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, kernel_size=7, stride=2, padding=3, groups=width, bias=False),
            nn.Conv1d(width, width * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(width * 2, width * 2, kernel_size=7, stride=2, padding=3, groups=width * 2, bias=False),
            nn.Conv1d(width * 2, width * 3, kernel_size=1, bias=False),
            nn.BatchNorm1d(width * 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(width * 3, 1)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.classifier(x).squeeze(-1)


@dataclass
class CNNResult:
    metrics: dict[str, object]
    history: list[dict[str, float]]
    param_count: int
    state_dict_mb: float
    int8_weight_est_mb: float
    inference_ms_per_sample: float


def evaluate_cnn(model, X: np.ndarray, y: np.ndarray, device: str, threshold: float) -> dict[str, object]:
    model.eval()
    probs = []
    loader = DataLoader(VBNSequenceDataset(X, y), batch_size=16, shuffle=False)
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    p = np.concatenate(probs)
    return metric_row(y, p, threshold)


def run_tiny_cnn(X_seq: np.ndarray, y: np.ndarray, rows: list[dict[str, str]], output_dir: Path, seed: int, epochs: int, batch_size: int, lr: float, width: int) -> CNNResult:
    if torch is None:
        raise RuntimeError(f"PyTorch is unavailable: {TORCH_IMPORT_ERROR}")
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_indices = {split: [idx for idx, row in enumerate(rows) if row["split"] == split] for split in ["train", "val", "test"]}
    X_train, y_train = X_seq[split_indices["train"]], y[split_indices["train"]]
    X_val, y_val = X_seq[split_indices["val"]], y[split_indices["val"]]
    X_test, y_test = X_seq[split_indices["test"]], y[split_indices["test"]]
    model = TinyVBN1DCNN(input_channels=int(X_seq.shape[1]), width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    pos = max(float(np.sum(y_train == 1)), 1.0)
    neg = max(float(np.sum(y_train == 0)), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device))
    train_loader = DataLoader(VBNSequenceDataset(X_train, y_train, augment=True), batch_size=batch_size, shuffle=True)
    history: list[dict[str, float]] = []
    best_state = None
    best_val_f1 = -1.0
    best_threshold = 0.5
    patience = 20
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(torch.from_numpy(X_val).to(device))).cpu().numpy()
        threshold = best_threshold_from_probs(val_probs, y_val)
        val_metrics = metric_row(y_val, val_probs, threshold)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_f1": float(val_metrics["f1_positive"]), "val_recall": float(val_metrics["recall_positive"]), "threshold": threshold}
        history.append(row)
        if float(val_metrics["f1_positive"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1_positive"])
            best_threshold = threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    val_metrics = evaluate_cnn(model, X_val, y_val, device, best_threshold)
    test_metrics = evaluate_cnn(model, X_test, y_test, device, best_threshold)
    start = time.perf_counter()
    _ = evaluate_cnn(model, X_test, y_test, device, best_threshold)
    elapsed = time.perf_counter() - start
    inference_ms = elapsed / max(len(y_test), 1) * 1000.0
    param_count = sum(p.numel() for p in model.parameters())
    state_dict_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    int8_weight_est_bytes = sum(p.numel() for p in model.parameters())
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "width": width, "input_channels": int(X_seq.shape[1]), "sequence_length": X_seq.shape[-1], "threshold": best_threshold}, output_dir / "tiny_1d_cnn_best.pt")
    return CNNResult(
        metrics={"model": "tiny_1d_cnn", "split": "val", **val_metrics, "device": device} | {"test": test_metrics},
        history=history,
        param_count=int(param_count),
        state_dict_mb=state_dict_bytes / (1024 * 1024),
        int8_weight_est_mb=int8_weight_est_bytes / (1024 * 1024),
        inference_ms_per_sample=inference_ms,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_cv(summary_rows: list[dict[str, object]], path: Path) -> None:
    metrics = ["recall_positive", "f1_positive", "roc_auc"]
    models = [str(row["model"]) for row in summary_rows]
    x = np.arange(len(metrics))
    width = 0.82 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=140)
    for idx, row in enumerate(summary_rows):
        means = [float(row[f"{metric}_mean"]) for metric in metrics]
        stds = [float(row[f"{metric}_std"]) for metric in metrics]
        ax.bar(x + idx * width, means, width=width, yerr=stds, capsize=2, label=str(row["model"]))
    ax.set_xticks(x + width * (len(models) - 1) / 2, ["Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heldout(rows: list[dict[str, object]], path: Path) -> None:
    test_rows = [row for row in rows if row.get("split") == "test"]
    models = [str(row["model"]) for row in test_rows]
    f1 = [float(row["f1_positive"]) for row in test_rows]
    recall = [float(row["recall_positive"]) for row in test_rows]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=140)
    ax.bar(x - 0.18, f1, width=0.36, label="F1")
    ax.bar(x + 0.18, recall, width=0.36, label="Recall")
    ax.set_xticks(x, models, rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def feature_importance_rows(model: object, feature_names: list[str], top_k: int) -> list[dict[str, object]]:
    candidate = model
    if hasattr(model, "named_steps"):
        candidate = list(model.named_steps.values())[-1]
    values = getattr(candidate, "feature_importances_", None)
    if values is None:
        return []
    rows = [
        {"rank": idx + 1, "feature": feature, "importance": float(value)}
        for idx, (feature, value) in enumerate(sorted(zip(feature_names, values), key=lambda item: item[1], reverse=True)[:top_k])
    ]
    return rows


def plot_importance(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    top = list(reversed(rows[:20]))
    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=140)
    ax.barh([str(row["feature"]) for row in top], [float(row["importance"]) for row in top])
    ax.set_title("VBN literature-guided feature importance")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fmt(value: object) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value_float):
        return "nan"
    return f"{value_float:.4f}"


def build_report(
    output_dir: Path,
    feature_count: int,
    sample_counts: dict[str, int],
    heldout_rows: list[dict[str, object]],
    cv_summary: list[dict[str, object]],
    cnn_result: CNNResult | None,
    importance_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# VBN Literature-Guided Baseline Upgrade",
        "",
        "## Purpose",
        "",
        "This experiment strengthens the vibration/time-series node (VBN) using methods commonly discussed in vibration-based structural health monitoring literature. It adds interpretable time-domain features, FFT-based frequency features, shallow ML baselines and a TinyML-oriented raw-signal 1D CNN.",
        "",
        "The dataset remains the ORION AE proxy dataset. Therefore, these results improve the methodological defensibility of the VBN node, but they do not validate a real MPU6050 building-vibration deployment.",
        "",
        "## Setup",
        "",
        f"- Binary proxy task: low torque (05/10 cNm) vs high torque (40/50/60 cNm).",
        f"- Moderate torque (20/30 cNm) is excluded from binary training.",
        f"- Feature extraction max points: {args.max_points}.",
        f"- 1D CNN sequence length: {args.sequence_length}.",
        f"- 1D CNN sequence mode: {args.sequence_mode}.",
        f"- Feature count: {feature_count}.",
        f"- Samples: train={sample_counts['train']}, val={sample_counts['val']}, test={sample_counts['test']}.",
        "",
        "## Held-Out Split Results",
        "",
        "| Model | Split | F1 | Recall | Precision | ROC-AUC | Accuracy | Threshold | Size estimate (MB) | Confusion TN/FP/FN/TP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in heldout_rows:
        lines.append(
            f"| {row['model']} | {row['split']} | {fmt(row['f1_positive'])} | {fmt(row['recall_positive'])} | {fmt(row['precision_positive'])} | {fmt(row['roc_auc'])} | {fmt(row['accuracy'])} | {fmt(row['threshold'])} | {fmt(row.get('model_size_mb_pickle', ''))} | {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |"
        )
    if cnn_result is not None:
        test = cnn_result.metrics["test"]
        lines.append(
            f"| tiny_1d_cnn | test | {fmt(test['f1_positive'])} | {fmt(test['recall_positive'])} | {fmt(test['precision_positive'])} | {fmt(test['roc_auc'])} | {fmt(test['accuracy'])} | {fmt(test['threshold'])} | {cnn_result.state_dict_mb:.4f} | {test['tn']}/{test['fp']}/{test['fn']}/{test['tp']} |"
        )
    lines.extend(
        [
            "",
            "## Repeated Cross-Validation Summary",
            "",
            "| Model | F1 mean | F1 std | Recall mean | Recall std | ROC-AUC mean | ROC-AUC std |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cv_summary:
        lines.append(
            f"| {row['model']} | {fmt(row['f1_positive_mean'])} | {fmt(row['f1_positive_std'])} | {fmt(row['recall_positive_mean'])} | {fmt(row['recall_positive_std'])} | {fmt(row['roc_auc_mean'])} | {fmt(row['roc_auc_std'])} |"
        )
    lines.extend(
        [
            "",
            "## Tiny 1D CNN",
            "",
        ]
    )
    if cnn_result is None:
        lines.append("The Tiny 1D CNN was not run because PyTorch was unavailable.")
    else:
        lines.extend(
            [
                f"- Parameters: {cnn_result.param_count}",
                f"- FP32 state dict estimate: {cnn_result.state_dict_mb:.4f} MB",
                f"- INT8 weight-only estimate: {cnn_result.int8_weight_est_mb:.4f} MB",
                f"- Test inference estimate: {cnn_result.inference_ms_per_sample:.4f} ms/sample on the current runtime",
            ]
        )
    lines.extend(
        [
            "",
            "## Top ExtraTrees Feature Importances",
            "",
            "| Rank | Feature | Importance |",
            "| ---: | --- | ---: |",
        ]
    )
    for row in importance_rows[:15]:
        lines.append(f"| {row['rank']} | {row['feature']} | {float(row['importance']):.6f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The VBN node is now evaluated with literature-grounded time-domain and FFT-based features rather than a single lightweight baseline only.",
            "- Repeated CV is important because the ORION binary proxy set is very small. A single train/val/test split can look overly optimistic or pessimistic.",
            "- The Tiny 1D CNN is included because vibration signals are naturally one-dimensional; this is more aligned with TinyML than converting signals into 2D images.",
            "- LightGBM and DWT/SWT were not run because the current environment does not include `lightgbm` or `pywt`; they can be added later if needed.",
            "",
            "## Claims to Avoid",
            "",
            "- Do not claim this is real MPU6050 building-vibration validation.",
            "- Do not claim VBN is validated for collapse prediction.",
            "- Do not claim high CV performance proves field robustness; this is still a small proxy dataset.",
            "- Do not claim 1D CNN is automatically better; select the final route based on F1/recall, size and stability.",
            "",
            "## Generated Files",
            "",
            "- `heldout_metrics.csv`",
            "- `cv_folds.csv`",
            "- `cv_summary.csv`",
            "- `feature_importance_extra_trees.csv`",
            "- `tiny_1d_cnn_history.csv`",
            "- `vbn_literature_guided_summary.json`",
            "- `heldout_f1_recall.png`",
            "- `cv_f1_recall_auc.png`",
            "- `feature_importance_extra_trees.png`",
        ]
    )
    report = "\n".join(lines)
    (output_dir / "VBN_Literature_Guided_Baseline_Report.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run literature-guided VBN proxy baselines.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vbn_literature_guided_baselines")
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--sample-rate", type=float, default=5_000_000.0)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--sequence-mode", choices=["envelope", "uniform"], default="envelope")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cnn-epochs", type=int, default=120)
    parser.add_argument("--cnn-width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.index, binary_only=True)
    feature_cache = args.output_dir / f"literature_features_max{args.max_points}.json"
    X, y, feature_names, cached_rows = build_feature_table(rows, feature_cache, args.max_points, args.sample_rate)
    sample_counts = {split: sum(1 for row in cached_rows if row["split"] == split) for split in ["train", "val", "test"]}

    heldout_rows, fitted = run_heldout_baselines(X, y, cached_rows, args.seed)
    cv_folds, cv_summary = run_cv_baselines(X, y, args.seed, args.folds, args.repeats)

    importance_model = fitted.get("extra_trees")
    importance_rows = feature_importance_rows(importance_model, feature_names, top_k=40) if importance_model is not None else []

    cnn_result = None
    if torch is not None:
        sequence_cache = args.output_dir / f"sequence_{args.sequence_mode}_len{args.sequence_length}.npz"
        X_seq, y_seq, seq_rows = build_sequence_table(cached_rows, sequence_cache, args.sequence_length, args.sequence_mode)
        cnn_result = run_tiny_cnn(
            X_seq,
            y_seq,
            seq_rows,
            args.output_dir,
            args.seed,
            args.cnn_epochs,
            batch_size=8,
            lr=1e-3,
            width=args.cnn_width,
        )
        write_csv(args.output_dir / "tiny_1d_cnn_history.csv", cnn_result.history)
    else:
        print(f"Skipping Tiny 1D CNN because PyTorch import failed: {TORCH_IMPORT_ERROR}", file=sys.stderr)

    write_csv(args.output_dir / "heldout_metrics.csv", heldout_rows)
    write_csv(args.output_dir / "cv_folds.csv", cv_folds)
    write_csv(args.output_dir / "cv_summary.csv", cv_summary)
    write_csv(args.output_dir / "feature_importance_extra_trees.csv", importance_rows)
    plot_heldout(heldout_rows, args.output_dir / "heldout_f1_recall.png")
    plot_cv(cv_summary, args.output_dir / "cv_f1_recall_auc.png")
    plot_importance(importance_rows, args.output_dir / "feature_importance_extra_trees.png")

    summary = {
        "task": "vbn_orion_literature_guided_proxy_baselines",
        "label_definition": "low risk proxy: 05/10 cNm; high risk proxy: 40/50/60 cNm; 20/30 cNm excluded",
        "feature_count": len(feature_names),
        "sample_counts": sample_counts,
        "heldout": heldout_rows,
        "cv_summary": cv_summary,
        "tiny_1d_cnn": None
        if cnn_result is None
        else {
            "metrics": cnn_result.metrics,
            "param_count": cnn_result.param_count,
            "state_dict_mb": cnn_result.state_dict_mb,
            "int8_weight_est_mb": cnn_result.int8_weight_est_mb,
            "inference_ms_per_sample": cnn_result.inference_ms_per_sample,
        },
        "top_extra_trees_features": importance_rows[:20],
        "limitations": [
            "ORION AE remains a proxy dataset, not real MPU6050 building vibration validation.",
            "Binary labels are torque-level proxies, not certified structural risk levels.",
            "Small sample count makes repeated CV useful but still not a substitute for field validation.",
        ],
    }
    (args.output_dir / "vbn_literature_guided_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_report(args.output_dir, len(feature_names), sample_counts, heldout_rows, cv_summary, cnn_result, importance_rows, args)
    print(f"Wrote {args.output_dir / 'VBN_Literature_Guided_Baseline_Report.md'}")


if __name__ == "__main__":
    main()
