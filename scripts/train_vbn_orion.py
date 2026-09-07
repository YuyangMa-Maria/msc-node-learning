"""Train the initial temporal model on the ORION structural proxy.

Bolt-torque classes provide a controlled structural time-series task for the
node interface. They do not validate accelerometer-based building vibration,
and that boundary is retained in every generated report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "experiments" / "vbn_orion" / "vbn_orion_index.csv"


def read_rows(index_path: Path, split: str | None = None, binary_only: bool = True) -> list[dict[str, str]]:
    rows = []
    with index_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split and row["split"] != split:
                continue
            if binary_only and row["use_binary"] != "1":
                continue
            rows.append(row)
    return rows


def load_signal(path: Path, max_points: int) -> dict[str, np.ndarray]:
    data = loadmat(path)
    channels = {}
    for channel in ["A", "B", "C", "D"]:
        x = np.asarray(data[channel]).reshape(-1).astype(np.float32)
        if x.size > max_points:
            step = max(1, x.size // max_points)
            x = x[::step][:max_points]
        channels[channel] = x
    return channels


def band_energy_features(x: np.ndarray, prefix: str, sample_rate: float) -> dict[str, float]:
    x = x - float(np.mean(x))
    spectrum = np.abs(np.fft.rfft(x))
    power = spectrum * spectrum
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    total = float(np.sum(power)) + 1e-12
    bands = [(0, 50_000), (50_000, 150_000), (150_000, 400_000), (400_000, 1_000_000), (1_000_000, sample_rate / 2)]
    feats = {}
    for idx, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        feats[f"{prefix}_band_{idx}_energy_ratio"] = float(np.sum(power[mask]) / total)
    centroid = float(np.sum(freqs * power) / total)
    feats[f"{prefix}_spectral_centroid"] = centroid
    return feats


def channel_features(x: np.ndarray, prefix: str, sample_rate: float) -> dict[str, float]:
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    peak = float(np.max(abs_x))
    segments = np.array_split(x, 8)
    segment_rms = np.array([np.sqrt(np.mean(seg * seg)) for seg in segments], dtype=np.float64)
    feats = {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_rms": rms,
        f"{prefix}_abs_mean": float(np.mean(abs_x)),
        f"{prefix}_peak": peak,
        f"{prefix}_crest": peak / rms,
        f"{prefix}_kurtosis": float(kurtosis(x, fisher=True, bias=False)),
        f"{prefix}_skew": float(skew(x, bias=False)),
        f"{prefix}_p95_abs": float(np.percentile(abs_x, 95)),
        f"{prefix}_p99_abs": float(np.percentile(abs_x, 99)),
        f"{prefix}_zero_crossing_rate": float(np.mean(np.diff(np.signbit(x)) != 0)),
        f"{prefix}_segment_rms_mean": float(np.mean(segment_rms)),
        f"{prefix}_segment_rms_std": float(np.std(segment_rms)),
        f"{prefix}_segment_rms_max": float(np.max(segment_rms)),
    }
    feats.update(band_energy_features(x, prefix, sample_rate))
    return feats


def extract_features(path: Path, max_points: int, sample_rate: float) -> dict[str, float]:
    channels = load_signal(path, max_points)
    feats: dict[str, float] = {}
    rms_values = []
    peak_values = []
    for channel, values in channels.items():
        cfeats = channel_features(values, channel, sample_rate)
        feats.update(cfeats)
        rms_values.append(cfeats[f"{channel}_rms"])
        peak_values.append(cfeats[f"{channel}_peak"])
    feats["cross_channel_rms_mean"] = float(np.mean(rms_values))
    feats["cross_channel_rms_std"] = float(np.std(rms_values))
    feats["cross_channel_peak_mean"] = float(np.mean(peak_values))
    feats["cross_channel_peak_std"] = float(np.std(peak_values))
    return feats


def build_feature_table(rows: list[dict[str, str]], cache_path: Path, max_points: int, sample_rate: float) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        feature_names = cached["feature_names"]
        X = np.asarray(cached["X"], dtype=np.float32)
        y = np.asarray(cached["y"], dtype=np.int64)
        cached_rows = cached["rows"]
        return X, y, feature_names, cached_rows

    feature_rows = []
    labels = []
    for idx, row in enumerate(rows, 1):
        path = PROJECT_ROOT / row["path"]
        feats = extract_features(path, max_points=max_points, sample_rate=sample_rate)
        feature_rows.append(feats)
        labels.append(int(row["label"]))
        print(f"features {idx}/{len(rows)} {path.name}", flush=True)
    feature_names = sorted(feature_rows[0])
    X = np.asarray([[float(row[name]) for name in feature_names] for row in feature_rows], dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "X": X.tolist(),
                "y": y.tolist(),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return X, y, feature_names, rows


def clf_metrics(model, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    probs = model.predict_proba(X)[:, 1]
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
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_threshold(model, X: np.ndarray, y: np.ndarray) -> float:
    probs = model.predict_proba(X)[:, 1]
    best = (0.5, -1.0)
    for threshold in sorted(set(float(p) for p in probs)):
        f1 = clf_metrics(model, X, y, threshold)["f1_positive"]
        if f1 > best[1]:
            best = (threshold, f1)
    return best[0]


def plot_confusion(metrics: dict[str, float], path: Path, title: str) -> None:
    matrix = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=140)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1], labels=["Pred low", "Pred high"])
    ax.set_yticks([0, 1], labels=["True low", "True high"])
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VBN Orion AE proxy baseline.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vbn_orion")
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--sample-rate", type=float, default=5_000_000.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = read_rows(args.index, binary_only=True)
    cache_path = args.output_dir / f"features_max{args.max_points}.json"
    X_all, y_all, feature_names, cached_rows = build_feature_table(all_rows, cache_path, args.max_points, args.sample_rate)
    split_indices = {
        split: [idx for idx, row in enumerate(cached_rows) if row["split"] == split]
        for split in ["train", "val", "test"]
    }
    X_train, y_train = X_all[split_indices["train"]], y_all[split_indices["train"]]
    X_val, y_val = X_all[split_indices["val"]], y_all[split_indices["val"]]
    X_test, y_test = X_all[split_indices["test"]], y_all[split_indices["test"]]

    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=4, class_weight="balanced", random_state=42),
    }
    model_results = {}
    best_name = None
    best_val_f1 = -1.0
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        threshold = best_threshold(model, X_val, y_val)
        val_metrics = clf_metrics(model, X_val, y_val, threshold)
        test_metrics = clf_metrics(model, X_test, y_test, threshold)
        model_results[name] = {"threshold": threshold, "val": val_metrics, "test": test_metrics}
        print(f"{name}: val_f1={val_metrics['f1_positive']:.4f} test_f1={test_metrics['f1_positive']:.4f} threshold={threshold:.4f}")
        if val_metrics["f1_positive"] > best_val_f1:
            best_name = name
            best_val_f1 = val_metrics["f1_positive"]

    assert best_name is not None
    best_model = candidates[best_name]
    model_path = args.output_dir / "best_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"model": best_model, "feature_names": feature_names, "max_points": args.max_points, "sample_rate": args.sample_rate}, f)

    start = time.perf_counter()
    _ = best_model.predict_proba(X_test)
    inference_ms = (time.perf_counter() - start) / max(len(X_test), 1) * 1000.0
    best = model_results[best_name]
    result = {
        "model": best_name,
        "task": "orion_low_vs_high_torque_proxy",
        "label_definition": "low risk: 05/10 cNm, high risk: 40/50/60 cNm; 20/30 cNm excluded from binary training",
        "feature_count": len(feature_names),
        "sample_counts": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "average_inference_ms": inference_ms,
        "models": model_results,
        "selected": best,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "split", "accuracy", "precision_positive", "recall_positive", "f1_positive", "roc_auc", "threshold", "tn", "fp", "fn", "tp"],
        )
        writer.writeheader()
        for model_name, rows in model_results.items():
            for split in ["val", "test"]:
                writer.writerow({"model": model_name, "split": split, **rows[split]})
    plot_confusion(best["test"], args.output_dir / "confusion_best.png", f"VBN Proxy Confusion - {best_name}")
    report = [
        "# VBN Orion AE Proxy Baseline Report",
        "",
        "## Setup",
        "",
        "- Dataset: `orion-ae-sensor-b-subset`",
        "- Signal: 4-channel AE/time-series data from `.mat` files (`A`, `B`, `C`, `D`)",
        "- Proxy task: low torque vs high torque classification",
        "- Low-risk proxy: 05/10 cNm",
        "- High-risk proxy: 40/50/60 cNm",
        "- Moderate 20/30 cNm files are excluded from binary baseline and reserved for future risk-level experiments.",
        f"- Feature count: {len(feature_names)}",
        f"- Samples: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}",
        "",
        "## Results",
        "",
        "| Model | Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Threshold | Confusion matrix (TN/FP/FN/TP) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model_name, rows in model_results.items():
        for split in ["val", "test"]:
            m = rows[split]
            report.append(
                f"| {model_name} | {split} | {m['accuracy']:.4f} | {m['precision_positive']:.4f} | {m['recall_positive']:.4f} | {m['f1_positive']:.4f} | {m['roc_auc']:.4f} | {m['threshold']:.4f} | {m['tn']}/{m['fp']}/{m['fn']}/{m['tp']} |"
            )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Selected model: `{best_name}`. This is a VBN proxy baseline, not a final MPU6050 vibration model. It demonstrates that a third local time-series node can produce a compact risk score from structural-sensing-like signals.",
            "",
            "## Files",
            "",
            "- Feature cache: `features_max200000.json`",
            "- Serialized model: `best_model.pkl`",
            "- Metrics CSV: `summary_metrics.csv`",
            "- Confusion matrix: `confusion_best.png`",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
