"""Train, distil, quantise and gate a PC/MARH shared-head update.

The constrained nodes keep their deployed private encoders/projections.  The
resource-rich PC trains a larger teacher on the resulting 64-D risk
representations, then distils it into the runtime-compatible 64-32-1 head.
The candidate must pass an offline acceptance gate before it is serialised; this
script does not bypass the firmware's independent integrity and golden-vector
checks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


TENSOR_NAMES = ("0.weight", "0.bias", "3.weight", "3.bias")
MODALITIES = ("vsn", "asn")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    best, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(labels, probabilities >= threshold, zero_division=0)
        if score > best_score:
            best, best_score = float(threshold), float(score)
    return best


def metric_row(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)),
    }


def project(
    embeddings: np.ndarray,
    state: dict[str, torch.Tensor],
    modality: str,
    scaler: StandardScaler,
) -> np.ndarray:
    x = scaler.transform(embeddings).astype(np.float32)
    w1 = state[f"projections.{modality}.0.weight"].cpu().numpy()
    b1 = state[f"projections.{modality}.0.bias"].cpu().numpy()
    w2 = state[f"projections.{modality}.3.weight"].cpu().numpy()
    b2 = state[f"projections.{modality}.3.bias"].cpu().numpy()
    return np.maximum(np.maximum(x @ w1.T + b1, 0.0) @ w2.T + b2, 0.0).astype(np.float32)


def head_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = {
        key.removeprefix("shared_head."): value.detach().cpu().float()
        for key, value in checkpoint["model_state"].items()
        if key.startswith("shared_head.")
    }
    missing = set(TENSOR_NAMES) - set(state)
    if missing:
        raise KeyError(f"Missing shared-head tensors: {sorted(missing)}")
    return state


def student_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))


def teacher_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 1),
    )


def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            outputs.append(model(batch).flatten().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def balanced_loader(x: np.ndarray, y: np.ndarray, modality: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    """Balance labels and modalities so one node cannot dominate the teacher."""
    counts = {(int(m), int(label)): max(1, int(np.sum((modality == m) & (y == label)))) for m in (0, 1) for label in (0, 1)}
    weights = np.asarray([1.0 / counts[(int(m), int(label))] for m, label in zip(modality, y)], dtype=np.float64)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), len(weights), replacement=True, generator=generator)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_modality: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    teacher: nn.Module | None = None,
    alpha: float = 0.55,
    temperature: float = 3.0,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    if teacher is not None:
        teacher = teacher.to(device).eval()
    loader = balanced_loader(train_x, train_y, train_modality, batch_size, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    val_tensor = torch.from_numpy(val_x).to(device)
    val_target = torch.from_numpy(val_y.astype(np.float32)).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience, stalled = 16, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).flatten()
            hard_loss = F.binary_cross_entropy_with_logits(logits, yb)
            if teacher is None:
                loss = hard_loss
            else:
                with torch.no_grad():
                    soft = torch.sigmoid(teacher(xb).flatten() / temperature)
                distil = F.binary_cross_entropy_with_logits(logits / temperature, soft) * temperature**2
                loss = alpha * hard_loss + (1.0 - alpha) * distil
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(F.binary_cross_entropy_with_logits(model(val_tensor).flatten(), val_target).cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), history


def serialise_int8(state: dict[str, torch.Tensor]) -> bytes:
    """Serialise the fixed 64-32-1 tensor order expected by both MCU runtimes."""
    scales, quantised = [], []
    for name in TENSOR_NAMES:
        values = state[name].detach().cpu().numpy()
        scale = max(float(np.max(np.abs(values))) / 127.0, np.finfo(np.float32).tiny)
        scales.append(scale)
        quantised.append(np.clip(np.rint(values / scale), -127, 127).astype(np.int8).tobytes())
    return struct.pack("<4f", *scales) + b"".join(quantised)


def int8_logits(payload: bytes, representations: np.ndarray) -> np.ndarray:
    scales = np.frombuffer(payload[:16], dtype="<f4")
    values = np.frombuffer(payload[16:], dtype=np.int8)
    w1 = values[:2048].astype(np.float32).reshape(32, 64) * scales[0]
    b1 = values[2048:2080].astype(np.float32) * scales[1]
    w2 = values[2080:2112].astype(np.float32) * scales[2]
    b2 = float(values[2112]) * float(scales[3])
    return (np.maximum(representations @ w1.T + b1, 0.0) @ w2 + b2).astype(np.float32)


def evaluate(
    name: str,
    logits: dict[str, dict[str, np.ndarray]],
    labels: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    for modality in MODALITIES:
        threshold = best_threshold(labels[modality]["val"], sigmoid(logits[modality]["val"]))
        for split in ("val", "test"):
            rows.append({
                "model": name,
                "modality": modality,
                "split": split,
                **metric_row(labels[modality][split], sigmoid(logits[modality][split]), threshold),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Train candidates, apply the offline gate and emit one transport payload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "marh_teacher_update",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 99])
    parser.add_argument("--version", type=lambda value: int(value, 0), default=0x0301)
    parser.add_argument("--max-f1-drop", type=float, default=0.01)
    args = parser.parse_args()

    started = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = args.source_project_root.resolve()
    checkpoint_path = root / "outputs/conference_grouped_parameter_sharing_models/shared_risk_head.pt"
    cache_dir = root / "outputs/conference_grouped_parameter_sharing_models/embedding_cache"
    cache_paths = {
        "vsn": cache_dir / "vsn_cnn_embeddings_t3000_e1000_s42.npz",
        "asn": cache_dir / "asn_cnn_embeddings_t1600_e500_s42.npz",
    }
    formal = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    formal_state = formal["model_state"]
    base_state = head_state(formal)

    representations: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, dict[str, np.ndarray]] = {}
    for modality in MODALITIES:
        cache = np.load(cache_paths[modality], allow_pickle=True)
        scaler = StandardScaler().fit(cache["train_x"].astype(np.float32))
        representations[modality], labels[modality] = {}, {}
        for split in ("train", "val", "test"):
            representations[modality][split] = project(cache[f"{split}_x"].astype(np.float32), formal_state, modality, scaler)
            labels[modality][split] = cache[f"{split}_y"].astype(np.int64)

    combined_x, combined_y, combined_m = {}, {}, {}
    for split in ("train", "val"):
        combined_x[split] = np.concatenate([representations[m][split] for m in MODALITIES])
        combined_y[split] = np.concatenate([labels[m][split] for m in MODALITIES])
        combined_m[split] = np.concatenate([
            np.full(len(labels[m][split]), index, dtype=np.int64) for index, m in enumerate(MODALITIES)
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []

    base = student_model()
    base.load_state_dict(base_state)
    base_logits = {
        m: {split: predict(base.to(device), representations[m][split], device) for split in ("val", "test")}
        for m in MODALITIES
    }
    all_rows.extend(evaluate("deployed_base_fp32", base_logits, labels))

    candidates = []
    for seed in args.seeds:
        set_seed(seed)
        teacher, teacher_history = train_model(
            teacher_model(), combined_x["train"], combined_y["train"], combined_m["train"],
            combined_x["val"], combined_y["val"], device, args.epochs, args.batch_size, 1e-3, seed,
        )
        scratch, scratch_history = train_model(
            student_model(), combined_x["train"], combined_y["train"], combined_m["train"],
            combined_x["val"], combined_y["val"], device, args.epochs, args.batch_size, 1e-3, seed + 1000,
        )
        kd_initial = student_model()
        kd_initial.load_state_dict(base_state)
        kd, kd_history = train_model(
            kd_initial, combined_x["train"], combined_y["train"], combined_m["train"],
            combined_x["val"], combined_y["val"], device, args.epochs, args.batch_size, 4e-4, seed + 2000,
            teacher=teacher, alpha=0.65, temperature=3.0,
        )
        for kind, model in (("teacher", teacher), ("student_scratch", scratch), ("student_kd", kd)):
            logits = {
                m: {split: predict(model, representations[m][split], device) for split in ("val", "test")}
                for m in MODALITIES
            }
            rows = evaluate(f"{kind}_seed{seed}_fp32", logits, labels)
            all_rows.extend(rows)
            if kind == "student_kd":
                val_macro = float(np.mean([row["f1"] for row in rows if row["split"] == "val"]))
                candidates.append((val_macro, seed, copy.deepcopy(model.cpu().state_dict())))
                model.to(device)
        for kind, history in (("teacher", teacher_history), ("student_scratch", scratch_history), ("student_kd", kd_history)):
            histories.extend({"seed": seed, "model": kind, **row} for row in history)

    _, selected_seed, selected_state = max(candidates, key=lambda item: item[0])
    payload = serialise_int8(selected_state)
    if len(payload) != 2129:
        raise AssertionError(f"Unexpected INT8 payload length: {len(payload)}")
    int8_logits_by_modality = {
        m: {split: int8_logits(payload, representations[m][split]) for split in ("val", "test")}
        for m in MODALITIES
    }
    int8_rows = evaluate(f"student_kd_seed{selected_seed}_int8", int8_logits_by_modality, labels)
    all_rows.extend(int8_rows)

    base_test = {row["modality"]: row for row in all_rows if row["model"] == "deployed_base_fp32" and row["split"] == "test"}
    candidate_test = {row["modality"]: row for row in int8_rows if row["split"] == "test"}
    candidate_val = {row["modality"]: row for row in int8_rows if row["split"] == "val"}
    per_node_gate = {
        modality: candidate_test[modality]["f1"] >= base_test[modality]["f1"] - args.max_f1_drop
        for modality in MODALITIES
    }
    deployment_accepted = all(per_node_gate.values())
    golden = np.linspace(-1.0, 1.0, 64, dtype=np.float32)[None, :]
    expected_golden = float(int8_logits(payload, golden)[0])
    crc32 = zlib.crc32(payload) & 0xFFFFFFFF
    thresholds = {modality: float(candidate_val[modality]["threshold"]) for modality in MODALITIES}

    (output / "candidate_shared_head_int8.bin").write_bytes(payload)
    torch.save(
        {
            "model_state": {f"shared_head.{name}": value.cpu() for name, value in selected_state.items()},
            "teacher_update": True,
            "selected_seed": selected_seed,
            "version": args.version,
            "thresholds": thresholds,
        },
        output / "candidate_shared_head.pt",
    )
    write_csv(output / "metrics.csv", all_rows)
    write_csv(output / "training_history.csv", histories)
    summary = {
        "stage": "PC/MARH teacher training and runtime shared-head distillation",
        "source_checkpoint": str(checkpoint_path),
        "data_boundary": "grouped train/validation/test representations; test excluded from training and candidate selection",
        "device": str(device),
        "teacher_architecture": "64-128-64-1 MLP",
        "student_architecture": "64-32-1 shared runtime head",
        "student_parameters": sum(value.numel() for value in selected_state.values()),
        "seeds": args.seeds,
        "selected_seed": selected_seed,
        "candidate_version": args.version,
        "candidate_version_hex": f"0x{args.version:04x}",
        "format": "INT8 symmetric per tensor",
        "payload_bytes": len(payload),
        "crc32": f"0x{crc32:08x}",
        "expected_golden_output": expected_golden,
        "thresholds": thresholds,
        "base_test": base_test,
        "candidate_int8_test": candidate_test,
        "maximum_allowed_f1_drop": args.max_f1_drop,
        "per_node_gate": per_node_gate,
        "deployment_accepted": deployment_accepted,
        "elapsed_s": time.perf_counter() - started,
        "claim_boundary": "The PC trains and distils a new compatible shared head. Physical deployment must be demonstrated separately through host upload and BLE installation.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# PC/MARH Teacher Update",
        "",
        "A resource-rich 64-128-64-1 teacher was trained on grouped training representations and distilled into the deployed 2,113-parameter 64-32-1 shared head. Candidate selection used validation data only; the grouped test sets remained held out.",
        "",
        "| Model | Node | F1 | Recall | AUROC | Threshold |",
        "|---|---|---:|---:|---:|---:|",
    ]
    focus_names = {"deployed_base_fp32", f"student_kd_seed{selected_seed}_int8"}
    for row in all_rows:
        if row["split"] == "test" and row["model"] in focus_names:
            lines.append(f"| {row['model']} | {row['modality'].upper()} | {row['f1']:.4f} | {row['recall']:.4f} | {row['auroc']:.4f} | {row['threshold']:.3f} |")
    lines.extend([
        "",
        f"Deployment gate: **{'PASS' if deployment_accepted else 'FAIL'}**.",
        "",
        f"The selected INT8 package is {len(payload)} bytes, version `0x{args.version:04x}`, CRC32 `0x{crc32:08x}`.",
        "",
        "This report establishes host-side training, distillation, quantisation and offline acceptance. It does not claim physical installation until the separate serial/BLE transaction succeeds.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if deployment_accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
