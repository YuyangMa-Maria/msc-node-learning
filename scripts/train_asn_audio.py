"""Train the initial acoustic risk-classification baselines.

Waveforms are converted to log-Mel features inside the model so preprocessing
remains part of the reproducible contract. The resulting score is an airborne
acoustic abnormality proxy, not laboratory acoustic-emission quantification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torchaudio
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AudioSample:
    path: Path
    label: int
    source: str
    split: str


class AsnAudioDataset(Dataset):
    def __init__(
        self,
        samples: list[AudioSample],
        target_samples: int,
        train: bool,
        seed: int,
    ) -> None:
        self.samples = samples
        self.target_samples = target_samples
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def _fix_length(self, waveform: torch.Tensor, index: int) -> torch.Tensor:
        waveform = waveform.mean(dim=0)
        n = waveform.numel()
        if n == self.target_samples:
            return waveform
        if n > self.target_samples:
            if self.train:
                rng = random.Random(self.seed + index)
                start = rng.randint(0, n - self.target_samples)
            else:
                start = (n - self.target_samples) // 2
            return waveform[start : start + self.target_samples]
        pad = self.target_samples - n
        return torch.nn.functional.pad(waveform, (0, pad))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        waveform, _ = torchaudio.load(sample.path)
        waveform = self._fix_length(waveform, index)
        if self.train:
            rng = random.Random(self.seed + index)
            gain = rng.uniform(0.85, 1.15)
            waveform = waveform * gain
            if rng.random() < 0.35:
                waveform = waveform + torch.randn_like(waveform) * rng.uniform(0.0005, 0.002)
        return waveform.float(), torch.tensor(sample.label, dtype=torch.float32)


class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, n_mels: int) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            normalized=False,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.mel(waveform)
        spec = self.amplitude_to_db(spec.clamp_min(1e-10))
        mean = spec.mean(dim=(-2, -1), keepdim=True)
        std = spec.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return ((spec - mean) / std).unsqueeze(1)


class TinyAudioCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.15), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.net(x)).flatten()


class AsnModel(nn.Module):
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, n_mels: int) -> None:
        super().__init__()
        self.frontend = LogMelFrontend(sample_rate, n_fft, hop_length, n_mels)
        self.encoder = TinyAudioCnn()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.frontend(waveform))


def read_samples(index_path: Path, split: str, sources: set[str] | None) -> list[AudioSample]:
    rows: list[AudioSample] = []
    with index_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            if sources and row["source"] not in sources:
                continue
            rows.append(
                AudioSample(
                    path=PROJECT_ROOT / row["path"],
                    label=int(row["label"]),
                    source=row["source"],
                    split=row["split"],
                )
            )
    return rows


def limit_samples(samples: list[AudioSample], limit: int | None, seed: int) -> list[AudioSample]:
    if limit is None or limit <= 0 or len(samples) <= limit:
        return samples
    by_label: dict[int, list[AudioSample]] = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)
    rng = random.Random(seed)
    for group in by_label.values():
        rng.shuffle(group)
    selected: list[AudioSample] = []
    for label in (0, 1):
        quota = max(1, round(limit * len(by_label[label]) / max(len(samples), 1)))
        selected.extend(by_label[label][:quota])
    rng.shuffle(selected)
    return selected[:limit]


def best_f1_threshold(probs: torch.Tensor, labels: torch.Tensor) -> float:
    y = labels.detach().cpu().int()
    candidates = torch.unique(probs.detach().cpu()).tolist()
    if not candidates:
        return 0.5
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        pred = (probs >= threshold).int()
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu()
    y = labels.detach().cpu().int()
    pred = (probs >= threshold).int()
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    try:
        roc_auc = float(roc_auc_score(y.numpy(), probs.numpy()))
    except ValueError:
        roc_auc = float("nan")
    return {
        "accuracy": accuracy,
        "precision_positive": precision,
        "recall_positive": recall,
        "specificity": specificity,
        "f1_positive": f1,
        "roc_auc": roc_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    tune_threshold: bool = False,
    threshold: float = 0.5,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for waveform, labels in loader:
        waveform = waveform.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(waveform)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * waveform.size(0)
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    probs = torch.sigmoid(logits_cat).detach().cpu()
    selected_threshold = best_f1_threshold(probs, labels_cat) if tune_threshold else threshold
    metrics = binary_metrics(logits_cat, labels_cat, selected_threshold)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def build_loss(train_samples: list[AudioSample], device: torch.device, pos_weight: str) -> nn.Module:
    if pos_weight == "none":
        return nn.BCEWithLogitsLoss()
    if pos_weight == "auto":
        positives = sum(sample.label == 1 for sample in train_samples)
        negatives = sum(sample.label == 0 for sample in train_samples)
        weight = negatives / max(positives, 1)
    else:
        weight = float(pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32, device=device))


def benchmark_inference(model: nn.Module, loader: DataLoader, device: torch.device, batches: int = 20) -> float:
    model.eval()
    times: list[float] = []
    with torch.no_grad():
        for idx, (waveform, _) in enumerate(loader):
            if idx >= batches:
                break
            waveform = waveform.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(waveform)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed / waveform.size(0) * 1000.0)
    return float(sum(times) / max(len(times), 1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASN binary acoustic risk model.")
    parser.add_argument("--index", type=Path, default=PROJECT_ROOT / "experiments" / "asn_audio" / "asn_audio_index.csv")
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pos-weight", default="auto")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_audio")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    sources = set(args.sources) if args.sources else None
    target_samples = int(args.sample_rate * args.duration)
    train_samples = limit_samples(read_samples(args.index, "train", sources), args.max_train, args.seed)
    val_samples = limit_samples(read_samples(args.index, "val", sources), args.max_val, args.seed)
    test_samples = limit_samples(read_samples(args.index, "test", sources), args.max_test, args.seed)
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("Empty train/val/test split. Run scripts/build_asn_index.py first.")

    device = torch.device(args.device)
    train_loader = DataLoader(
        AsnAudioDataset(train_samples, target_samples, train=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        AsnAudioDataset(val_samples, target_samples, train=False, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        AsnAudioDataset(test_samples, target_samples, train=False, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = AsnModel(args.sample_rate, args.n_fft, args.hop_length, args.n_mels).to(device)
    criterion = build_loss(train_samples, device, args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_id = "tiny_logmel_cnn"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    history: list[dict[str, object]] = []
    best_val_f1 = -math.inf

    print(f"Device: {device}", flush=True)
    print("Model: tiny_logmel_cnn", flush=True)
    print(f"Parameters: {count_parameters(model)}", flush=True)
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, None, tune_threshold=True)
        row: dict[str, object] = {"epoch": epoch, "model": run_id, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1_positive']:.4f} val_recall={val_metrics['recall_positive']:.4f} "
            f"threshold={val_metrics['threshold']:.4f}",
            flush=True,
        )
        if val_metrics["f1_positive"] > best_val_f1:
            best_val_f1 = val_metrics["f1_positive"]
            torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": val_metrics}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    decision_threshold = float(checkpoint["val_metrics"].get("threshold", 0.5))
    test_metrics = run_epoch(model, test_loader, criterion, device, None, threshold=decision_threshold)
    inference_ms = benchmark_inference(model, test_loader, device)
    result = {
        "model": run_id,
        "parameters": count_parameters(model),
        "average_inference_ms": inference_ms,
        "best_val_f1": best_val_f1,
        "decision_threshold": decision_threshold,
        "test": test_metrics,
        "sources": sorted(sources) if sources else "all",
        "training": {
            "index": str(args.index),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sample_rate": args.sample_rate,
            "duration": args.duration,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "n_mels": args.n_mels,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pos_weight": args.pos_weight,
            "max_train": args.max_train,
            "max_val": args.max_val,
            "max_test": args.max_test,
            "sample_counts": {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
            "device": str(device),
            "seed": args.seed,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_jsonl(output_dir / "history.jsonl", history)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
