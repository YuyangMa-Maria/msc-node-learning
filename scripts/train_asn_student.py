"""Train compact ASN students from scratch or by knowledge distillation.

The student may use a shorter audio window than its teacher. Both crops come
from the same recording, which lets the experiment test whether long-context
teacher behaviour can be transferred to a lower-latency TinyML model.
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
from types import SimpleNamespace
from typing import Iterable

import torch
import torchaudio
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_asn_audio import PROJECT_ROOT, AudioSample, LogMelFrontend, AsnModel, build_loss, read_samples


@dataclass
class StudentBatch:
    """Logical contents of a student/teacher training item."""

    student_waveform: torch.Tensor
    teacher_waveform: torch.Tensor
    label: torch.Tensor


class AsnStudentDataset(Dataset):
    """Return paired short and long views of the same labelled recording."""

    def __init__(
        self,
        samples: list[AudioSample],
        sample_rate: int,
        student_duration: float,
        teacher_duration: float,
        train: bool,
        seed: int,
    ) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.student_samples = int(sample_rate * student_duration)
        self.teacher_samples = int(sample_rate * teacher_duration)
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_or_pad(self, waveform: torch.Tensor, target_samples: int, index: int) -> torch.Tensor:
        """Use repeatable random training crops and centred evaluation crops."""
        waveform = waveform.mean(dim=0)
        n = waveform.numel()
        if n == target_samples:
            return waveform
        if n > target_samples:
            if self.train:
                rng = random.Random(self.seed + index)
                start = rng.randint(0, n - target_samples)
            else:
                start = (n - target_samples) // 2
            return waveform[start : start + target_samples]
        return torch.nn.functional.pad(waveform, (0, target_samples - n))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        waveform, sr = torchaudio.load(sample.path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        student_waveform = self._crop_or_pad(waveform, self.student_samples, index)
        teacher_waveform = self._crop_or_pad(waveform, self.teacher_samples, index)
        if self.train:
            rng = random.Random(self.seed + index)
            gain = rng.uniform(0.85, 1.15)
            student_waveform = student_waveform * gain
            teacher_waveform = teacher_waveform * gain
            if rng.random() < 0.35:
                noise_scale = rng.uniform(0.0005, 0.002)
                student_waveform = student_waveform + torch.randn_like(student_waveform) * noise_scale
                teacher_waveform = teacher_waveform + torch.randn_like(teacher_waveform) * noise_scale
        return student_waveform.float(), teacher_waveform.float(), torch.tensor(sample.label, dtype=torch.float32)


class StudentAudioCnn(nn.Module):
    """Small log-Mel CNN whose width multiplier controls the MCU cost."""

    def __init__(self, width: float, dropout: float) -> None:
        super().__init__()
        channels = [max(4, int(round(c * width))) for c in [16, 32, 64, 96]]
        c1, c2, c3, c4 = channels
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.net(x)).flatten()


class AsnStudentModel(nn.Module):
    """End-to-end waveform model combining the fixed frontend and tiny CNN."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        width: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.frontend = LogMelFrontend(sample_rate, n_fft, hop_length, n_mels)
        self.encoder = StudentAudioCnn(width, dropout)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.frontend(waveform))


def as_namespace(data: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**data)


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
    probs_cpu = probs.detach().cpu()
    y = labels.detach().cpu().int()
    candidates = torch.unique(probs_cpu).tolist()
    if not candidates:
        return 0.5
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        pred = (probs_cpu >= threshold).int()
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


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float | int]:
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


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_loader(args: argparse.Namespace, samples: list[AudioSample], train: bool) -> DataLoader:
    return DataLoader(
        AsnStudentDataset(
            samples,
            sample_rate=args.sample_rate,
            student_duration=args.duration,
            teacher_duration=args.teacher_duration,
            train=train,
            seed=args.seed,
        ),
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and str(args.device).startswith("cuda"),
    )


def load_teacher(run_dir: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model_args = as_namespace(checkpoint["args"])  # type: ignore[arg-type]
    teacher = AsnModel(
        sample_rate=int(getattr(model_args, "sample_rate")),
        n_fft=int(getattr(model_args, "n_fft")),
        hop_length=int(getattr(model_args, "hop_length")),
        n_mels=int(getattr(model_args, "n_mels")),
    ).to(device)
    teacher.load_state_dict(checkpoint["model"])  # type: ignore[arg-type]
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    hard_criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    teacher: nn.Module | None,
    kd_alpha: float,
    temperature: float,
    tune_threshold: bool = False,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Run one train/evaluation epoch, optionally with a frozen teacher."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_hard = 0.0
    total_kd = 0.0
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for student_waveform, teacher_waveform, labels in loader:
        student_waveform = student_waveform.to(device, non_blocking=True)
        teacher_waveform = teacher_waveform.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(student_waveform)
        hard_loss = hard_criterion(logits, labels)
        kd_loss = torch.tensor(0.0, device=device)
        if teacher is not None:
            # The teacher receives its longer view; no gradient or optimiser
            # state is retained for this resource-rich reference model.
            with torch.no_grad():
                teacher_logits = teacher(teacher_waveform)
                soft_targets = torch.sigmoid(teacher_logits / temperature)
            kd_loss = nn.functional.binary_cross_entropy_with_logits(logits / temperature, soft_targets) * (temperature**2)
            loss = (1.0 - kd_alpha) * hard_loss + kd_alpha * kd_loss
        else:
            loss = hard_loss
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_hard += float(hard_loss.item()) * labels.size(0)
        total_kd += float(kd_loss.item()) * labels.size(0)
        logits_all.append(logits.detach())
        labels_all.append(labels.detach())
    logits_cat = torch.cat(logits_all)
    labels_cat = torch.cat(labels_all)
    threshold_selected = best_f1_threshold(torch.sigmoid(logits_cat), labels_cat) if tune_threshold else threshold
    metrics = binary_metrics(logits_cat, labels_cat, threshold_selected)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    metrics["hard_loss"] = total_hard / max(len(loader.dataset), 1)
    metrics["kd_loss"] = total_kd / max(len(loader.dataset), 1)
    return metrics


def benchmark(model: nn.Module, loader: DataLoader, device: torch.device, batches: int = 20) -> float:
    model.eval()
    times: list[float] = []
    with torch.no_grad():
        for idx, (student_waveform, _, _) in enumerate(loader):
            if idx >= batches:
                break
            student_waveform = student_waveform.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(student_waveform)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed / student_waveform.size(0) * 1000.0)
    return float(sum(times) / max(len(times), 1))


def write_summary(output_dir: Path, result: dict[str, object]) -> None:
    lines = [
        "# ASN Student Training Result",
        "",
        f"- Run: `{result['run_id']}`",
        f"- Training mode: {result['training_mode']}",
        f"- Duration: {result['duration']} s",
        f"- Width: {result['width']}",
        f"- Parameters: {result['parameters']} ({result['parameters_m']:.4f}M)",
        f"- Estimated FP32 weights: {result['fp32_weight_mb']:.4f} MB",
        f"- Estimated INT8 weights: {result['int8_weight_mb']:.4f} MB",
        f"- Best validation F1: {result['best_val_f1']:.4f}",
        f"- Decision threshold: {result['decision_threshold']:.4f}",
        f"- Test F1: {result['test']['f1_positive']:.4f}",
        f"- Test recall: {result['test']['recall_positive']:.4f}",
        f"- Test accuracy: {result['test']['accuracy']:.4f}",
        f"- Test ROC-AUC: {result['test']['roc_auc']:.4f}",
        f"- Average inference: {result['average_inference_ms']:.4f} ms/audio on {result['device']}",
        "",
        "## Caution",
        "",
        "- This is an acoustic abnormality proxy experiment using the available multimodal concrete crack audio data.",
        "- The memory values are parameter-size estimates, not MCU tensor-arena measurements.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Train one scratch or KD configuration selected by command-line options."""
    parser = argparse.ArgumentParser(description="Train ASN student model from scratch or with KD.")
    parser.add_argument("--index", type=Path, default=PROJECT_ROOT / "experiments" / "asn_audio" / "asn_audio_index.csv")
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--teacher-run-dir", type=Path, default=None)
    parser.add_argument("--teacher-duration", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", default="auto")
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_student")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    sources = set(args.sources) if args.sources else None
    train_samples = limit_samples(read_samples(args.index, "train", sources), args.max_train, args.seed)
    val_samples = limit_samples(read_samples(args.index, "val", sources), args.max_val, args.seed)
    test_samples = limit_samples(read_samples(args.index, "test", sources), args.max_test, args.seed)
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("Empty train/val/test split. Run scripts/build_asn_index.py first.")

    device = torch.device(args.device)
    model = AsnStudentModel(
        args.sample_rate,
        args.n_fft,
        args.hop_length,
        args.n_mels,
        args.width,
        args.dropout,
    ).to(device)
    teacher = load_teacher(args.teacher_run_dir, device) if args.teacher_run_dir is not None else None
    training_mode = "kd" if teacher is not None else "scratch"
    duration_tag = str(args.duration).replace(".", "p")
    width_tag = str(args.width).replace(".", "p")
    run_id = f"asn_student_w{width_tag}_d{duration_tag}s_{training_mode}"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(args, train_samples, train=True)
    val_loader = make_loader(args, val_samples, train=False)
    test_loader = make_loader(args, test_samples, train=False)
    criterion = build_loss(train_samples, device, args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_path = output_dir / "best.pt"
    history: list[dict[str, object]] = []
    best_val_f1 = -math.inf
    print(f"Device: {device}", flush=True)
    print(f"Run: {run_id}", flush=True)
    print(f"Training mode: {training_mode}", flush=True)
    print(f"Parameters: {count_parameters(model)}", flush=True)
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            teacher,
            args.kd_alpha,
            args.temperature,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            None,
            teacher=None,
            kd_alpha=0.0,
            temperature=args.temperature,
            tune_threshold=True,
        )
        row: dict[str, object] = {"epoch": epoch, "run_id": run_id, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1_positive']:.4f} val_recall={val_metrics['recall_positive']:.4f} "
            f"threshold={val_metrics['threshold']:.4f}",
            flush=True,
        )
        if float(val_metrics["f1_positive"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1_positive"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "run_id": run_id,
                    "training_mode": training_mode,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    decision_threshold = float(checkpoint["val_metrics"].get("threshold", 0.5))
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        None,
        teacher=None,
        kd_alpha=0.0,
        temperature=args.temperature,
        threshold=decision_threshold,
    )
    inference_ms = benchmark(model, test_loader, device)
    params = count_parameters(model)
    result = {
        "run_id": run_id,
        "training_mode": training_mode,
        "duration": args.duration,
        "teacher_duration": args.teacher_duration,
        "width": args.width,
        "dropout": args.dropout,
        "parameters": params,
        "parameters_m": params / 1_000_000,
        "fp32_weight_mb": params * 4 / (1024 * 1024),
        "int8_weight_mb": params / (1024 * 1024),
        "average_inference_ms": inference_ms,
        "best_val_f1": best_val_f1,
        "decision_threshold": decision_threshold,
        "test": test_metrics,
        "sources": sorted(sources) if sources else "all",
        "device": str(device),
        "training": {
            "index": str(args.index),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sample_rate": args.sample_rate,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "n_mels": args.n_mels,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pos_weight": args.pos_weight,
            "kd_alpha": args.kd_alpha if teacher is not None else None,
            "temperature": args.temperature if teacher is not None else None,
            "teacher_run_dir": str(args.teacher_run_dir) if args.teacher_run_dir else None,
            "sample_counts": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
            "seed": args.seed,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_jsonl(output_dir / "history.jsonl", history)
    write_summary(output_dir, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
