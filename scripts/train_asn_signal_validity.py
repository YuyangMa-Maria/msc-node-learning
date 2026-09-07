"""Train the auxiliary ASN signal-validity classifier."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchaudio
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_asn_audio import PROJECT_ROOT, AudioSample, LogMelFrontend, build_loss, read_samples


VALIDITY_NAMES = ("usable", "degraded", "invalid")
CONDITIONS = (
    ("clean", 0),
    ("snr_noise", 1),
    ("snr_noise", 2),
    ("snr_noise", 3),
    ("low_volume", 1),
    ("low_volume", 2),
    ("low_volume", 3),
    ("clipping", 1),
    ("clipping", 2),
    ("clipping", 3),
    ("dropout", 1),
    ("dropout", 2),
    ("dropout", 3),
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fix_length(waveform: torch.Tensor, target: int, train: bool, seed: int) -> torch.Tensor:
    waveform = waveform.mean(dim=0)
    n = waveform.numel()
    if n > target:
        if train:
            start = random.Random(seed).randint(0, n - target)
        else:
            start = (n - target) // 2
        waveform = waveform[start : start + target]
    elif n < target:
        waveform = torch.nn.functional.pad(waveform, (0, target - n))
    return waveform.float()


def add_noise_at_snr(waveform: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
    signal_power = waveform.square().mean().clamp_min(1e-12)
    noise_power = noise.square().mean().clamp_min(1e-12)
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return waveform + noise * torch.sqrt(target_noise_power / noise_power)


def dropout_chunks(waveform: torch.Tensor, ratio: float, chunks: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    result = waveform.clone()
    chunk_length = max(1, int(result.numel() * ratio / chunks))
    for _ in range(chunks):
        max_start = max(result.numel() - chunk_length, 1)
        start = int(torch.randint(0, max_start, (1,), generator=generator).item())
        result[start : start + chunk_length] = 0.0
    return result


def corrupt_audio(waveform: torch.Tensor, condition: str, severity: int, seed: int) -> torch.Tensor:
    if condition == "clean":
        return waveform
    if condition == "snr_noise":
        return add_noise_at_snr(waveform, {1: 20.0, 2: 10.0, 3: 0.0}[severity], seed)
    if condition == "low_volume":
        return waveform * {1: 0.7, 2: 0.4, 3: 0.2}[severity]
    if condition == "clipping":
        quantile = {1: 0.995, 2: 0.98, 3: 0.95}[severity]
        limit = torch.quantile(waveform.abs(), quantile).clamp_min(1e-5)
        return waveform.clamp(-float(limit), float(limit))
    if condition == "dropout":
        return dropout_chunks(waveform, {1: 0.05, 2: 0.15, 3: 0.30}[severity], 5, seed)
    raise ValueError(condition)


def validity_target(condition: str, severity: int) -> int:
    if condition == "clean" or condition == "low_volume":
        return 0
    if condition == "snr_noise":
        return {1: 0, 2: 1, 3: 2}[severity]
    if condition == "clipping":
        return {1: 0, 2: 1, 3: 1}[severity]
    if condition == "dropout":
        return {1: 1, 2: 2, 3: 2}[severity]
    raise ValueError(condition)


DEGRADED_VIEWS = (("snr_noise", 2), ("clipping", 3), ("dropout", 1))
INVALID_VIEWS = (("snr_noise", 3), ("dropout", 2), ("dropout", 3))


class TrainDataset(Dataset):
    def __init__(self, samples: list[AudioSample], sample_rate: int, duration: float, seed: int) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * duration)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        waveform, sample_rate = torchaudio.load(sample.path)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        clean = fix_length(waveform, self.target_samples, True, self.seed + index)
        rng = random.Random(self.seed * 100003 + index)
        clean = clean * rng.uniform(0.9, 1.1)
        if index % 2 == 0:
            validity = 1
            condition, severity = DEGRADED_VIEWS[(index // 2) % len(DEGRADED_VIEWS)]
        else:
            validity = 2
            condition, severity = INVALID_VIEWS[(index // 2) % len(INVALID_VIEWS)]
        view = corrupt_audio(clean, condition, severity, self.seed + index * 17)
        return (
            clean,
            view,
            torch.tensor(sample.label, dtype=torch.float32),
            torch.tensor(validity, dtype=torch.long),
        )


class CleanDataset(Dataset):
    def __init__(self, samples: list[AudioSample], sample_rate: int, duration: float) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * duration)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        waveform, sample_rate = torchaudio.load(sample.path)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        waveform = fix_length(waveform, self.target_samples, False, index)
        return waveform, torch.tensor(sample.label, dtype=torch.float32)


class ValidityDataset(CleanDataset):
    def __init__(
        self,
        samples: list[AudioSample],
        sample_rate: int,
        duration: float,
        seed: int,
        fixed_condition: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(samples, sample_rate, duration)
        self.seed = seed
        self.fixed_condition = fixed_condition

    def __getitem__(self, index: int):
        waveform, label = super().__getitem__(index)
        if self.fixed_condition is None:
            validity = index % 3
            if validity == 0:
                condition, severity = "clean", 0
            elif validity == 1:
                condition, severity = DEGRADED_VIEWS[(index // 3) % len(DEGRADED_VIEWS)]
            else:
                condition, severity = INVALID_VIEWS[(index // 3) % len(INVALID_VIEWS)]
        else:
            condition, severity = self.fixed_condition
            validity = validity_target(condition, severity)
        waveform = corrupt_audio(waveform, condition, severity, self.seed + index * 19)
        return waveform, label, torch.tensor(validity, dtype=torch.long)


class SharedAudioEncoder(nn.Module):
    def __init__(self, width: float) -> None:
        super().__init__()
        channels = [max(4, int(round(value * width))) for value in (16, 32, 64, 96)]
        c1, c2, c3, c4 = channels
        self.output_channels = c4
        self.net = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, 3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class AsnValidityModel(nn.Module):
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
        self.encoder = SharedAudioEncoder(width)
        channels = self.encoder.output_channels
        self.dropout = nn.Dropout(dropout)
        self.risk_head = nn.Linear(channels, 1)
        self.validity_head = nn.Linear(channels, 3)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_features(self.frontend(waveform))

    def forward_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(features)
        dropped = self.dropout(embedding)
        return self.risk_head(dropped).flatten(), self.validity_head(dropped)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def best_f1_threshold(probs: torch.Tensor, labels: torch.Tensor) -> float:
    values = torch.unique(probs.detach().cpu()).tolist()
    labels = labels.detach().cpu().int()
    best = (-1.0, 0.5)
    for threshold in values:
        prediction = (probs.detach().cpu() >= threshold).int()
        score = f1_score(labels.numpy(), prediction.numpy(), zero_division=0)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1]


def risk_metrics(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float | int]:
    probs = probs.detach().cpu()
    labels = labels.detach().cpu().int()
    prediction = (probs >= threshold).int()
    tp = int(((prediction == 1) & (labels == 1)).sum())
    tn = int(((prediction == 0) & (labels == 0)).sum())
    fp = int(((prediction == 1) & (labels == 0)).sum())
    fn = int(((prediction == 0) & (labels == 1)).sum())
    try:
        auc = float(roc_auc_score(labels.numpy(), probs.numpy()))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "f1_positive": float(f1_score(labels.numpy(), prediction.numpy(), zero_division=0)),
        "recall_positive": tp / max(tp + fn, 1),
        "roc_auc": auc,
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def validity_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    prediction = logits.argmax(dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    return {
        "macro_f1": float(f1_score(labels_np, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, prediction)),
        "invalid_recall": float(recall_score(labels_np == 2, prediction == 2, zero_division=0)),
        "confusion_matrix": [
            [int(((labels_np == truth) & (prediction == pred)).sum()) for pred in range(3)]
            for truth in range(3)
        ],
    }


def collect_clean(model: nn.Module, loader: DataLoader, device: torch.device):
    risk, validity, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for waveform, target in loader:
            logits, validity_logits = model(waveform.to(device, non_blocking=True))
            risk.append(logits.cpu())
            validity.append(validity_logits.cpu())
            labels.append(target.cpu())
    return torch.cat(risk), torch.cat(validity), torch.cat(labels)


def collect_validity(model: nn.Module, loader: DataLoader, device: torch.device):
    risk, validity, labels, quality = [], [], [], []
    model.eval()
    with torch.no_grad():
        for waveform, target, quality_target in loader:
            logits, validity_logits = model(waveform.to(device, non_blocking=True))
            risk.append(logits.cpu())
            validity.append(validity_logits.cpu())
            labels.append(target.cpu())
            quality.append(quality_target.cpu())
    return torch.cat(risk), torch.cat(validity), torch.cat(labels), torch.cat(quality)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, multiclass: bool) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = logits.to(device)
    labels = labels.to(device)
    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=80)
    criterion: nn.Module = nn.CrossEntropyLoss() if multiclass else nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        if multiclass:
            loss = criterion(logits / temperature, labels.long())
        else:
            loss = criterion(logits / temperature, labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 10.0).item())


def train_epoch(
    model: AsnValidityModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    risk_criterion: nn.Module,
    device: torch.device,
    quality_weight: float,
    robust_weight: float,
    consistency_weight: float,
) -> dict[str, float]:
    model.train()
    quality_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1.0, 1.0], device=device))
    totals = {"loss": 0.0, "risk": 0.0, "quality": 0.0, "robust": 0.0}
    samples = 0
    for clean, view, target, validity_target_tensor in loader:
        clean = clean.to(device, non_blocking=True)
        view = view.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        validity_target_tensor = validity_target_tensor.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        clean_risk, clean_validity = model(clean)
        view_risk, view_validity = model(view)
        clean_quality_target = torch.zeros_like(validity_target_tensor)
        risk_loss = risk_criterion(clean_risk, target)
        quality_loss = 0.5 * (
            quality_criterion(clean_validity, clean_quality_target)
            + quality_criterion(view_validity, validity_target_tensor)
        )
        degraded = validity_target_tensor == 1
        if bool(degraded.any()):
            robust_loss = risk_criterion(view_risk[degraded], target[degraded])
            consistency = torch.nn.functional.mse_loss(
                torch.sigmoid(view_risk[degraded]), torch.sigmoid(clean_risk[degraded]).detach()
            )
        else:
            robust_loss = risk_loss.new_zeros(())
            consistency = risk_loss.new_zeros(())
        loss = risk_loss + quality_weight * quality_loss + robust_weight * robust_loss + consistency_weight * consistency
        loss.backward()
        optimizer.step()
        batch = target.numel()
        samples += batch
        totals["loss"] += float(loss.item()) * batch
        totals["risk"] += float(risk_loss.item()) * batch
        totals["quality"] += float(quality_loss.item()) * batch
        totals["robust"] += float(robust_loss.item()) * batch
    return {key: value / max(samples, 1) for key, value in totals.items()}


def evaluate_validation(
    model: AsnValidityModel,
    clean_loader: DataLoader,
    validity_loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    clean_logits, _, clean_labels = collect_clean(model, clean_loader, device)
    threshold = best_f1_threshold(torch.sigmoid(clean_logits), clean_labels)
    clean = risk_metrics(torch.sigmoid(clean_logits), clean_labels, threshold)
    _, validity_logits, _, validity_labels = collect_validity(model, validity_loader, device)
    validity = validity_metrics(validity_logits, validity_labels)
    selection = 0.7 * float(clean["f1_positive"]) + 0.3 * float(validity["macro_f1"])
    return {"clean": clean, "validity": validity, "selection_score": selection}


def evaluate_condition(
    model: AsnValidityModel,
    samples: list[AudioSample],
    args: argparse.Namespace,
    condition: str,
    severity: int,
    risk_temperature: float,
    validity_temperature: float,
    threshold: float,
    device: torch.device,
) -> dict[str, object]:
    dataset = ValidityDataset(samples, args.sample_rate, args.duration, args.seed, (condition, severity))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    risk_logits, validity_logits, labels, validity_labels = collect_validity(model, loader, device)
    risk_probs = torch.sigmoid(risk_logits / risk_temperature)
    validity_probs = torch.softmax(validity_logits / validity_temperature, dim=1)
    validity_pred = validity_probs.argmax(dim=1)
    base = risk_metrics(risk_probs, labels, threshold)
    validity = validity_metrics(validity_logits / validity_temperature, validity_labels)
    accepted = validity_pred != 2
    if bool(accepted.any()) and len(torch.unique(labels[accepted])) > 1:
        accepted_metrics = risk_metrics(risk_probs[accepted], labels[accepted], threshold)
    else:
        accepted_metrics = {"f1_positive": float("nan"), "recall_positive": float("nan")}
    entropy = -(
        risk_probs.clamp(1e-7, 1 - 1e-7) * torch.log2(risk_probs.clamp(1e-7, 1 - 1e-7))
        + (1 - risk_probs).clamp(1e-7, 1 - 1e-7) * torch.log2((1 - risk_probs).clamp(1e-7, 1 - 1e-7))
    )
    confidence = 1 - entropy
    incorrect = (risk_probs >= threshold).int() != labels.int()
    base_hce = float((incorrect & (confidence >= 0.8)).float().mean())
    gated_hce = float((incorrect & (confidence >= 0.8) & accepted).float().mean())
    return {
        "condition": condition,
        "severity": severity,
        "target_validity": VALIDITY_NAMES[validity_target(condition, severity)],
        "risk": base,
        "validity": validity,
        "coverage": float(accepted.float().mean()),
        "accepted_risk": accepted_metrics,
        "base_high_confidence_error_rate": base_hce,
        "gated_high_confidence_error_rate": gated_hce,
        "mean_confidence": float(confidence.mean()),
        "mean_invalid_probability": float(validity_probs[:, 2].mean()),
    }


def write_report(result: dict[str, object], output_dir: Path) -> None:
    test = result.get("test")
    lines = [
        "# ASN Signal-Validity Multi-task Result",
        "",
        f"- Run: `{result['run_id']}`",
        f"- Parameters: {result['model']['parameters']:,}",
        f"- Estimated INT8 weights: {result['model']['int8_weight_kib']:.2f} KiB",
        f"- Best epoch: {result['training']['best_epoch']}",
        f"- Best validation selection: {result['training']['best_validation_selection_score']:.4f}",
        "",
    ]
    if test is None:
        lines.extend(["Validation-only run; the official test split was not evaluated.", ""])
    else:
        lines.extend(
            [
                "## Clean test",
                "",
                f"- F1: {test['clean']['risk']['f1_positive']:.4f}",
                f"- Recall: {test['clean']['risk']['recall_positive']:.4f}",
                f"- AUROC: {test['clean']['risk']['roc_auc']:.4f}",
                f"- Clean validity prediction: {test['clean']['validity']['macro_f1']:.4f} Macro-F1 over the one present class (diagnostic only)",
                "",
                "## Robustness and status gating",
                "",
                "| Condition | Target status | Risk F1 | Invalid recall | Coverage | Base HC error | Gated HC error |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in test["robustness"]:
            lines.append(
                f"| {row['condition']}-s{row['severity']} | {row['target_validity']} | "
                f"{row['risk']['f1_positive']:.4f} | {row['validity']['invalid_recall']:.4f} | "
                f"{row['coverage']:.4f} | {row['base_high_confidence_error_rate']:.4f} | "
                f"{row['gated_high_confidence_error_rate']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Claims boundary",
            "",
            "The validity states are supervised with controlled synthetic corruptions. They are node-status proxies, not field OOD certification. The risk output remains an acoustic abnormality proxy.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def limit_samples(samples: list[AudioSample], limit: int | None, seed: int) -> list[AudioSample]:
    if not limit or len(samples) <= limit:
        return samples
    rng = random.Random(seed)
    result = samples.copy()
    rng.shuffle(result)
    return result[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=PROJECT_ROOT / "experiments" / "conference_grouped" / "asn_audio_index_grouped.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_signal_validity")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--quality-weight", type=float, default=0.25)
    parser.add_argument("--robust-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    train_samples = limit_samples(read_samples(args.index, "train", None), args.max_train, args.seed)
    val_samples = limit_samples(read_samples(args.index, "val", None), args.max_val, args.seed)
    test_samples = limit_samples(read_samples(args.index, "test", None), args.max_test, args.seed)
    device = torch.device(args.device)
    model = AsnValidityModel(args.sample_rate, args.n_fft, args.hop_length, args.n_mels, args.width, args.dropout).to(device)
    run_id = f"asn_signal_validity_w{str(args.width).replace('.', 'p')}_d{str(args.duration).replace('.', 'p')}s_seed{args.seed}"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(TrainDataset(train_samples, args.sample_rate, args.duration, args.seed), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_clean_loader = DataLoader(CleanDataset(val_samples, args.sample_rate, args.duration), batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_validity_loader = DataLoader(ValidityDataset(val_samples, args.sample_rate, args.duration, args.seed), batch_size=args.batch_size, shuffle=False, num_workers=0)
    risk_criterion = build_loss(train_samples, device, "auto")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history = []
    best_path = output_dir / "best.pt"
    started = time.perf_counter()
    print(f"Run={run_id} device={device} parameters={count_parameters(model)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train = train_epoch(model, train_loader, optimizer, risk_criterion, device, args.quality_weight, args.robust_weight, args.consistency_weight)
        validation = evaluate_validation(model, val_clean_loader, val_validity_loader, device)
        history.append({"epoch": epoch, "train": train, "validation": validation})
        print(
            f"{run_id} epoch={epoch:02d} loss={train['loss']:.4f} "
            f"risk_f1={validation['clean']['f1_positive']:.4f} "
            f"validity_f1={validation['validity']['macro_f1']:.4f} "
            f"invalid_recall={validation['validity']['invalid_recall']:.4f} "
            f"selection={validation['selection_score']:.4f}", flush=True
        )
        score = float(validation["selection_score"])
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "validation": validation, "run_id": run_id}, best_path)
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    val_risk_logits, _, val_labels = collect_clean(model, val_clean_loader, device)
    _, val_validity_logits, _, val_validity_labels = collect_validity(model, val_validity_loader, device)
    risk_temperature = fit_temperature(val_risk_logits, val_labels, False)
    validity_temperature = fit_temperature(val_validity_logits, val_validity_labels, True)
    threshold = best_f1_threshold(torch.sigmoid(val_risk_logits / risk_temperature), val_labels)

    result: dict[str, object] = {
        "run_id": run_id,
        "protocol": {"index": str(args.index), "seed": args.seed, "test_used_for_selection": False, "validity_names": VALIDITY_NAMES},
        "model": {"parameters": count_parameters(model), "fp32_weight_kib": count_parameters(model) * 4 / 1024, "int8_weight_kib": count_parameters(model) / 1024},
        "loss_weights": {"quality": args.quality_weight, "robust": args.robust_weight, "consistency": args.consistency_weight},
        "training": {"epochs_completed": len(history), "best_epoch": best_epoch, "best_validation_selection_score": best_score, "elapsed_seconds": time.perf_counter() - started, "sample_counts": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)}, "history": history},
        "calibration": {"risk_temperature": risk_temperature, "validity_temperature": validity_temperature, "risk_threshold": threshold},
        "test": None,
        "claims_boundary": "Synthetic signal-validity supervision; acoustic abnormality risk proxy; no hardware measurements.",
    }

    if not args.validation_only:
        test_clean_loader = DataLoader(CleanDataset(test_samples, args.sample_rate, args.duration), batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_logits, test_validity_logits, test_labels = collect_clean(model, test_clean_loader, device)
        clean_risk = risk_metrics(torch.sigmoid(test_logits / risk_temperature), test_labels, threshold)
        clean_validity_labels = torch.zeros(len(test_labels), dtype=torch.long)
        clean_validity = validity_metrics(test_validity_logits / validity_temperature, clean_validity_labels)
        robustness = [evaluate_condition(model, test_samples, args, condition, severity, risk_temperature, validity_temperature, threshold, device) for condition, severity in CONDITIONS]
        result["test"] = {"clean": {"risk": clean_risk, "validity": clean_validity}, "robustness": robustness}

    torch.save({"model": model.state_dict(), "args": vars(args), "calibration": result["calibration"], "run_id": run_id}, output_dir / "deployed_state_dict.pt")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, output_dir)
    print(f"Completed {run_id}; validation_only={args.validation_only}", flush=True)


if __name__ == "__main__":
    main()
