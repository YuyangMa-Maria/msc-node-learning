"""Measure ASN robustness to additive noise and temporal dropout.

Signal-to-noise ratio and missing-window severity are controlled independently.
The experiment therefore tests degradation trends under repeatable conditions,
not field acoustic-emission validity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
import torchaudio
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_asn_audio import PROJECT_ROOT, AsnModel, AudioSample, read_samples
from train_asn_student import AsnStudentModel


def as_namespace(data: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def source_set(value: object) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return set(str(item) for item in value)
    return None


def fix_length(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    waveform = waveform.mean(dim=0)
    n = waveform.numel()
    if n == target_samples:
        return waveform
    if n > target_samples:
        start = (n - target_samples) // 2
        return waveform[start : start + target_samples]
    return torch.nn.functional.pad(waveform, (0, target_samples - n))


def add_noise_at_snr(waveform: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(waveform.shape, generator=generator)
    signal_power = waveform.pow(2).mean().clamp_min(1e-12)
    noise_power = noise.pow(2).mean().clamp_min(1e-12)
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return waveform + noise * torch.sqrt(target_noise_power / noise_power)


def dropout_chunks(waveform: torch.Tensor, drop_ratio: float, chunks: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    out = waveform.clone()
    n = out.numel()
    chunk_len = max(1, int(n * drop_ratio / chunks))
    for _ in range(chunks):
        max_start = max(n - chunk_len, 1)
        start = int(torch.randint(0, max_start, (1,), generator=generator).item())
        out[start : start + chunk_len] = 0.0
    return out


def corrupt_audio(waveform: torch.Tensor, condition: str, severity: int, seed: int) -> torch.Tensor:
    if condition == "clean":
        return waveform
    if condition == "snr_noise":
        snr = {1: 20.0, 2: 10.0, 3: 0.0}[severity]
        return add_noise_at_snr(waveform, snr, seed)
    if condition == "low_volume":
        factor = {1: 0.7, 2: 0.4, 3: 0.2}[severity]
        return waveform * factor
    if condition == "clipping":
        quantile = {1: 0.995, 2: 0.98, 3: 0.95}[severity]
        limit = torch.quantile(waveform.abs(), quantile).clamp_min(1e-5)
        return waveform.clamp(-float(limit), float(limit))
    if condition == "dropout":
        ratio = {1: 0.05, 2: 0.15, 3: 0.30}[severity]
        return dropout_chunks(waveform, ratio, chunks=5, seed=seed)
    raise ValueError(f"Unsupported condition: {condition}")


class RobustAsnDataset(Dataset):
    def __init__(
        self,
        samples: list[AudioSample],
        sample_rate: int,
        target_samples: int,
        condition: str,
        severity: int,
        seed: int,
    ) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.target_samples = target_samples
        self.condition = condition
        self.severity = severity
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        waveform, sr = torchaudio.load(sample.path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = fix_length(waveform, self.target_samples)
        waveform = corrupt_audio(waveform, self.condition, self.severity, self.seed + index)
        return waveform.float(), torch.tensor(sample.label, dtype=torch.float32)


def confidence_from_risk(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-7, 1 - 1e-7)
    entropy = -(probs * torch.log2(probs) + (1 - probs) * torch.log2(1 - probs))
    return 1.0 - entropy


def metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    scores = scores.detach().cpu()
    labels = labels.detach().cpu().int()
    pred = (scores >= threshold).int()
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    try:
        auc = float(roc_auc_score(labels.numpy(), scores.numpy()))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy,
        "precision_positive": precision,
        "recall_positive": recall,
        "specificity": specificity,
        "f1_positive": f1,
        "roc_auc": auc,
        "mean_risk_score": float(scores.mean()),
        "mean_confidence": float(confidence_from_risk(scores).mean()),
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def collect_scores(
    model: nn.Module,
    samples: list[AudioSample],
    sample_rate: int,
    target_samples: int,
    temperature: float,
    condition: str,
    severity: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = RobustAsnDataset(samples, sample_rate, target_samples, condition, severity, seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for waveform, y in loader:
            waveform = waveform.to(device, non_blocking=True)
            logits = model(waveform).flatten().detach().cpu()
            scores.append(torch.sigmoid(logits / temperature))
            labels.append(y.detach().cpu())
    return torch.cat(scores), torch.cat(labels).float()


def build_model(checkpoint: dict[str, object], train_args: SimpleNamespace, device: torch.device) -> nn.Module:
    if hasattr(train_args, "width"):
        model = AsnStudentModel(
            sample_rate=int(train_args.sample_rate),
            n_fft=int(train_args.n_fft),
            hop_length=int(train_args.hop_length),
            n_mels=int(train_args.n_mels),
            width=float(train_args.width),
            dropout=float(getattr(train_args, "dropout", 0.15)),
        )
    else:
        model = AsnModel(
            int(train_args.sample_rate),
            int(train_args.n_fft),
            int(train_args.hop_length),
            int(train_args.n_mels),
        )
    model.load_state_dict(checkpoint["model"])  # type: ignore[arg-type]
    return model.to(device)


def write_outputs(rows: list[dict[str, object]], output_dir: Path, threshold: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "asn_robustness_metrics.csv"
    json_path = output_dir / "asn_robustness_metrics.json"
    md_path = output_dir / "REPORT.md"
    plot_path = output_dir / "asn_robustness_f1_recall.png"
    fieldnames = [
        "condition",
        "severity",
        "accuracy",
        "precision_positive",
        "recall_positive",
        "specificity",
        "f1_positive",
        "roc_auc",
        "mean_risk_score",
        "mean_confidence",
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    json_path.write_text(json.dumps({"decision_threshold": threshold, "results": rows}, indent=2), encoding="utf-8")

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=140, sharey=True)
    for ax, metric_name in zip(axes, ["f1_positive", "recall_positive"]):
        for condition, condition_rows in grouped.items():
            condition_rows = sorted(condition_rows, key=lambda item: int(item["severity"]))
            x = [int(row["severity"]) for row in condition_rows]
            y = [float(row[metric_name]) for row in condition_rows]
            ax.plot(x, y, marker="o", label=condition)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("Severity")
        ax.set_ylabel(metric_name.replace("_positive", "").upper())
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    lines = [
        "# ASN Robustness Report",
        "",
        "## Setup",
        "",
        "- Model: calibrated `tiny_logmel_cnn` acoustic node",
        "- Dataset split: ASN held-out test split",
        "- Threshold: fixed calibrated validation threshold",
        "- Robustness corruptions: SNR noise, low volume, clipping, and temporal dropout",
        "",
        "## Results",
        "",
        "| Condition | Severity | Accuracy | Precision | Recall | F1 | ROC-AUC | Mean confidence | Confusion matrix (TN/FP/FN/TP) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        auc = float(row["roc_auc"])
        auc_text = "nan" if math.isnan(auc) else f"{auc:.4f}"
        lines.append(
            "| {condition} | {severity} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {auc} | {conf:.4f} | {tn}/{fp}/{fn}/{tp} |".format(
                condition=row["condition"],
                severity=row["severity"],
                acc=row["accuracy"],
                prec=row["precision_positive"],
                rec=row["recall_positive"],
                f1=row["f1_positive"],
                auc=auc_text,
                conf=row["mean_confidence"],
                tn=row["tn"],
                fp=row["fp"],
                fn=row["fn"],
                tp=row["tp"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This experiment tests whether the acoustic node remains reliable when the microphone signal is degraded. A strong drop under high noise or dropout supports the need for confidence/status-aware fusion with VSN and VBN rather than relying on ASN alone.",
            "",
            "## Files",
            "",
            "- Metrics CSV: `asn_robustness_metrics.csv`",
            "- Metrics JSON: `asn_robustness_metrics.json`",
            "- Plot: `asn_robustness_f1_recall.png`",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASN acoustic robustness under signal degradations.")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_audio_formal" / "tiny_logmel_cnn")
    parser.add_argument("--calibration-json", type=Path, default=PROJECT_ROOT / "outputs" / "asn_calibration" / "asn_audio_formal_tiny_logmel_cnn_calibration.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_robustness")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.run_dir / "best.pt", map_location=device)
    train_args = as_namespace(checkpoint["args"])
    index = Path(train_args.index)
    if not index.is_absolute():
        index = PROJECT_ROOT / index
    sources = source_set(getattr(train_args, "sources", None))
    samples = read_samples(index, args.split, sources)
    target_samples = int(train_args.sample_rate * train_args.duration)

    model = build_model(checkpoint, train_args, device)
    calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))
    temperature = float(calibration["temperature"])
    threshold = float(calibration["decision_threshold"])

    conditions = [("clean", 0)]
    for condition in ["snr_noise", "low_volume", "clipping", "dropout"]:
        for severity in [1, 2, 3]:
            conditions.append((condition, severity))

    rows: list[dict[str, object]] = []
    for condition, severity in conditions:
        scores, labels = collect_scores(
            model,
            samples,
            int(train_args.sample_rate),
            target_samples,
            temperature,
            condition,
            severity,
            args.batch_size,
            device,
            args.seed,
        )
        result = metrics(scores, labels, threshold)
        row = {"condition": condition, "severity": severity, **result}
        rows.append(row)
        print(f"{condition}_{severity}: f1={result['f1_positive']:.4f} recall={result['recall_positive']:.4f}")
    write_outputs(rows, args.output_dir, threshold)


if __name__ == "__main__":
    main()
