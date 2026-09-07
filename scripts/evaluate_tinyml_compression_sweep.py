"""Compare TinyML compression candidates against memory and accuracy limits.

The sweep holds evaluation data fixed while varying student choice, structured
pruning and quantisation estimates. A candidate is not considered deployable on
file size alone; performance and peak working-memory proxies are reported
together.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import torch
import torchaudio
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_asn_audio import AsnModel, AudioSample, read_samples as read_asn_samples
from train_vsn_binary import Sample, build_model, build_transforms, read_samples as read_vsn_samples


SRAM_LIMIT_BYTES = 2 * 1024 * 1024


def as_namespace(data: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def mb(value: float) -> float:
    return float(value) / (1024.0 * 1024.0)


def stratified_limit_vsn(samples: list[Sample], limit: int | None, seed: int) -> list[Sample]:
    if limit is None or limit <= 0 or len(samples) <= limit:
        return samples
    rng = random.Random(seed)
    by_label: dict[int, list[Sample]] = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)
    for group in by_label.values():
        rng.shuffle(group)
    selected: list[Sample] = []
    for label, group in by_label.items():
        quota = max(1, round(limit * len(group) / len(samples)))
        selected.extend(group[:quota])
    rng.shuffle(selected)
    return selected[:limit]


def stratified_limit_asn(samples: list[AudioSample], limit: int | None, seed: int) -> list[AudioSample]:
    if limit is None or limit <= 0 or len(samples) <= limit:
        return samples
    rng = random.Random(seed)
    by_label: dict[int, list[AudioSample]] = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)
    for group in by_label.values():
        rng.shuffle(group)
    selected: list[AudioSample] = []
    for label, group in by_label.items():
        quota = max(1, round(limit * len(group) / len(samples)))
        selected.extend(group[:quota])
    rng.shuffle(selected)
    return selected[:limit]


class VsnEvalDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int) -> None:
        _, self.transform = build_transforms(image_size)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        return self.transform(image), torch.tensor(sample.label, dtype=torch.float32)


class AsnWindowDataset(Dataset):
    def __init__(self, samples: list[AudioSample], sample_rate: int, duration: float) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * duration)

    def __len__(self) -> int:
        return len(self.samples)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.mean(dim=0)
        n = waveform.numel()
        if n == self.target_samples:
            return waveform
        if n > self.target_samples:
            start = (n - self.target_samples) // 2
            return waveform[start : start + self.target_samples]
        return torch.nn.functional.pad(waveform, (0, self.target_samples - n))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        waveform, sr = torchaudio.load(sample.path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = self._fix_length(waveform)
        return waveform.float(), torch.tensor(sample.label, dtype=torch.float32)


def binary_metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
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
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).flatten().detach().cpu()
            scores.append(torch.sigmoid(logits))
            labels.append(y.detach().cpu())
    return binary_metrics(torch.cat(scores), torch.cat(labels))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def prunable_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    modules: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules.append((module, "weight"))
    return modules


def apply_global_pruning(model: nn.Module, amount: float) -> None:
    """Apply unstructured magnitude pruning consistently across eligible layers."""
    if amount <= 0:
        return
    modules = prunable_modules(model)
    prune.global_unstructured(modules, pruning_method=prune.L1Unstructured, amount=amount)
    for module, name in modules:
        prune.remove(module, name)


def sparsity(model: nn.Module) -> float:
    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.detach()
            total += weight.numel()
            zeros += int((weight == 0).sum())
    return zeros / max(total, 1)


def module_output_numel(output: object) -> int:
    if isinstance(output, torch.Tensor):
        return output.numel()
    if isinstance(output, (list, tuple)):
        return sum(module_output_numel(item) for item in output)
    if isinstance(output, dict):
        return sum(module_output_numel(item) for item in output.values())
    return 0


def activation_summary(model: nn.Module, dummy_input: torch.Tensor) -> dict[str, int]:
    model.eval()
    outputs: list[int] = []
    handles = []

    def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        numel = module_output_numel(output)
        if numel > 0:
            outputs.append(numel)

    for module in model.modules():
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        _ = model(dummy_input)
    for handle in handles:
        handle.remove()
    return {"peak_numel": max(outputs) if outputs else 0, "sum_numel": sum(outputs)}


def estimate_dense_int8_weight_mb(params: int) -> float:
    return mb(params)


def estimate_sparse_int8_weight_mb(params: int, sparse_fraction: float) -> float:
    nonzero = params * (1.0 - sparse_fraction)
    # Rough CSR/COO-like overhead estimate: 1 byte value + 2 bytes index per nonzero.
    return mb(nonzero * 3)


def vsn_sram_estimates(model: nn.Module, image_size: int) -> dict[str, float]:
    dummy = torch.zeros(1, 3, image_size, image_size)
    activations = activation_summary(model, dummy)
    input_uint8 = 3 * image_size * image_size
    return {
        "peak_runtime_sram_int8_mb": mb(input_uint8 + activations["peak_numel"]),
        "conservative_runtime_sram_int8_mb": mb(input_uint8 + activations["sum_numel"]),
    }


def asn_sram_estimates(model: nn.Module, sample_rate: int, duration: float) -> dict[str, float]:
    target_samples = int(sample_rate * duration)
    dummy = torch.zeros(1, target_samples)
    activations = activation_summary(model, dummy)
    input_int16 = target_samples * 2
    return {
        "peak_runtime_sram_int8_mb": mb(input_int16 + activations["peak_numel"]),
        "conservative_runtime_sram_int8_mb": mb(input_int16 + activations["sum_numel"]),
    }


def load_vsn(run_dir: Path) -> tuple[nn.Module, SimpleNamespace]:
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    args = as_namespace(checkpoint["args"])
    model = build_model(args.model, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, args


def load_asn(run_dir: Path) -> tuple[nn.Module, SimpleNamespace]:
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    args = as_namespace(checkpoint["args"])
    model = AsnModel(args.sample_rate, args.n_fft, args.hop_length, args.n_mels)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, args


def run_vsn_sweep(args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    base_model, model_args = load_vsn(args.vsn_run_dir)
    samples = read_vsn_samples(Path(model_args.index), "test", set(model_args.sources) if isinstance(model_args.sources, list) else None)
    samples = stratified_limit_vsn(samples, args.max_vsn_test, args.seed)
    rows: list[dict[str, object]] = []
    params = count_parameters(base_model)
    for image_size in args.vsn_image_sizes:
        loader = DataLoader(
            VsnEvalDataset(samples, image_size),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        for prune_amount in args.prune_amounts:
            model = copy.deepcopy(base_model)
            apply_global_pruning(model, prune_amount)
            sp = sparsity(model)
            sram = vsn_sram_estimates(model, image_size)
            model = model.to(device)
            metrics = eval_model(model, loader, device)
            rows.append(
                {
                    "node": "VSN",
                    "compression": f"resize{image_size}_prune{prune_amount:.2f}_int8_est",
                    "image_size": image_size,
                    "audio_duration": "",
                    "prune_amount": prune_amount,
                    "measured_sparsity": sp,
                    "parameters": params,
                    "dense_int8_weight_mb": estimate_dense_int8_weight_mb(params),
                    "sparse_int8_weight_est_mb": estimate_sparse_int8_weight_mb(params, sp),
                    "peak_runtime_sram_int8_mb": sram["peak_runtime_sram_int8_mb"],
                    "conservative_runtime_sram_int8_mb": sram["conservative_runtime_sram_int8_mb"],
                    "strict_dense_peak_mb": estimate_dense_int8_weight_mb(params) + sram["peak_runtime_sram_int8_mb"],
                    "strict_dense_conservative_mb": estimate_dense_int8_weight_mb(params) + sram["conservative_runtime_sram_int8_mb"],
                    "sample_count": len(samples),
                    **metrics,
                }
            )
            print(f"VSN image={image_size} prune={prune_amount:.2f} f1={metrics['f1_positive']:.4f}")
    return rows


def run_asn_sweep(args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    base_model, model_args = load_asn(args.asn_run_dir)
    samples = read_asn_samples(Path(model_args.index), "test", set(model_args.sources) if isinstance(model_args.sources, list) else None)
    samples = stratified_limit_asn(samples, args.max_asn_test, args.seed)
    rows: list[dict[str, object]] = []
    params = count_parameters(base_model)
    for duration in args.asn_durations:
        loader = DataLoader(
            AsnWindowDataset(samples, int(model_args.sample_rate), duration),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        for prune_amount in args.prune_amounts:
            model = copy.deepcopy(base_model)
            apply_global_pruning(model, prune_amount)
            sp = sparsity(model)
            sram = asn_sram_estimates(model, int(model_args.sample_rate), duration)
            model = model.to(device)
            metrics = eval_model(model, loader, device)
            rows.append(
                {
                    "node": "ASN",
                    "compression": f"duration{duration:g}s_prune{prune_amount:.2f}_int8_est",
                    "image_size": "",
                    "audio_duration": duration,
                    "prune_amount": prune_amount,
                    "measured_sparsity": sp,
                    "parameters": params,
                    "dense_int8_weight_mb": estimate_dense_int8_weight_mb(params),
                    "sparse_int8_weight_est_mb": estimate_sparse_int8_weight_mb(params, sp),
                    "peak_runtime_sram_int8_mb": sram["peak_runtime_sram_int8_mb"],
                    "conservative_runtime_sram_int8_mb": sram["conservative_runtime_sram_int8_mb"],
                    "strict_dense_peak_mb": estimate_dense_int8_weight_mb(params) + sram["peak_runtime_sram_int8_mb"],
                    "strict_dense_conservative_mb": estimate_dense_int8_weight_mb(params) + sram["conservative_runtime_sram_int8_mb"],
                    "sample_count": len(samples),
                    **metrics,
                }
            )
            print(f"ASN duration={duration:g}s prune={prune_amount:.2f} f1={metrics['f1_positive']:.4f}")
    return rows


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "compression_sweep_metrics.csv"
    json_path = output_dir / "compression_sweep_metrics.json"
    report_path = output_dir / "REPORT.md"
    plot_path = output_dir / "compression_tradeoff.png"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_plot(rows, plot_path)
    report_path.write_text(report_text(rows), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {report_path}")


def write_plot(rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)
    for node, marker in [("VSN", "o"), ("ASN", "s")]:
        subset = [row for row in rows if row["node"] == node]
        x = [float(row["strict_dense_peak_mb"]) for row in subset]
        y = [float(row["f1_positive"]) for row in subset]
        ax.scatter(x, y, marker=marker, label=node, alpha=0.8)
    ax.axvline(2.0, linestyle="--", color="black", linewidth=1, label="2 MB")
    ax.set_xlabel("Dense INT8 weights + peak runtime SRAM estimate (MB)")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates = [row for row in rows if float(row["strict_dense_peak_mb"]) <= 2.0]
    if not candidates:
        candidates = rows
    return sorted(candidates, key=lambda row: (str(row["node"]), -float(row["f1_positive"]), float(row["strict_dense_peak_mb"])))


def report_text(rows: list[dict[str, object]]) -> str:
    best = best_rows(rows)
    lines = [
        "# TinyML Compression Sweep Report",
        "",
        "## Scope",
        "",
        "- This is a screening experiment over compression-friendly configurations, not final deployment quantization.",
        "- VSN combinations vary image size and global unstructured pruning.",
        "- ASN combinations vary audio window duration and global unstructured pruning.",
        "- INT8 memory is estimated from parameter and activation counts. Actual deployment needs tensor-arena profiling.",
        "",
        "## Best Configurations Under Dense INT8 Peak Estimate",
        "",
        "| Node | Compression | F1 | Accuracy | Dense INT8 weights MB | Strict peak MB | Conservative SRAM MB | Sparsity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    emitted = set()
    for row in best:
        node = str(row["node"])
        if node in emitted:
            continue
        emitted.add(node)
        lines.append(
            "| {node} | {comp} | {f1:.4f} | {acc:.4f} | {w:.3f} | {strict:.3f} | {cons:.3f} | {sp:.3f} |".format(
                node=node,
                comp=row["compression"],
                f1=float(row["f1_positive"]),
                acc=float(row["accuracy"]),
                w=float(row["dense_int8_weight_mb"]),
                strict=float(row["strict_dense_peak_mb"]),
                cons=float(row["conservative_runtime_sram_int8_mb"]),
                sp=float(row["measured_sparsity"]),
            )
        )
    lines.extend(
        [
            "",
            "## Full Results",
            "",
            "| Node | Compression | F1 | Recall | Accuracy | Strict peak MB | Strict conservative MB |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["node"]), str(item["compression"]))):
        lines.append(
            "| {node} | {comp} | {f1:.4f} | {rec:.4f} | {acc:.4f} | {peak:.3f} | {cons:.3f} |".format(
                node=row["node"],
                comp=row["compression"],
                f1=float(row["f1_positive"]),
                rec=float(row["recall_positive"]),
                acc=float(row["accuracy"]),
                peak=float(row["strict_dense_peak_mb"]),
                cons=float(row["strict_dense_conservative_mb"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If smaller VSN image sizes preserve F1, they become strong student-model candidates for later distillation.",
            "- Pruning here is unstructured and mainly tests accuracy tolerance. Real TinyML benefit requires sparse-aware storage/runtime or structured pruning.",
            "- ASN duration reduction is a proxy for streaming/chunked inference. If shorter windows keep F1 high, it reduces SRAM pressure.",
            "- Knowledge distillation should focus on the best compression-friendly configurations from this sweep.",
            "",
            "## Files",
            "",
            "- Metrics CSV: `compression_sweep_metrics.csv`",
            "- Metrics JSON: `compression_sweep_metrics.json`",
            "- Trade-off plot: `compression_tradeoff.png`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TinyML compression-friendly VSN/ASN configurations.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "tinyml_compression_sweep")
    parser.add_argument("--vsn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_binary_mixed_full" / "mobilenet_v3_small_pretrained")
    parser.add_argument("--asn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_audio_formal" / "tiny_logmel_cnn")
    parser.add_argument("--vsn-image-sizes", nargs="+", type=int, default=[224, 160, 128, 96])
    parser.add_argument("--asn-durations", nargs="+", type=float, default=[10.0, 5.0, 2.5, 1.0])
    parser.add_argument("--prune-amounts", nargs="+", type=float, default=[0.0, 0.3, 0.5])
    parser.add_argument("--max-vsn-test", type=int, default=1500)
    parser.add_argument("--max-asn-test", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rows: list[dict[str, object]] = []
    rows.extend(run_vsn_sweep(args, device))
    rows.extend(run_asn_sweep(args, device))
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
