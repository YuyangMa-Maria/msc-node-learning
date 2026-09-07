"""Evaluate the quantised ASN signal-validity gate before deployment."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import DeQuantStub, QuantStub, convert, fuse_modules, get_default_qconfig, prepare
from torch.utils.data import DataLoader

from train_asn_audio import PROJECT_ROOT, read_samples
from train_asn_signal_validity import (
    CONDITIONS,
    AsnValidityModel,
    ValidityDataset,
    risk_metrics,
    validity_metrics,
)


SEVERE = (("snr_noise", 3), ("dropout", 2), ("dropout", 3))


class DualHeadFeatureModel(nn.Module):
    """Quantizable CNN and heads; the log-mel DSP frontend is intentionally external."""

    def __init__(self, source: AsnValidityModel) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.encoder = copy.deepcopy(source.encoder)
        self.risk_head = copy.deepcopy(source.risk_head)
        self.validity_head = copy.deepcopy(source.validity_head)
        self.dequant_risk = DeQuantStub()
        self.dequant_validity = DeQuantStub()

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(self.quant(features))
        risk = self.dequant_risk(self.risk_head(embedding)).flatten()
        validity = self.dequant_validity(self.validity_head(embedding))
        return risk, validity


def namespace(values: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**values)


def load_source(checkpoint_path: Path) -> tuple[AsnValidityModel, dict[str, object], SimpleNamespace]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = namespace(checkpoint["args"])
    model = AsnValidityModel(
        int(args.sample_rate),
        int(args.n_fft),
        int(args.hop_length),
        int(args.n_mels),
        float(args.width),
        float(args.dropout),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, args


def fuse_encoder(model: DualHeadFeatureModel) -> None:
    model.eval()
    fuse_modules(
        model.encoder.net,
        [["0", "1", "2"], ["4", "5", "6"], ["8", "9", "10"], ["12", "13", "14"]],
        inplace=True,
    )


def make_loader(
    samples,
    args: SimpleNamespace,
    seed: int,
    condition: str,
    severity: int,
    batch_size: int,
) -> DataLoader:
    dataset = ValidityDataset(
        samples,
        int(args.sample_rate),
        float(args.duration),
        seed,
        (condition, severity),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def calibrate(
    prepared: nn.Module,
    frontend: nn.Module,
    samples,
    args: SimpleNamespace,
    seed: int,
    max_samples: int,
) -> None:
    conditions = (("clean", 0), ("snr_noise", 1), ("snr_noise", 2), ("snr_noise", 3), ("clipping", 3), ("dropout", 1), ("dropout", 2), ("dropout", 3))
    per_condition = max(1, max_samples // len(conditions))
    prepared.eval()
    frontend.eval()
    with torch.no_grad():
        for condition, severity in conditions:
            loader = make_loader(samples[:per_condition], args, seed, condition, severity, min(32, per_condition))
            for waveform, _, _ in loader:
                prepared(frontend(waveform))


def sigmoid(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.sigmoid(logits / temperature)


def condition_metrics(
    risk_logits: torch.Tensor,
    validity_logits: torch.Tensor,
    labels: torch.Tensor,
    validity_labels: torch.Tensor,
    risk_temperature: float,
    validity_temperature: float,
    threshold: float,
) -> dict[str, object]:
    risk_probs = sigmoid(risk_logits, risk_temperature)
    validity_probs = torch.softmax(validity_logits / validity_temperature, dim=1)
    validity_pred = validity_probs.argmax(dim=1)
    accepted = validity_pred != 2
    risk = risk_metrics(risk_probs, labels, threshold)
    validity = validity_metrics(validity_logits / validity_temperature, validity_labels)
    entropy = -(
        risk_probs.clamp(1e-7, 1 - 1e-7) * torch.log2(risk_probs.clamp(1e-7, 1 - 1e-7))
        + (1 - risk_probs).clamp(1e-7, 1 - 1e-7) * torch.log2((1 - risk_probs).clamp(1e-7, 1 - 1e-7))
    )
    confidence = 1 - entropy
    incorrect = (risk_probs >= threshold).int() != labels.int()
    return {
        "risk": risk,
        "validity": validity,
        "coverage": float(accepted.float().mean()),
        "high_confidence_error_rate": float((incorrect & (confidence >= 0.8)).float().mean()),
        "gated_high_confidence_error_rate": float((incorrect & (confidence >= 0.8) & accepted).float().mean()),
        "mean_invalid_probability": float(validity_probs[:, 2].mean()),
    }


def collect_pair(
    frontend: nn.Module,
    fp32: nn.Module,
    int8: nn.Module,
    loader: DataLoader,
) -> tuple[torch.Tensor, ...]:
    fp_risk, fp_validity, int_risk, int_validity, labels, validity_labels = [], [], [], [], [], []
    fp32.eval()
    int8.eval()
    frontend.eval()
    with torch.no_grad():
        for waveform, target, validity_target in loader:
            features = frontend(waveform)
            fp_r, fp_v = fp32(features)
            q_r, q_v = int8(features)
            fp_risk.append(fp_r)
            fp_validity.append(fp_v)
            int_risk.append(q_r)
            int_validity.append(q_v)
            labels.append(target)
            validity_labels.append(validity_target)
    return tuple(torch.cat(values) for values in (fp_risk, fp_validity, int_risk, int_validity, labels, validity_labels))


def serialized_size_kib(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        path = Path(handle.name)
    try:
        torch.save(model.state_dict(), path)
        return path.stat().st_size / 1024.0
    finally:
        path.unlink(missing_ok=True)


def run_one(run_dir: Path, output_dir: Path, backend: str, batch_size: int, calibration_samples: int) -> dict[str, object]:
    source, checkpoint, train_args = load_source(run_dir / "deployed_state_dict.pt")
    seed = int(train_args.seed)
    index = Path(train_args.index)
    if not index.is_absolute():
        index = PROJECT_ROOT / index
    val_samples = read_samples(index, "val", None)
    test_samples = read_samples(index, "test", None)

    fp32 = DualHeadFeatureModel(source).eval()
    quant_base = DualHeadFeatureModel(source).eval()
    fuse_encoder(quant_base)
    quant_base.qconfig = get_default_qconfig(backend)
    prepared = prepare(quant_base, inplace=False)
    calibrate(prepared, source.frontend, val_samples, train_args, seed, calibration_samples)
    int8 = convert(prepared, inplace=False).eval()

    calibration = checkpoint["calibration"]
    risk_temperature = float(calibration["risk_temperature"])
    validity_temperature = float(calibration["validity_temperature"])
    threshold = float(calibration["risk_threshold"])
    conditions = []
    example_shape = None
    for condition, severity in CONDITIONS:
        loader = make_loader(test_samples, train_args, seed, condition, severity, batch_size)
        started = time.perf_counter()
        fp_r, fp_v, q_r, q_v, labels, validity_labels = collect_pair(source.frontend, fp32, int8, loader)
        elapsed = time.perf_counter() - started
        if example_shape is None:
            first_loader = make_loader(test_samples[:1], train_args, seed, condition, severity, 1)
            waveform, _, _ = next(iter(first_loader))
            example_shape = list(source.frontend(waveform).shape)
        fp_metrics = condition_metrics(fp_r, fp_v, labels, validity_labels, risk_temperature, validity_temperature, threshold)
        q_metrics = condition_metrics(q_r, q_v, labels, validity_labels, risk_temperature, validity_temperature, threshold)
        fp_risk_probs = sigmoid(fp_r, risk_temperature)
        q_risk_probs = sigmoid(q_r, risk_temperature)
        fp_validity_pred = (fp_v / validity_temperature).argmax(dim=1)
        q_validity_pred = (q_v / validity_temperature).argmax(dim=1)
        conditions.append(
            {
                "condition": condition,
                "severity": severity,
                "target_validity": int(validity_labels[0].item()),
                "fp32": fp_metrics,
                "int8": q_metrics,
                "parity": {
                    "risk_probability_mae": float(torch.mean(torch.abs(fp_risk_probs - q_risk_probs))),
                    "risk_decision_agreement": float(torch.mean(((fp_risk_probs >= threshold) == (q_risk_probs >= threshold)).float())),
                    "validity_agreement": float(torch.mean((fp_validity_pred == q_validity_pred).float())),
                    "invalid_gate_agreement": float(torch.mean(((fp_validity_pred == 2) == (q_validity_pred == 2)).float())),
                },
                "paired_eval_seconds": elapsed,
            }
        )

    run_output = output_dir / run_dir.name
    run_output.mkdir(parents=True, exist_ok=True)
    torch.save(int8.state_dict(), run_output / "int8_feature_model_state_dict.pt")
    result = {
        "run_id": run_dir.name,
        "seed": seed,
        "backend": backend,
        "frontend": {
            "location": "external FP32 reference; embedded implementation must reproduce this contract",
            "sample_rate": int(train_args.sample_rate),
            "duration_seconds": float(train_args.duration),
            "n_fft": int(train_args.n_fft),
            "hop_length": int(train_args.hop_length),
            "n_mels": int(train_args.n_mels),
            "feature_shape": example_shape,
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in source.parameters()),
            "fp32_serialized_kib": serialized_size_kib(fp32),
            "int8_serialized_kib": serialized_size_kib(int8),
        },
        "calibration": calibration,
        "conditions": conditions,
        "claims_boundary": "PyTorch CPU PTQ parity, not ESP32-S3 latency, SRAM, Flash, or energy measurement.",
    }
    (run_output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("fbgemm", "qnnpack"), default="fbgemm")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calibration-samples", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.quantized.engine = args.backend
    results = [run_one(path, args.output_dir, args.backend, args.batch_size, args.calibration_samples) for path in args.run_dirs]

    clean_deltas = []
    invalid_recall_deltas = []
    severe_gate_agreements = []
    rows = []
    for result in results:
        for item in result["conditions"]:
            if item["condition"] == "clean":
                clean_deltas.append(float(item["int8"]["risk"]["f1_positive"]) - float(item["fp32"]["risk"]["f1_positive"]))
            if (item["condition"], int(item["severity"])) in SEVERE:
                invalid_recall_deltas.append(float(item["int8"]["validity"]["invalid_recall"]) - float(item["fp32"]["validity"]["invalid_recall"]))
                severe_gate_agreements.append(float(item["parity"]["invalid_gate_agreement"]))
            rows.append(
                {
                    "seed": result["seed"],
                    "condition": item["condition"],
                    "severity": item["severity"],
                    "fp32_f1": item["fp32"]["risk"]["f1_positive"],
                    "int8_f1": item["int8"]["risk"]["f1_positive"],
                    "fp32_invalid_recall": item["fp32"]["validity"]["invalid_recall"],
                    "int8_invalid_recall": item["int8"]["validity"]["invalid_recall"],
                    "risk_probability_mae": item["parity"]["risk_probability_mae"],
                    "risk_decision_agreement": item["parity"]["risk_decision_agreement"],
                    "invalid_gate_agreement": item["parity"]["invalid_gate_agreement"],
                }
            )
    qat_required = min(clean_deltas) < -0.015 or min(invalid_recall_deltas) < -0.05 or min(severe_gate_agreements) < 0.97
    summary = {
        "runs": results,
        "decision": {
            "qat_required": qat_required,
            "criteria": {
                "minimum_clean_f1_delta": min(clean_deltas),
                "minimum_severe_invalid_recall_delta": min(invalid_recall_deltas),
                "minimum_severe_invalid_gate_agreement": min(severe_gate_agreements),
            },
            "rule": "Run QAT only if clean F1 falls by >0.015, severe invalid recall by >0.05, or severe invalid-gate agreement is <0.97.",
        },
    }
    (args.output_dir / "ptq_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    clean_rows = [row for row in rows if row["condition"] == "clean"]
    severe_rows = [row for row in rows if (row["condition"], int(row["severity"])) in SEVERE]
    lines = [
        "# ASN Signal-Validity PTQ INT8 Parity",
        "",
        "Static INT8 PTQ was applied to the shared CNN encoder and both output heads. The log-mel frontend remained outside the quantized graph.",
        "",
        "## Clean parity",
        "",
        "| Seed | FP32 F1 | INT8 F1 | Risk decision agreement | Invalid-gate agreement |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in clean_rows:
        lines.append(f"| {row['seed']} | {row['fp32_f1']:.4f} | {row['int8_f1']:.4f} | {row['risk_decision_agreement']:.4f} | {row['invalid_gate_agreement']:.4f} |")
    lines.extend(["", "## Severe-condition parity", "", "| Seed | Condition | FP32 invalid recall | INT8 invalid recall | Gate agreement |", "| ---: | --- | ---: | ---: | ---: |"])
    for row in severe_rows:
        lines.append(f"| {row['seed']} | {row['condition']}-s{row['severity']} | {row['fp32_invalid_recall']:.4f} | {row['int8_invalid_recall']:.4f} | {row['invalid_gate_agreement']:.4f} |")
    lines.extend(
        [
            "",
            "## Compression decision",
            "",
            f"QAT required under the locked rule: **{'yes' if qat_required else 'no'}**.",
            "",
            "These measurements establish software FP32/INT8 consistency only. MCU memory, latency, and energy remain hardware measurements.",
            "",
        ]
    )
    (args.output_dir / "PTQ_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
