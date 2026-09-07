"""Measure the contribution of explicit ASN validity gating to risk fusion.

The ablation compares the same quantised risk scores with and without removing
samples classified as invalid. This isolates the value of status communication
from any improvement in the underlying acoustic risk classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import convert, get_default_qconfig, prepare
from torch.utils.data import DataLoader

from evaluate_asn_signal_validity_ptq import DualHeadFeatureModel, calibrate, fuse_encoder, load_source
from evaluate_compressed_fusion import (
    CompressedPairDataset,
    align_vbn_scores,
    best_f1_threshold,
    build_transforms,
    confidence_from_risk,
    degrade_vbn,
    load_temperature,
    load_vbn_split_scores,
    load_vsn_int8,
    metrics,
    read_grouped_pairs,
)
from train_asn_audio import PROJECT_ROOT, read_samples
from train_vbn_orion import DEFAULT_INDEX


class QuantizedAsnWaveformModel(nn.Module):
    """Compose the FP32 frontend with the converted dual-head feature model."""

    def __init__(self, frontend: nn.Module, feature_model: nn.Module) -> None:
        super().__init__()
        self.frontend = frontend
        self.feature_model = feature_model

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.feature_model(self.frontend(waveform))


def load_asn_int8(run_dir: Path, backend: str, calibration_samples: int) -> tuple[nn.Module, dict[str, object]]:
    source, checkpoint, train_args = load_source(run_dir / "deployed_state_dict.pt")
    index = Path(train_args.index)
    if not index.is_absolute():
        index = PROJECT_ROOT / index
    val_samples = read_samples(index, "val", None)
    quant_base = DualHeadFeatureModel(source).eval()
    fuse_encoder(quant_base)
    quant_base.qconfig = get_default_qconfig(backend)
    prepared = prepare(quant_base, inplace=False)
    calibrate(prepared, source.frontend, val_samples, train_args, int(train_args.seed), calibration_samples)
    converted = convert(prepared, inplace=False).eval()
    return QuantizedAsnWaveformModel(source.frontend, converted).eval(), checkpoint


def condition_definitions(include_vbn: bool) -> list[dict[str, object]]:
    rows = [
        {"name": "clean", "image_corruption": "clean", "image_severity": 0, "audio_corruption": "clean", "audio_severity": 0, "vbn_mode": "clean", "visual_quality": 1.0, "audio_quality": 1.0, "vbn_quality": 1.0},
        {"name": "vsn_blur_asn_clean" + ("_vbn_clean" if include_vbn else ""), "image_corruption": "blur", "image_severity": 3, "audio_corruption": "clean", "audio_severity": 0, "vbn_mode": "clean", "visual_quality": 0.20, "audio_quality": 1.0, "vbn_quality": 1.0},
        {"name": "asn_invalid_vsn_clean" + ("_vbn_clean" if include_vbn else ""), "image_corruption": "clean", "image_severity": 0, "audio_corruption": "snr_noise", "audio_severity": 3, "vbn_mode": "clean", "visual_quality": 1.0, "audio_quality": 1.0, "vbn_quality": 1.0},
        {"name": "vsn_degraded_asn_invalid" + ("_vbn_clean" if include_vbn else ""), "image_corruption": "blur", "image_severity": 3, "audio_corruption": "snr_noise", "audio_severity": 3, "vbn_mode": "clean", "visual_quality": 0.20, "audio_quality": 1.0, "vbn_quality": 1.0},
    ]
    if include_vbn:
        rows.extend(
            [
                {"name": "vbn_uncertain_vsn_asn_clean", "image_corruption": "clean", "image_severity": 0, "audio_corruption": "clean", "audio_severity": 0, "vbn_mode": "uncertain", "visual_quality": 1.0, "audio_quality": 1.0, "vbn_quality": 0.35},
                {"name": "all_modalities_degraded_or_invalid", "image_corruption": "blur", "image_severity": 3, "audio_corruption": "snr_noise", "audio_severity": 3, "vbn_mode": "uncertain", "visual_quality": 0.20, "audio_quality": 1.0, "vbn_quality": 0.35},
            ]
        )
    return rows


def collect(
    samples,
    condition: dict[str, object],
    vsn: nn.Module,
    vsn_temperature: float,
    asn: nn.Module,
    risk_temperature: float,
    validity_temperature: float,
    image_size: int,
    audio_samples: int,
    batch_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    _, transform = build_transforms(image_size)
    dataset = CompressedPairDataset(
        samples,
        transform,
        audio_samples,
        str(condition["image_corruption"]),
        int(condition["image_severity"]),
        str(condition["audio_corruption"]),
        int(condition["audio_severity"]),
        seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    vsn_scores, asn_scores, accepted, labels = [], [], [], []
    with torch.no_grad():
        for images, waveforms, target in loader:
            visual_logits = vsn(images).flatten()
            risk_logits, validity_logits = asn(waveforms)
            vsn_scores.append(torch.sigmoid(visual_logits / vsn_temperature).cpu())
            asn_scores.append(torch.sigmoid(risk_logits / risk_temperature).cpu())
            accepted.append(((validity_logits / validity_temperature).argmax(dim=1) != 2).cpu())
            labels.append(target.cpu())
    return {"vsn": torch.cat(vsn_scores), "asn": torch.cat(asn_scores), "asn_accepted": torch.cat(accepted), "labels": torch.cat(labels)}


def fuse(scores: dict[str, torch.Tensor], condition: dict[str, object], include_vbn: bool) -> dict[str, torch.Tensor]:
    """Compare legacy quality weighting with a hard invalid-status gate."""
    visual = scores["vsn"]
    audio = scores["asn"]
    accepted = scores["asn_accepted"].float()
    visual_weight = confidence_from_risk(visual) * float(condition["visual_quality"])
    audio_weight = confidence_from_risk(audio) * float(condition["audio_quality"])
    if not include_vbn:
        legacy = (visual_weight * visual + audio_weight * audio) / (visual_weight + audio_weight).clamp_min(1e-7)
        # An invalid ASN contributes exactly zero weight; a merely degraded
        # node remains available for softer quality weighting.
        gated_audio_weight = audio_weight * accepted
        gated = (visual_weight * visual + gated_audio_weight * audio) / (visual_weight + gated_audio_weight).clamp_min(1e-7)
        return {"vsn_only": visual, "asn_only": audio, "legacy_quality_fusion": legacy, "status_gated_fusion": gated}

    vibration = scores["vbn"]
    vibration_weight = confidence_from_risk(vibration) * float(condition["vbn_quality"])
    legacy = (visual_weight * visual + audio_weight * audio + vibration_weight * vibration) / (visual_weight + audio_weight + vibration_weight).clamp_min(1e-7)
    gated_audio_weight = audio_weight * accepted
    gated = (visual_weight * visual + gated_audio_weight * audio + vibration_weight * vibration) / (visual_weight + gated_audio_weight + vibration_weight).clamp_min(1e-7)
    return {"vsn_only": visual, "asn_only": audio, "vbn_only": vibration, "legacy_quality_fusion_3": legacy, "status_gated_fusion_3": gated}


def evaluate(
    val_samples,
    test_samples,
    vsn,
    vsn_temperature: float,
    asn,
    risk_temperature: float,
    validity_temperature: float,
    image_size: int,
    audio_samples: int,
    batch_size: int,
    seed: int,
    include_vbn: bool,
    vbn_pool,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Freeze clean-validation thresholds before applying degradation tests."""
    clean = condition_definitions(include_vbn)[0]
    val_scores = collect(val_samples, clean, vsn, vsn_temperature, asn, risk_temperature, validity_temperature, image_size, audio_samples, batch_size, seed)
    if include_vbn:
        val_scores["vbn"] = align_vbn_scores(val_scores["labels"], vbn_pool["val"], seed)
    val_methods = fuse(val_scores, clean, include_vbn)
    thresholds = {name: best_f1_threshold(score, val_scores["labels"]) for name, score in val_methods.items()}

    rows = []
    for condition in condition_definitions(include_vbn):
        scores = collect(test_samples, condition, vsn, vsn_temperature, asn, risk_temperature, validity_temperature, image_size, audio_samples, batch_size, seed)
        if include_vbn:
            aligned = align_vbn_scores(scores["labels"], vbn_pool["test"], seed)
            scores["vbn"] = degrade_vbn(aligned, str(condition["vbn_mode"]))
        for method, values in fuse(scores, condition, include_vbn).items():
            row = metrics(values, scores["labels"], thresholds[method])
            rows.append(
                {
                    "condition": condition["name"],
                    "method": method,
                    "sample_count": len(test_samples),
                    "asn_invalid_rate": float((~scores["asn_accepted"]).float().mean()),
                    **row,
                }
            )
    return rows, thresholds


def write_report(output_dir: Path, rows: list[dict[str, object]], thresholds: dict[str, float], metadata: dict[str, object], include_vbn: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps({"metadata": metadata, "thresholds": thresholds, "results": rows}, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    fusion_names = ("legacy_quality_fusion_3", "status_gated_fusion_3") if include_vbn else ("legacy_quality_fusion", "status_gated_fusion")
    lines = [
        f"# ASN Status-Gated {'Three-Node' if include_vbn else 'Two-Node'} Compressed Fusion",
        "",
        "Thresholds were fitted on the clean validation split and fixed before test-condition evaluation.",
        "",
        "| Condition | ASN invalid rate | Legacy F1 | Status-gated F1 | Legacy recall | Status-gated recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in dict.fromkeys(str(row["condition"]) for row in rows):
        legacy = next(row for row in rows if row["condition"] == condition and row["method"] == fusion_names[0])
        gated = next(row for row in rows if row["condition"] == condition and row["method"] == fusion_names[1])
        lines.append(
            f"| {condition} | {legacy['asn_invalid_rate']:.4f} | {legacy['f1_positive']:.4f} | {gated['f1_positive']:.4f} | "
            f"{legacy['recall_positive']:.4f} | {gated['recall_positive']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The status gate removes ASN risk evidence only when the validity head predicts `invalid`. The `degraded` state remains diagnostic and is not used as a hard gate.",
            "",
            "VBN results are label-aligned proxy stress tests rather than synchronous tri-modal observations." if include_vbn else "The paired VSN/ASN data use the grouped split; this is still a controlled corruption study rather than field sensing.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--vsn-run-dir", type=Path, required=True)
    parser.add_argument("--vsn-calibration-json", type=Path, required=True)
    parser.add_argument("--asn-run-dir", type=Path, required=True)
    parser.add_argument("--vbn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vbn_orion")
    parser.add_argument("--vbn-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", default="fbgemm", choices=("fbgemm", "qnnpack"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()
    torch.backends.quantized.engine = args.backend

    pairs = read_grouped_pairs(args.pair_index)
    val_samples = [sample for sample in pairs if sample.split == "val"]
    test_samples = [sample for sample in pairs if sample.split == "test"]
    vsn_checkpoint = torch.load(args.vsn_run_dir / "best.pt", map_location="cpu")
    image_size = int(vsn_checkpoint.get("image_size", 128))
    audio_samples = 80_000
    _, transform = build_transforms(image_size)
    calibration_loader = DataLoader(CompressedPairDataset(val_samples, transform, audio_samples, "clean", 0, "clean", 0, args.seed), batch_size=args.batch_size, shuffle=False, num_workers=0)
    vsn = load_vsn_int8(args.vsn_run_dir, calibration_loader, args.backend, None)
    vsn_temperature = load_temperature(args.vsn_calibration_json)
    asn, asn_checkpoint = load_asn_int8(args.asn_run_dir, args.backend, args.calibration_samples)
    risk_temperature = float(asn_checkpoint["calibration"]["risk_temperature"])
    validity_temperature = float(asn_checkpoint["calibration"]["validity_temperature"])
    metadata = {
        "pair_index": str(args.pair_index),
        "vsn_run_dir": str(args.vsn_run_dir),
        "asn_run_dir": str(args.asn_run_dir),
        "vsn_temperature": vsn_temperature,
        "asn_risk_temperature": risk_temperature,
        "asn_validity_temperature": validity_temperature,
        "backend": args.backend,
        "seed": args.seed,
        "claims_boundary": "Software INT8 fusion stress test; no synchronized VBN or physical deployment claim.",
    }
    two_rows, two_thresholds = evaluate(val_samples, test_samples, vsn, vsn_temperature, asn, risk_temperature, validity_temperature, image_size, audio_samples, args.batch_size, args.seed, False, None)
    write_report(args.output_dir / "vsn_asn", two_rows, two_thresholds, metadata, False)
    vbn_pool = load_vbn_split_scores(args.vbn_run_dir, args.vbn_index, 200_000, 5_000_000.0)
    three_rows, three_thresholds = evaluate(val_samples, test_samples, vsn, vsn_temperature, asn, risk_temperature, validity_temperature, image_size, audio_samples, args.batch_size, args.seed, True, vbn_pool)
    write_report(args.output_dir / "vsn_asn_vbn_proxy", three_rows, three_thresholds, metadata, True)


if __name__ == "__main__":
    main()
