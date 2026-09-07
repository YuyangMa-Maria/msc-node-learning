"""Estimate parameter, activation and working-memory costs for node models.

These host estimates are used for candidate rejection before conversion. They
do not replace measured Tensor Arena, PSRAM, Flash or latency values from the
physical firmware.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_asn_audio import AsnModel
from train_vsn_binary import build_model


SRAM_LIMIT_BYTES = 2 * 1024 * 1024


def as_namespace(data: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def bytes_to_mb(value: float) -> float:
    return float(value) / (1024.0 * 1024.0)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def state_dict_size_bytes(model: nn.Module) -> int:
    total = 0
    for value in model.state_dict().values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


def module_output_numel(output: object) -> int:
    if isinstance(output, torch.Tensor):
        return output.numel()
    if isinstance(output, (list, tuple)):
        return sum(module_output_numel(item) for item in output)
    if isinstance(output, dict):
        return sum(module_output_numel(item) for item in output.values())
    return 0


def is_leaf(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def activation_audit(model: nn.Module, dummy_input: torch.Tensor) -> dict[str, int]:
    """Record leaf-module outputs as a conservative activation-size proxy."""
    model.eval()
    outputs: list[tuple[str, int]] = []
    handles = []

    def hook(name: str):
        def _hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            numel = module_output_numel(output)
            if numel > 0:
                outputs.append((name, numel))

        return _hook

    for name, module in model.named_modules():
        if name and is_leaf(module):
            handles.append(module.register_forward_hook(hook(name)))
    with torch.no_grad():
        _ = model(dummy_input)
    for handle in handles:
        handle.remove()
    peak_name, peak_numel = max(outputs, key=lambda item: item[1]) if outputs else ("", 0)
    total_numel = sum(numel for _, numel in outputs)
    return {
        "leaf_output_count": len(outputs),
        "peak_activation_numel": int(peak_numel),
        "sum_activation_numel": int(total_numel),
        "peak_activation_layer": peak_name,
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


def audit_vsn(run_dir: Path, image_sizes: list[int]) -> list[dict[str, object]]:
    model, args = load_vsn(run_dir)
    params = count_parameters(model)
    state_size = state_dict_size_bytes(model)
    rows = []
    for image_size in image_sizes:
        dummy = torch.zeros(1, 3, image_size, image_size)
        activations = activation_audit(model, dummy)
        input_uint8 = 3 * image_size * image_size
        input_fp32 = input_uint8 * 4
        peak = activations["peak_activation_numel"]
        total = activations["sum_activation_numel"]
        int8_weight = params
        fp32_weight = params * 4
        sram_int8_runtime = input_uint8 + peak
        sram_fp32_runtime = input_fp32 + peak * 4
        conservative_sram_int8_runtime = input_uint8 + total
        conservative_sram_fp32_runtime = input_fp32 + total * 4
        rows.append(
            {
                "node": "VSN",
                "model": f"{args.model}_image{image_size}",
                "input_shape": f"1x3x{image_size}x{image_size}",
                "parameters": params,
                "checkpoint_mb": bytes_to_mb((run_dir / "best.pt").stat().st_size),
                "state_dict_mb": bytes_to_mb(state_size),
                "fp32_weight_mb": bytes_to_mb(fp32_weight),
                "int8_weight_est_mb": bytes_to_mb(int8_weight),
                "input_uint8_mb": bytes_to_mb(input_uint8),
                "input_fp32_mb": bytes_to_mb(input_fp32),
                "peak_activation_fp32_mb": bytes_to_mb(peak * 4),
                "peak_activation_int8_mb": bytes_to_mb(peak),
                "sum_activation_fp32_mb": bytes_to_mb(total * 4),
                "sum_activation_int8_mb": bytes_to_mb(total),
                "peak_activation_layer": activations["peak_activation_layer"],
                "estimated_runtime_sram_int8_mb": bytes_to_mb(sram_int8_runtime),
                "estimated_runtime_sram_fp32_mb": bytes_to_mb(sram_fp32_runtime),
                "conservative_runtime_sram_int8_mb": bytes_to_mb(conservative_sram_int8_runtime),
                "conservative_runtime_sram_fp32_mb": bytes_to_mb(conservative_sram_fp32_runtime),
                "strict_int8_weight_plus_runtime_mb": bytes_to_mb(int8_weight + sram_int8_runtime),
                "strict_int8_weight_plus_conservative_runtime_mb": bytes_to_mb(int8_weight + conservative_sram_int8_runtime),
                "int8_weight_under_2mb": int8_weight <= SRAM_LIMIT_BYTES,
                "runtime_sram_int8_under_2mb": sram_int8_runtime <= SRAM_LIMIT_BYTES,
                "conservative_runtime_sram_int8_under_2mb": conservative_sram_int8_runtime <= SRAM_LIMIT_BYTES,
                "strict_weight_plus_runtime_under_2mb": (int8_weight + sram_int8_runtime) <= SRAM_LIMIT_BYTES,
                "strict_weight_plus_conservative_runtime_under_2mb": (int8_weight + conservative_sram_int8_runtime) <= SRAM_LIMIT_BYTES,
                "note": "INT8 is estimated. Actual MCU arena must be measured with deployment runtime.",
            }
        )
    return rows


def audit_asn(run_dir: Path) -> list[dict[str, object]]:
    model, args = load_asn(run_dir)
    params = count_parameters(model)
    state_size = state_dict_size_bytes(model)
    target_samples = int(args.sample_rate * args.duration)
    dummy = torch.zeros(1, target_samples)
    activations = activation_audit(model, dummy)
    input_int16 = target_samples * 2
    input_fp32 = target_samples * 4
    peak = activations["peak_activation_numel"]
    total = activations["sum_activation_numel"]
    int8_weight = params
    fp32_weight = params * 4
    sram_int8_runtime = input_int16 + peak
    sram_fp32_runtime = input_fp32 + peak * 4
    conservative_sram_int8_runtime = input_int16 + total
    conservative_sram_fp32_runtime = input_fp32 + total * 4
    return [
        {
            "node": "ASN",
            "model": "tiny_logmel_cnn_10s",
            "input_shape": f"1x{target_samples}",
            "parameters": params,
            "checkpoint_mb": bytes_to_mb((run_dir / "best.pt").stat().st_size),
            "state_dict_mb": bytes_to_mb(state_size),
            "fp32_weight_mb": bytes_to_mb(fp32_weight),
            "int8_weight_est_mb": bytes_to_mb(int8_weight),
            "input_uint8_mb": "",
            "input_int16_mb": bytes_to_mb(input_int16),
            "input_fp32_mb": bytes_to_mb(input_fp32),
            "peak_activation_fp32_mb": bytes_to_mb(peak * 4),
            "peak_activation_int8_mb": bytes_to_mb(peak),
            "sum_activation_fp32_mb": bytes_to_mb(total * 4),
            "sum_activation_int8_mb": bytes_to_mb(total),
            "peak_activation_layer": activations["peak_activation_layer"],
            "estimated_runtime_sram_int8_mb": bytes_to_mb(sram_int8_runtime),
            "estimated_runtime_sram_fp32_mb": bytes_to_mb(sram_fp32_runtime),
            "conservative_runtime_sram_int8_mb": bytes_to_mb(conservative_sram_int8_runtime),
            "conservative_runtime_sram_fp32_mb": bytes_to_mb(conservative_sram_fp32_runtime),
            "strict_int8_weight_plus_runtime_mb": bytes_to_mb(int8_weight + sram_int8_runtime),
            "strict_int8_weight_plus_conservative_runtime_mb": bytes_to_mb(int8_weight + conservative_sram_int8_runtime),
            "int8_weight_under_2mb": int8_weight <= SRAM_LIMIT_BYTES,
            "runtime_sram_int8_under_2mb": sram_int8_runtime <= SRAM_LIMIT_BYTES,
            "conservative_runtime_sram_int8_under_2mb": conservative_sram_int8_runtime <= SRAM_LIMIT_BYTES,
            "strict_weight_plus_runtime_under_2mb": (int8_weight + sram_int8_runtime) <= SRAM_LIMIT_BYTES,
            "strict_weight_plus_conservative_runtime_under_2mb": (int8_weight + conservative_sram_int8_runtime) <= SRAM_LIMIT_BYTES,
            "note": "Audio input assumed int16 before frontend. Full log-mel frontend memory must be checked on target runtime.",
        }
    ]


def audit_vbn(run_dir: Path) -> list[dict[str, object]]:
    with (run_dir / "best_model.pkl").open("rb") as f:
        saved = pickle.load(f)
    feature_names = saved["feature_names"]
    feature_count = len(feature_names)
    feature_buffer_fp32 = feature_count * 4
    model_file = run_dir / "best_model.pkl"
    return [
        {
            "node": "VBN",
            "model": "logistic_regression_proxy",
            "input_shape": f"{feature_count} handcrafted features",
            "parameters": feature_count + 1,
            "checkpoint_mb": bytes_to_mb(model_file.stat().st_size),
            "state_dict_mb": "",
            "fp32_weight_mb": bytes_to_mb((feature_count + 1) * 4),
            "int8_weight_est_mb": bytes_to_mb(feature_count + 1),
            "input_uint8_mb": "",
            "input_int16_mb": "",
            "input_fp32_mb": bytes_to_mb(feature_buffer_fp32),
            "peak_activation_fp32_mb": bytes_to_mb(feature_buffer_fp32),
            "peak_activation_int8_mb": bytes_to_mb(feature_count),
            "sum_activation_fp32_mb": bytes_to_mb(feature_buffer_fp32),
            "sum_activation_int8_mb": bytes_to_mb(feature_count),
            "peak_activation_layer": "feature_vector",
            "estimated_runtime_sram_int8_mb": bytes_to_mb(feature_count * 2),
            "estimated_runtime_sram_fp32_mb": bytes_to_mb(feature_buffer_fp32 * 2),
            "conservative_runtime_sram_int8_mb": bytes_to_mb(feature_count * 2),
            "conservative_runtime_sram_fp32_mb": bytes_to_mb(feature_buffer_fp32 * 2),
            "strict_int8_weight_plus_runtime_mb": bytes_to_mb((feature_count + 1) + feature_count * 2),
            "strict_int8_weight_plus_conservative_runtime_mb": bytes_to_mb((feature_count + 1) + feature_count * 2),
            "int8_weight_under_2mb": True,
            "runtime_sram_int8_under_2mb": True,
            "conservative_runtime_sram_int8_under_2mb": True,
            "strict_weight_plus_runtime_under_2mb": True,
            "strict_weight_plus_conservative_runtime_under_2mb": True,
            "note": "VBN is an ORION AE proxy model; real accelerometer feature extraction may change buffer needs.",
        }
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# TinyML Memory Audit Report",
        "",
        "## Scope",
        "",
        "- Target constraint: SRAM below 2 MB for VSN, ASN, and VBN. MARH is excluded because it is resource-rich.",
        "- This is a feasibility audit, not final deployment profiling.",
        "- Weight memory and runtime SRAM are reported separately because TinyML devices may store weights in Flash while SRAM is used for input, activations, and arena buffers.",
        "- INT8 values are estimates from parameter count and activation tensor sizes. Actual MCU memory must later be measured with the chosen runtime.",
        "",
        "## Summary",
        "",
        "| Node | Model/Input | FP32 weights MB | INT8 weights MB | Peak-runtime SRAM INT8 MB | Conservative SRAM INT8 MB | Strict peak MB | Strict conservative MB | Peak <2MB | Conservative <2MB | Main interpretation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        interpretation = ""
        node = str(row["node"])
        if node == "VSN":
            interpretation = "INT8 weights likely fit; strict SRAM+weights depends on image size/runtime."
        elif node == "ASN":
            interpretation = "Small weights; audio/frontend buffers dominate runtime memory."
        elif node == "VBN":
            interpretation = "Easily fits current proxy feature model."
        lines.append(
            "| {node} | {model} | {fp32:.3f} | {int8:.3f} | {sram:.3f} | {cons_sram:.3f} | {strict:.3f} | {cons_strict:.3f} | {runtime_pass} | {cons_pass} | {interp} |".format(
                node=row["node"],
                model=row["model"],
                fp32=float(row["fp32_weight_mb"]),
                int8=float(row["int8_weight_est_mb"]),
                sram=float(row["estimated_runtime_sram_int8_mb"]),
                cons_sram=float(row["conservative_runtime_sram_int8_mb"]),
                strict=float(row["strict_int8_weight_plus_runtime_mb"]),
                cons_strict=float(row["strict_int8_weight_plus_conservative_runtime_mb"]),
                runtime_pass="yes" if row["runtime_sram_int8_under_2mb"] else "no",
                cons_pass="yes" if row["conservative_runtime_sram_int8_under_2mb"] else "no",
                interp=interpretation,
            )
        )
    lines.extend(
        [
            "",
            "## Detailed Notes",
            "",
            "- VSN FP32 weights exceed 2 MB, so quantization is required for TinyML storage.",
            "- Peak-runtime SRAM assumes the deployment runtime reuses activation buffers efficiently.",
            "- Conservative SRAM sums leaf-module outputs and is intentionally pessimistic; it highlights where tensor-arena measurement is needed.",
            "- VSN INT8 weight size is close to the limit but can fit if weights are stored compactly; lower image sizes reduce input and activation memory.",
            "- ASN has very small weights, but a 10-second waveform and log-mel frontend still require careful streaming or chunking on MCU, especially under conservative memory accounting.",
            "- VBN is tiny in its current proxy form.",
            "",
            "## Recommended Next Compression Steps",
            "",
            "1. Export VSN and ASN to a deployment-oriented INT8 path such as TFLite Micro, TVM, or ONNX Runtime embedded.",
            "2. Measure real tensor-arena SRAM rather than relying only on PyTorch activation estimates.",
            "3. Try VSN image-size reduction, especially 160 and 128, and evaluate accuracy impact.",
            "4. If VSN accuracy drops after reducing size or quantizing, train a student model using the current MobileNetV3 model as teacher.",
            "5. For ASN, test streaming/chunked log-mel extraction to avoid storing the full 10-second waveform in SRAM.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TinyML memory feasibility for VSN, ASN, and VBN.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "tinyml_memory_audit")
    parser.add_argument("--vsn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vsn_binary_mixed_full" / "mobilenet_v3_small_pretrained")
    parser.add_argument("--asn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "asn_audio_formal" / "tiny_logmel_cnn")
    parser.add_argument("--vbn-run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "vbn_orion")
    parser.add_argument("--image-sizes", nargs="+", type=int, default=[224, 160, 128])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(audit_vsn(args.vsn_run_dir, args.image_sizes))
    rows.extend(audit_asn(args.asn_run_dir))
    rows.extend(audit_vbn(args.vbn_run_dir))
    write_csv(args.output_dir / "tinyml_memory_audit.csv", rows)
    (args.output_dir / "tinyml_memory_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_report(args.output_dir / "REPORT.md", rows)
    print(f"Wrote {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
