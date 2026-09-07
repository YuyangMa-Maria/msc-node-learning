"""Export trained shared risk heads as deterministic BLE transport payloads.

Tensor order is part of the wire contract. The generated record includes format,
length, CRC and a golden output so firmware can reject a byte-valid payload that
does not implement the expected computation.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import torch


TENSOR_NAMES = ("0.weight", "0.bias", "3.weight", "3.bias")
PARAMETER_COUNT = 2113


def load_head(path: Path, variant: str | None) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if variant is None:
        state = {
            key.removeprefix("shared_head."): value
            for key, value in checkpoint["model_state"].items()
            if key.startswith("shared_head.")
        }
    else:
        state = checkpoint["final_shared_head_states"][variant]
    missing = set(TENSOR_NAMES) - set(state)
    if missing:
        raise KeyError(f"Missing shared-head tensors: {sorted(missing)}")
    return {name: state[name].detach().cpu().float().contiguous() for name in TENSOR_NAMES}


def serialise_fp32(state: dict[str, torch.Tensor]) -> bytes:
    """Pack tensors in the model-exchange protocol's canonical order."""
    return b"".join(state[name].numpy().astype("<f4", copy=False).tobytes() for name in TENSOR_NAMES)


def serialise_int8(state: dict[str, torch.Tensor]) -> bytes:
    """Quantise each tensor symmetrically and append its scale before values."""
    scales: list[float] = []
    quantised: list[bytes] = []
    for name in TENSOR_NAMES:
        values = state[name].numpy()
        max_abs = float(np.max(np.abs(values)))
        scale = max(max_abs / 127.0, np.finfo(np.float32).tiny)
        qvalues = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
        scales.append(scale)
        quantised.append(qvalues.tobytes())
    return struct.pack("<4f", *scales) + b"".join(quantised)


def evaluate_fp32(payload: bytes, embedding: np.ndarray) -> float:
    values = np.frombuffer(payload, dtype="<f4")
    if values.size != PARAMETER_COUNT:
        raise ValueError(f"Expected {PARAMETER_COUNT} FP32 parameters, got {values.size}")
    w1 = values[:2048].reshape(32, 64)
    b1 = values[2048:2080]
    w2 = values[2080:2112]
    b2 = values[2112]
    hidden = np.maximum(w1 @ embedding + b1, 0.0)
    return float(w2 @ hidden + b2)


def evaluate_int8(payload: bytes, embedding: np.ndarray) -> float:
    scales = np.frombuffer(payload[:16], dtype="<f4")
    qvalues = np.frombuffer(payload[16:], dtype=np.int8)
    if qvalues.size != PARAMETER_COUNT:
        raise ValueError(f"Expected {PARAMETER_COUNT} INT8 parameters, got {qvalues.size}")
    w1 = qvalues[:2048].astype(np.float32).reshape(32, 64) * scales[0]
    b1 = qvalues[2048:2080].astype(np.float32) * scales[1]
    w2 = qvalues[2080:2112].astype(np.float32) * scales[2]
    b2 = float(qvalues[2112]) * float(scales[3])
    hidden = np.maximum(w1 @ embedding + b1, 0.0)
    return float(w2 @ hidden + b2)


def format_bytes(name: str, payload: bytes) -> str:
    rows = []
    for offset in range(0, len(payload), 16):
        row = ", ".join(f"0x{value:02x}" for value in payload[offset : offset + 16])
        rows.append(f"    {row},")
    return f"alignas(4) inline constexpr uint8_t {name}[] = {{\n" + "\n".join(rows) + "\n};\n"


def format_floats(name: str, values: np.ndarray) -> str:
    rows = []
    for offset in range(0, len(values), 8):
        row = ", ".join(f"{float(value):.9e}f" for value in values[offset : offset + 8])
        rows.append(f"    {row},")
    return f"inline constexpr float {name}[] = {{\n" + "\n".join(rows) + "\n};\n"


def package_record(prefix: str, format_name: str, payload: bytes, expected: float) -> str:
    fmt = "kFp32" if format_name == "fp32" else "kInt8Symmetric"
    version = "0x0201" if prefix == "kVsnBase" else "0x0202"
    return (
        f"inline constexpr SharedHeadPackage {prefix}{format_name.upper()} = {{\n"
        f"    {prefix}{format_name.upper()}Bytes, sizeof({prefix}{format_name.upper()}Bytes),\n"
        f"    0x{zlib.crc32(payload) & 0xFFFFFFFF:08x}u, {version}, "
        f"SharedHeadFormat::{fmt}, {expected:.9g}f\n"
        "};\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--federated-checkpoint", type=Path, required=True)
    parser.add_argument("--federated-variant", default="continual_pretrained_sample_weighted")
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    base = load_head(args.base_checkpoint, None)
    federated = load_head(args.federated_checkpoint, args.federated_variant)
    embedding = np.linspace(-1.0, 1.0, 64, dtype=np.float32)

    packages: dict[str, tuple[bytes, float]] = {}
    for source_name, state in (("vsn_base", base), ("asn_federated", federated)):
        fp32 = serialise_fp32(state)
        int8 = serialise_int8(state)
        packages[f"{source_name}_fp32"] = (fp32, evaluate_fp32(fp32, embedding))
        packages[f"{source_name}_int8"] = (int8, evaluate_int8(int8, embedding))

    if len(packages["vsn_base_fp32"][0]) != PARAMETER_COUNT * 4:
        raise AssertionError("Unexpected FP32 payload size")
    if len(packages["vsn_base_int8"][0]) != PARAMETER_COUNT + 16:
        raise AssertionError("Unexpected INT8 payload size")

    preamble = """// Generated by tools/export_shared_head_payloads.py. Do not edit manually.
#pragma once

#include <cstddef>
#include <cstdint>

namespace model_exchange {

enum class SharedHeadFormat : uint8_t {
    kFp32 = 1,
    kInt8Symmetric = 2,
};

struct SharedHeadPackage {
    const uint8_t *data;
    size_t size;
    uint32_t crc32;
    uint16_t model_version;
    SharedHeadFormat format;
    float expected_golden_output;
};

inline constexpr size_t kSharedHeadParameterCount = 2113;
inline constexpr size_t kSharedHeadFp32Bytes = 8452;
inline constexpr size_t kSharedHeadInt8Bytes = 2129;
"""
    sections = [preamble, format_floats("kGoldenEmbedding", embedding)]
    name_map = {
        "vsn_base_fp32": "kVsnBaseFP32Bytes",
        "vsn_base_int8": "kVsnBaseINT8Bytes",
        "asn_federated_fp32": "kAsnFederatedFP32Bytes",
        "asn_federated_int8": "kAsnFederatedINT8Bytes",
    }
    for key, symbol in name_map.items():
        sections.append(format_bytes(symbol, packages[key][0]))
    sections.extend(
        [
            package_record("kVsnBase", "fp32", *packages["vsn_base_fp32"]),
            package_record("kVsnBase", "int8", *packages["vsn_base_int8"]),
            package_record("kAsnFederated", "fp32", *packages["asn_federated_fp32"]),
            package_record("kAsnFederated", "int8", *packages["asn_federated_int8"]),
            "}  // namespace model_exchange\n",
        ]
    )
    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text("\n".join(sections), encoding="ascii")

    manifest = {
        "source": {
            "vsn_base": str(args.base_checkpoint),
            "asn_federated": str(args.federated_checkpoint),
            "federated_variant": args.federated_variant,
        },
        "architecture": "Linear(64,32)-ReLU-Linear(32,1)",
        "parameter_count": PARAMETER_COUNT,
        "golden_embedding": embedding.tolist(),
        "packages": {
            key: {
                "bytes": len(payload),
                "crc32": f"0x{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
                "expected_golden_output": expected,
            }
            for key, (payload, expected) in packages.items()
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
