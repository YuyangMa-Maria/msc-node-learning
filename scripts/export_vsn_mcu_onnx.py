"""Export a deployment-compatible VSN student and parity vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


def select_balanced_samples(samples: list[Any], per_class: int) -> list[Any]:
    selected: list[Any] = []
    for label in (0, 1):
        label_samples = [sample for sample in samples if sample.label == label]
        selected.extend(label_samples[:per_class])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the final grouped VSN student to ONNX and verify numerical parity."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Final VSN FP32 checkpoint. Defaults to the grouped no-class-weight student.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Deployment export directory.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--samples-per-class", type=int, default=8)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = project_root / "scripts"
    sys.path.insert(0, str(scripts_dir))

    from train_vsn_binary import build_transforms, read_samples
    from train_vsn_student_baseline import VsnStudentDwCnn

    checkpoint_path = args.checkpoint or (
        project_root
        / "outputs"
        / "conference_grouped_vsn_student_nw0"
        / "vsn_student_dwcnn_s128_w1_scratch"
        / "best.pt"
    )
    output_dir = args.output_dir or (
        project_root / "outputs" / "hardware_deployment" / "vsn_esp32s3" / "export"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = vars(checkpoint_args)

    image_size = int(checkpoint.get("image_size", 128))
    width = float(checkpoint_args.get("width", 1.0))
    dropout = float(checkpoint_args.get("dropout", 0.1))

    model = VsnStudentDwCnn(width=width, dropout=dropout)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 15017:
        raise RuntimeError(f"Unexpected parameter count: {parameter_count}, expected 15017")
    if image_size != 128:
        raise RuntimeError(f"Unexpected input size: {image_size}, expected 128")

    onnx_path = output_dir / "vsn_student_dwcnn_s128_fp32.onnx"
    example = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            str(onnx_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["logit"],
            dynamic_axes=None,
            dynamo=False,
        )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, str(onnx_path))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.name != "image" or list(input_meta.shape) != [1, 3, 128, 128]:
        raise RuntimeError(f"Unexpected ONNX input contract: {input_meta.name} {input_meta.shape}")

    random_generator = np.random.default_rng(2026)
    random_differences: list[float] = []
    for _ in range(8):
        array = random_generator.standard_normal((1, 3, 128, 128)).astype(np.float32)
        with torch.no_grad():
            torch_output = model(torch.from_numpy(array)).cpu().numpy()
        onnx_output = session.run([output_meta.name], {input_meta.name: array})[0]
        random_differences.append(float(np.max(np.abs(torch_output - onnx_output))))

    index_value = checkpoint_args.get(
        "index", project_root / "experiments" / "conference_grouped" / "vsn_binary_index_grouped.csv"
    )
    index_path = Path(index_value)
    if not index_path.is_absolute():
        index_path = project_root / index_path
    sources_value = checkpoint_args.get("sources")
    sources = set(sources_value) if sources_value else None
    test_samples = read_samples(index_path, "test", sources)
    selected_samples = select_balanced_samples(test_samples, args.samples_per_class)
    _, eval_transform = build_transforms(image_size)

    real_differences: list[float] = []
    real_records: list[dict[str, Any]] = []
    test_vector_written = False
    for sample in selected_samples:
        image = Image.open(sample.path).convert("RGB")
        tensor = eval_transform(image).unsqueeze(0)
        array = tensor.numpy().astype(np.float32, copy=False)
        with torch.no_grad():
            torch_logit = float(model(tensor).item())
        onnx_logit = float(session.run([output_meta.name], {input_meta.name: array})[0].reshape(-1)[0])
        difference = abs(torch_logit - onnx_logit)
        real_differences.append(difference)
        real_records.append(
            {
                "path": str(sample.path),
                "label": int(sample.label),
                "source": sample.source,
                "torch_logit": torch_logit,
                "onnx_logit": onnx_logit,
                "torch_risk_score": sigmoid(torch_logit),
                "onnx_risk_score": sigmoid(onnx_logit),
                "absolute_logit_difference": difference,
            }
        )

        if not test_vector_written:
            resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
            rgb = np.asarray(resized, dtype=np.uint8)
            manual = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None, ...]
            preprocessing_difference = float(np.max(np.abs(manual - array)))
            if preprocessing_difference > 1e-6:
                raise RuntimeError(
                    f"Manual preprocessing does not match torchvision: {preprocessing_difference}"
                )
            np.save(output_dir / "test_vector_rgb128_uint8.npy", rgb)
            np.save(output_dir / "test_vector_nchw_fp32.npy", array)
            (output_dir / "test_vector_expected.json").write_text(
                json.dumps(
                    {
                        **real_records[-1],
                        "rgb_shape": list(rgb.shape),
                        "tensor_shape": list(array.shape),
                        "preprocessing_max_abs_difference": preprocessing_difference,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            test_vector_written = True

    if not selected_samples:
        raise RuntimeError("No grouped test samples were available for parity validation")

    operator_counts = Counter(node.op_type for node in onnx_model.graph.node)
    max_random_difference = max(random_differences)
    max_real_difference = max(real_differences)
    tolerance = 1e-5
    parity_passed = max(max_random_difference, max_real_difference) <= tolerance

    manifest = {
        "stage": "VSN MCU intermediate export",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_args": as_jsonable(checkpoint_args),
        "model": {
            "name": "VsnStudentDwCnn",
            "parameters": parameter_count,
            "width": width,
            "dropout_training_only": dropout,
        },
        "input_contract": {
            "name": input_meta.name,
            "shape": list(input_meta.shape),
            "layout": "NCHW",
            "dtype": "float32",
            "colour_order": "RGB",
            "resize": "direct 128x128 bilinear",
            "value_scale": "uint8 / 255.0",
            "normalisation_mean": MEAN.tolist(),
            "normalisation_std": STD.tolist(),
        },
        "output_contract": {
            "name": output_meta.name,
            "shape": list(output_meta.shape),
            "semantic": "binary visual damage logit",
            "risk_score": "sigmoid(logit)",
            "default_decision_threshold": 0.5,
        },
        "onnx": {
            "path": str(onnx_path),
            "sha256": sha256(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "opset": args.opset,
            "operators": dict(sorted(operator_counts.items())),
        },
        "parity": {
            "tolerance": tolerance,
            "passed": parity_passed,
            "random_vectors": len(random_differences),
            "real_grouped_test_images": len(real_differences),
            "random_max_abs_logit_difference": max_random_difference,
            "real_max_abs_logit_difference": max_real_difference,
            "real_mean_abs_logit_difference": float(np.mean(real_differences)),
        },
        "real_sample_records": real_records,
        "caution": [
            "This is an FP32 ONNX intermediate, not the deployable ESP32-S3 INT8 model.",
            "The existing PyTorch fbgemm PTQ artifact is not reused as an MCU binary.",
            "Board-side resize, colour order, normalisation, quantisation and output parity remain to be verified.",
        ],
    }
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_lines = [
        "# VSN ESP32-S3 Intermediate Export",
        "",
        "## Result",
        "",
        f"- ONNX checker: passed",
        f"- PyTorch/ONNX parity: {'passed' if parity_passed else 'failed'}",
        f"- Parameters: {parameter_count:,}",
        f"- ONNX size: {onnx_path.stat().st_size / 1024:.2f} KiB",
        f"- Random-vector maximum absolute logit difference: {max_random_difference:.3e}",
        f"- Real-image maximum absolute logit difference: {max_real_difference:.3e}",
        f"- Real grouped-test images checked: {len(real_differences)}",
        "",
        "## Interface",
        "",
        "- Input: float32 NCHW `[1, 3, 128, 128]`, RGB.",
        "- Preprocessing: bilinear resize, divide by 255, ImageNet mean/std normalisation.",
        "- Output: one logit; `risk_score = sigmoid(logit)`.",
        "",
        "## Next Gate",
        "",
        "- Convert and calibrate specifically for ESP32-S3 using an MCU-supported runtime.",
        "- Re-run grouped-test accuracy and risk-score calibration after target quantisation.",
        "- Verify the supplied test vector on the board before live-camera inference.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(manifest["parity"], indent=2))
    print(onnx_path)
    print(manifest_path)
    if not parity_passed:
        raise SystemExit("ONNX export parity exceeded tolerance")


if __name__ == "__main__":
    main()
