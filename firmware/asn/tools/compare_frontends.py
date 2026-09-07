"""Compare desktop and ESP-DL log-Mel frontends on the same audio vectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
N_FFT = 1_024
HOP_LENGTH = 320
N_MELS = 64
TOP_DB = 80.0
INPUT_EXPONENT = -5


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log1p(np.asarray(frequency_hz) / 700.0)


def build_espdl_mel_bank() -> np.ndarray:
    frequencies = np.arange(N_FFT // 2 + 1, dtype=np.float64) * (SAMPLE_RATE / N_FFT)
    frequency_mels = hz_to_mel(frequencies)
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2)
    bank = np.zeros((N_MELS, frequencies.size), dtype=np.float64)
    for index in range(N_MELS):
        lower = (frequency_mels - mel_points[index]) / (mel_points[index + 1] - mel_points[index])
        upper = (mel_points[index + 2] - frequency_mels) / (mel_points[index + 2] - mel_points[index + 1])
        bank[index] = np.maximum(np.minimum(lower, upper), 0.0)
    bank[:, 0] = 0.0
    return bank


def espdl_compatible_logmel(pcm16: np.ndarray) -> np.ndarray:
    waveform = pcm16.astype(np.float64) / 32768.0
    padded = np.pad(waveform, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    if frames.shape[0] != 251:
        raise RuntimeError(f"Expected 251 frames, received {frames.shape[0]}")

    periodic_hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)
    spectrum = np.fft.rfft(frames * periodic_hann, n=N_FFT, axis=1)
    power = spectrum.real**2 + spectrum.imag**2
    mel_power = power @ build_espdl_mel_bank().T

    natural_log = np.log(np.maximum(mel_power, 1e-10))
    natural_log = np.maximum(natural_log, natural_log.max() - TOP_DB * math.log(10.0) / 10.0)
    normalised = (natural_log - natural_log.mean()) / max(natural_log.std(ddof=1), 1e-5)
    return normalised.T.astype(np.float32)


def quantise_int8(values: np.ndarray, exponent: int) -> np.ndarray:
    scale = math.ldexp(1.0, exponent)
    return np.clip(np.rint(values / scale), -128, 127).astype(np.int8)


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    reference_flat = reference.reshape(-1).astype(np.float64)
    candidate_flat = candidate.reshape(-1).astype(np.float64)
    correlation = float(np.corrcoef(reference_flat, candidate_flat)[0, 1])
    reference_int8 = quantise_int8(reference, INPUT_EXPONENT)
    candidate_int8 = quantise_int8(candidate, INPUT_EXPONENT)
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "pearson_r": correlation,
        "int8_exact_agreement": float(np.mean(reference_int8 == candidate_int8)),
        "int8_within_one_lsb": float(np.mean(np.abs(reference_int8.astype(np.int16) - candidate_int8.astype(np.int16)) <= 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the torchaudio golden log-Mel tensors with an ESP-DL-compatible frontend.")
    parser.add_argument("golden_npz", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    golden = np.load(args.golden_npz)
    pcm16 = golden["pcm16"]
    reference = golden["logmel"][:, 0]
    rows = []
    candidates = []
    for index in range(pcm16.shape[0]):
        candidate = espdl_compatible_logmel(pcm16[index])
        candidates.append(candidate)
        rows.append({"vector_index": index, **compare(reference[index], candidate)})

    result = {
        "configuration": {
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "top_db": TOP_DB,
            "input_exponent": INPUT_EXPONENT,
        },
        "vectors": rows,
        "mean_mae": float(np.mean([row["mae"] for row in rows])),
        "mean_pearson_r": float(np.mean([row["pearson_r"] for row in rows])),
        "mean_int8_exact_agreement": float(np.mean([row["int8_exact_agreement"] for row in rows])),
    }
    if args.onnx is not None:
        import onnxruntime as ort

        candidate_batch = np.stack(candidates)[:, None].astype(np.float32)
        candidate_int8 = quantise_int8(candidate_batch, INPUT_EXPONENT)
        candidate_dequantised = candidate_int8.astype(np.float32) * math.ldexp(1.0, INPUT_EXPONENT)
        session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        fp32_logits = np.concatenate(
            [session.run(None, {"logmel": sample[None]})[0] for sample in candidate_batch], axis=0
        )
        input_quantised_logits = np.concatenate(
            [session.run(None, {"logmel": sample[None]})[0] for sample in candidate_dequantised], axis=0
        )
        expected_logits = golden["expected_logits"]
        inference_rows = []
        for index in range(fp32_logits.shape[0]):
            expected_risk = 1.0 / (1.0 + math.exp(-float(expected_logits[index, 0]) / 0.452963233))
            candidate_risk = 1.0 / (1.0 + math.exp(-float(input_quantised_logits[index, 0]) / 0.452963233))
            inference_rows.append(
                {
                    "vector_index": index,
                    "frontend_fp32_max_abs_logit_error": float(np.max(np.abs(fp32_logits[index] - expected_logits[index]))),
                    "frontend_int8_input_max_abs_logit_error": float(
                        np.max(np.abs(input_quantised_logits[index] - expected_logits[index]))
                    ),
                    "expected_risk_score": expected_risk,
                    "candidate_risk_score": candidate_risk,
                    "risk_decision_agreement": bool((expected_risk >= 0.293282151) == (candidate_risk >= 0.293282151)),
                    "validity_argmax_agreement": bool(
                        np.argmax(expected_logits[index, 1:]) == np.argmax(input_quantised_logits[index, 1:])
                    ),
                }
            )
        result["onnx_frontend_inference"] = inference_rows
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
