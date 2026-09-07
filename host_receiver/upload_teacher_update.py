#!/usr/bin/env python3
"""Install a PC/MARH-trained shared risk head on VSN and relay it to ASN."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import serial


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "artifacts" / "marh_teacher_update"
DEFAULT_OUTPUT = ROOT / "runs" / "hardware" / "marh_teacher_update_physical"


class SerialLog:
    def __init__(self, port: str, baud: int) -> None:
        self.port = port
        self.serial = serial.Serial(port, baudrate=baud, timeout=0.1)
        self.lines: list[str] = []
        self.events: queue.Queue[dict] = queue.Queue()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        while not self.stop.is_set():
            raw = self.serial.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").rstrip()
            self.lines.append(line)
            marker = "NLJSON "
            if marker in line:
                try:
                    self.events.put(json.loads(line.split(marker, 1)[1]))
                except json.JSONDecodeError:
                    pass

    def write(self, command: str) -> None:
        self.serial.write((command + "\n").encode("ascii"))
        self.serial.flush()

    def wait_command(self, name: str, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=min(0.25, deadline - time.monotonic()))
            except queue.Empty:
                continue
            if event.get("event") == "command_result" and event.get("command") == name:
                return event
        raise TimeoutError(f"No {name} result from {self.port} within {timeout:.1f}s")

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=1.0)
        self.serial.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsn-port", default="COM7")
    parser.add_argument("--asn-port", default="COM8")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-bytes", type=int, default=16)
    parser.add_argument("--skip-sample", action="store_true")
    return parser.parse_args()


def command(log: SerialLog, text: str, result_name: str, timeout: float = 10.0) -> dict:
    log.write(text)
    result = log.wait_command(result_name, timeout)
    if not result.get("success"):
        raise RuntimeError(f"{result_name} failed: {result.get('detail')}")
    return result


def main() -> int:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((package_dir / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("deployment_accepted"):
        raise RuntimeError("Candidate did not pass the offline deployment gate")
    payload = (package_dir / "candidate_shared_head_int8.bin").read_bytes()
    computed_crc = zlib.crc32(payload) & 0xFFFFFFFF
    expected_crc = int(str(summary["crc32"]), 16)
    if len(payload) != int(summary["payload_bytes"]) or computed_crc != expected_crc:
        raise RuntimeError("Candidate payload length or CRC does not match summary.json")

    started = time.monotonic()
    vsn = SerialLog(args.vsn_port, args.baud)
    asn = SerialLog(args.asn_port, args.baud) if args.asn_port else None
    results: list[dict] = []
    try:
        time.sleep(2.0)
        mode_deadline = time.monotonic() + 20.0
        while True:
            vsn.write("MODE MANUAL")
            mode_result = vsn.wait_command("MODE_MANUAL", timeout=5.0)
            if mode_result.get("success"):
                results.append(mode_result)
                break
            if time.monotonic() >= mode_deadline:
                raise RuntimeError(f"ASN did not become available: {mode_result.get('detail')}")
            time.sleep(1.0)
        version_hex = str(summary["candidate_version_hex"]).removeprefix("0x")
        begin = (
            f"HOST MODEL BEGIN {version_hex} {len(payload)} {computed_crc:08x} "
            f"{float(summary['expected_golden_output']):.10g} "
            f"{float(summary['thresholds']['vsn']):.8g} "
            f"{float(summary['thresholds']['asn']):.8g}"
        )
        results.append(command(vsn, begin, "HOST_MODEL_BEGIN"))
        for offset in range(0, len(payload), args.chunk_bytes):
            chunk = payload[offset : offset + args.chunk_bytes]
            vsn.write(f"HOST MODEL CHUNK {offset} {chunk.hex()}")
            chunk_result = vsn.wait_command("HOST_MODEL_CHUNK", timeout=5.0)
            if not chunk_result.get("success"):
                print("\n".join(vsn.lines[-40:]))
                raise RuntimeError(
                    f"HOST_MODEL_CHUNK failed at offset {offset}: {chunk_result.get('detail')}"
                )
        results.append(command(vsn, "HOST MODEL END", "HOST_MODEL_END", timeout=15.0))
        results.append(command(vsn, "HOST MODEL PUSH", "HOST_MODEL_PUSH", timeout=30.0))
        vsn.write("NODE STATUS")
        time.sleep(1.0)
        if not args.skip_sample:
            results.append(command(vsn, "SAMPLE BOTH", "SAMPLE_BOTH"))
            time.sleep(8.0)
        else:
            time.sleep(2.0)
    finally:
        vsn.close()
        if asn is not None:
            asn.close()

    vsn_text = "\n".join(vsn.lines) + "\n"
    asn_text = "\n".join(asn.lines) + "\n" if asn is not None else ""
    (output_dir / "vsn_serial.log").write_text(vsn_text, encoding="utf-8")
    (output_dir / "asn_serial.log").write_text(asn_text, encoding="utf-8")

    checks = {
        "offline_gate_passed": bool(summary["deployment_accepted"]),
        "payload_crc_verified_on_pc": computed_crc == expected_crc,
        "vsn_staging_and_activation_passed": any(
            item.get("command") == "HOST_MODEL_END" and item.get("success") for item in results
        ),
        "ble_relay_and_asn_install_passed": any(
            item.get("command") == "HOST_MODEL_PUSH" and item.get("success") for item in results
        ),
        "asn_runtime_log_confirms_version": f"active_version=0x{int(summary['candidate_version']):04x}" in asn_text.lower(),
        "fresh_post_update_fusion_observed": (
            f'\"version\":{int(summary["candidate_version"])}' in vsn_text
            and '"event":"fusion"' in vsn_text
        ),
    }
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.monotonic() - started,
        "ports": {"vsn": args.vsn_port, "asn": args.asn_port},
        "candidate": {
            "version": summary["candidate_version"],
            "version_hex": summary["candidate_version_hex"],
            "payload_bytes": len(payload),
            "crc32": f"0x{computed_crc:08x}",
            "vsn_threshold": summary["thresholds"]["vsn"],
            "asn_threshold": summary["thresholds"]["asn"],
        },
        "command_results": results,
        "checks": checks,
        "overall_pass": all(checks.values()),
        "claim_boundary": (
            "This run validates PC-trained parameter delivery, integrity checking and hot installation "
            "on two physical nodes; it does not establish field structural-health validity."
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Physical PC/MARH Teacher Update",
        "",
        f"- Candidate: `{summary['candidate_version_hex']}` ({len(payload)} B INT8, CRC `{summary['crc32']}`)",
        f"- Route: PC/MARH -> `{args.vsn_port}` VSN -> BLE -> `{args.asn_port}` ASN",
        f"- Overall result: **{'PASS' if report['overall_pass'] else 'INCOMPLETE'}**",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in checks.items())
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
