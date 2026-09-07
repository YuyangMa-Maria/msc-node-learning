"""Live external receiver for the VSN representative node.

The receiver deliberately has no fusion model. It records the VSN's decisions
in raw, JSONL and tabular forms so that a physical run can be audited without
changing the on-device outcome after the fact.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from protocol import ProtocolError, ReceiverState, format_event, fusion_csv_row, parse_event


CSV_FIELDS = [
    "host_time_iso", "device_uptime_ms", "fusion_index", "mode",
    "vsn_sequence", "vsn_risk_score", "vsn_confidence", "vsn_status",
    "asn_available", "asn_sequence", "asn_age_ms", "asn_risk_score",
    "asn_confidence", "asn_status", "fused_risk_score", "fused_confidence",
    "risk_level", "risk_trend", "active_nodes", "alert_attention_required",
    "alert_severity", "alert_headline", "alert_evidence", "alert_action",
    "alert_message", "triggered_mask", "marh_active", "marh_recommended",
    "marh_reason", "model_version", "model_generation", "model_format",
]

NODE_CSV_FIELDS = [
    "host_time_iso", "device_uptime_ms", "node", "phase", "source", "transport",
    "sequence", "source_uptime_ms", "received_uptime_ms", "risk_score", "confidence",
    "status", "capture_ms", "decode_ms", "preprocess_ms", "inference_ms", "total_ms",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def default_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / "hardware" / "final_system" / stamp


class RunLogger:
    """Write synchronised raw, structured, fusion and node-level run records."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=False)
        self.raw: TextIO = (run_dir / "vsn_serial_raw.log").open("w", encoding="utf-8")
        self.events: TextIO = (run_dir / "events.jsonl").open("w", encoding="utf-8")
        self.csv_file: TextIO = (run_dir / "fusion.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.csv = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
        self.csv.writeheader()
        self.node_csv_file: TextIO = (run_dir / "node_outputs.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.node_csv = csv.DictWriter(self.node_csv_file, fieldnames=NODE_CSV_FIELDS)
        self.node_csv.writeheader()

    def write_raw(self, host_time: str, line: str) -> None:
        self.raw.write(f"{host_time} {line}\n")
        self.raw.flush()

    def write_event(self, host_time: str, event: dict) -> None:
        record = {"host_time_iso": host_time, **event}
        self.events.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.events.flush()
        if event["event"] == "fusion":
            self.csv.writerow(fusion_csv_row(event, host_time))
            self.csv_file.flush()
        elif event["event"] == "node_output":
            latency = event.get("latency_ms", {})
            self.node_csv.writerow(
                {
                    "host_time_iso": host_time,
                    "device_uptime_ms": event.get("uptime_ms"),
                    "node": event.get("node"),
                    "phase": event.get("phase"),
                    "source": event.get("source"),
                    "transport": event.get("transport"),
                    "sequence": event.get("sequence"),
                    "source_uptime_ms": event.get("source_uptime_ms"),
                    "received_uptime_ms": event.get("received_uptime_ms"),
                    "risk_score": event.get("risk_score"),
                    "confidence": event.get("confidence"),
                    "status": event.get("status"),
                    "capture_ms": latency.get("capture"),
                    "decode_ms": latency.get("decode"),
                    "preprocess_ms": latency.get("preprocess"),
                    "inference_ms": latency.get("inference"),
                    "total_ms": latency.get("total"),
                }
            )
            self.node_csv_file.flush()

    def close(self) -> None:
        self.raw.close()
        self.events.close()
        self.csv_file.close()
        self.node_csv_file.close()


SUPPORTED_COMMANDS = {
    "ping": "PING",
    "status": "STATUS",
    "node status": "NODE STATUS",
    "ble rescan": "BLE RESCAN",
    "system restart": "SYSTEM RESTART",
    "mode auto": "MODE AUTO",
    "mode manual": "MODE MANUAL",
    "sample vsn": "SAMPLE VSN",
    "sample asn": "SAMPLE ASN",
    "sample both": "SAMPLE BOTH",
    "marh join": "MARH JOIN",
    "marh leave": "MARH LEAVE",
    "model push": "MODEL PUSH",
    "model pull": "MODEL PULL",
    "model rollback": "MODEL ROLLBACK",
    "help": "HELP",
}


def command_worker(command_queue: queue.Queue[str]) -> None:
    """Translate interactive operator input into the constrained wire commands."""
    while True:
        try:
            typed = input().strip().lower()
        except EOFError:
            return
        if typed in {"quit", "exit"}:
            command_queue.put("__QUIT__")
            return
        command = SUPPORTED_COMMANDS.get(typed)
        if command is None:
            print("Commands: " + ", ".join(SUPPORTED_COMMANDS) + ", quit")
            continue
        command_queue.put(command)


def load_command_plan(path: Path | None) -> list[dict]:
    """Validate and time-order a reproducible hardware command plan."""
    if path is None:
        return []
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, list):
        raise ValueError("command plan must be a JSON list")
    supported_wire_commands = set(SUPPORTED_COMMANDS.values())
    checked = []
    for index, item in enumerate(plan):
        if not isinstance(item, dict):
            raise ValueError(f"command plan item {index} must be an object")
        at_s = float(item.get("at_s", -1))
        command = str(item.get("command", "")).upper()
        if at_s < 0 or command not in supported_wire_commands:
            raise ValueError(f"invalid command plan item {index}: {item!r}")
        checked.append({"at_s": at_s, "command": command})
    return sorted(checked, key=lambda item: item["at_s"])


def capture_debug(port_name: str, baud: int, destination: Path, stop: threading.Event) -> None:
    """Capture the ASN console separately; it is diagnostic, not a decision path."""
    import serial

    with serial.Serial(port_name, baud, timeout=0.25) as port, destination.open(
        "w", encoding="utf-8"
    ) as output:
        while not stop.is_set():
            raw = port.readline()
            if raw:
                output.write(f"{utc_now()} {raw.decode('utf-8', errors='replace').rstrip()}\n")
                output.flush()


def main() -> int:
    """Run the serial event loop until its duration expires or the user stops it."""
    parser = argparse.ArgumentParser(
        description="Receive fused decisions from the VSN representative node."
    )
    parser.add_argument("--port", default="COM7", help="VSN USB serial port")
    parser.add_argument("--asn-debug-port", help="Optional ASN serial port; raw logging only")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    parser.add_argument("--command-plan", type=Path, help="Timed JSON command plan")
    args = parser.parse_args()

    command_plan = load_command_plan(args.command_plan)

    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required: python -m pip install pyserial") from exc

    run_dir = args.run_dir or default_run_dir()
    logger = RunLogger(run_dir)
    state = ReceiverState()
    stop = threading.Event()
    command_queue: queue.Queue[str] = queue.Queue()
    debug_thread = None
    plan_index = 0

    print(f"Run data: {run_dir.resolve()}")
    print("Decision path: ASN --BLE--> VSN fusion/alert --USB serial--> this receiver")
    print("The receiver records VSN decisions; it does not recompute fusion.")

    try:
        with serial.Serial(args.port, args.baud, timeout=0.25) as port:
            if args.asn_debug_port:
                debug_thread = threading.Thread(
                    target=capture_debug,
                    args=(args.asn_debug_port, args.baud, run_dir / "asn_serial_debug.log", stop),
                    daemon=True,
                )
                debug_thread.start()
            if args.interactive:
                threading.Thread(
                    target=command_worker,
                    args=(command_queue,),
                    daemon=True,
                ).start()
                print("Interactive commands enabled; type 'help' or 'quit'.")

            started = time.monotonic()
            while not stop.is_set():
                elapsed = time.monotonic() - started
                if args.duration is not None and elapsed >= args.duration:
                    break
                while plan_index < len(command_plan) and command_plan[plan_index]["at_s"] <= elapsed:
                    command = command_plan[plan_index]["command"]
                    port.write((command + "\n").encode("ascii"))
                    print(f"[PLAN {elapsed:.1f}s] sent {command}")
                    plan_index += 1
                should_quit = False
                while True:
                    try:
                        queued_command = command_queue.get_nowait()
                    except queue.Empty:
                        break
                    if queued_command == "__QUIT__":
                        should_quit = True
                        break
                    port.write((queued_command + "\n").encode("ascii"))
                if should_quit:
                    break
                raw = port.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").rstrip()
                host_time = utc_now()
                logger.write_raw(host_time, line)
                try:
                    event = parse_event(line)
                except ProtocolError as exc:
                    print(f"[PROTOCOL WARNING] {exc}")
                    continue
                if event is None:
                    continue
                state.update(event)
                logger.write_event(host_time, event)
                print(format_event(event))
    except KeyboardInterrupt:
        print("\nCapture stopped.")
    finally:
        stop.set()
        if debug_thread is not None:
            debug_thread.join(timeout=1.0)
        logger.close()

    print(f"Saved {sum(state.event_counts.values())} structured events to {run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
