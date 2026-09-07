"""Local browser dashboard for the physical Node Learning demonstration.

The HTTP layer is intentionally dependency-free. Server-sent events carry the
validated serial records to the browser, while POST requests expose only the
small command vocabulary understood by the VSN.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import webbrowser
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from protocol import ProtocolError, ReceiverState, parse_event
from receiver import RunLogger, SUPPORTED_COMMANDS, capture_debug, default_run_dir, utc_now


STATIC_DIR = Path(__file__).with_name("dashboard_static")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


class EventHub:
    """Fan validated events out to the currently connected browser clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=128)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                # A slow browser may lose an old display event, but must not
                # stall serial ingestion or the physical sensing loop.
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(event)
                except queue.Empty:
                    pass


class SerialBridge:
    """Own the VSN serial link and expose thread-safe state to the HTTP server."""

    def __init__(
        self,
        port_name: str,
        baud: int,
        run_dir: Path,
        hub: EventHub,
        asn_debug_port: str | None,
    ) -> None:
        import serial

        self.serial = serial.Serial(port_name, baud, timeout=0.25)
        self.write_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.state = ReceiverState()
        self.history: deque[dict[str, Any]] = deque(maxlen=100)
        self.hub = hub
        self.logger = RunLogger(run_dir)
        self.stop = threading.Event()
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.debug_reader: threading.Thread | None = None
        if asn_debug_port:
            self.debug_reader = threading.Thread(
                target=capture_debug,
                args=(asn_debug_port, baud, run_dir / "asn_serial_debug.log", self.stop),
                daemon=True,
            )

    def start(self) -> None:
        self.reader.start()
        if self.debug_reader is not None:
            self.debug_reader.start()

    def _read_loop(self) -> None:
        """Record serial input before publishing validated events to clients."""
        while not self.stop.is_set():
            raw = self.serial.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").rstrip()
            host_time = utc_now()
            self.logger.write_raw(host_time, line)
            try:
                event = parse_event(line)
            except ProtocolError as exc:
                self.hub.publish(
                    {"event": "host_warning", "message": str(exc), "host_time_iso": host_time}
                )
                continue
            if event is None:
                continue
            record = {"host_time_iso": host_time, **event}
            with self.state_lock:
                self.state.update(event)
                self.history.append(record)
            self.logger.write_event(host_time, event)
            self.hub.publish(record)

    def command(self, command: str) -> None:
        with self.write_lock:
            self.serial.write((command + "\n").encode("ascii"))
            self.serial.flush()

    def snapshot(self) -> dict[str, Any]:
        with self.state_lock:
            return {
                "ble_connected": self.state.ble_connected,
                "marh_active": self.state.marh_active,
                "sampling_mode": self.state.sampling_mode,
                "model_version": self.state.model_version,
                "latest_nodes": self.state.latest_nodes,
                "node_states": self.state.node_states,
                "latest_fusion": self.state.latest_fusion,
                "event_counts": self.state.event_counts,
                "history": list(self.history),
            }

    def close(self) -> None:
        self.stop.set()
        self.reader.join(timeout=1.0)
        if self.debug_reader is not None:
            self.debug_reader.join(timeout=1.0)
        self.serial.close()
        self.logger.close()


class DashboardApp:
    def __init__(self, bridge: SerialBridge, hub: EventHub, run_dir: Path) -> None:
        self.bridge = bridge
        self.hub = hub
        self.run_dir = run_dir


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve static assets, snapshots, event streams and validated commands."""

    server: "DashboardServer"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            payload = self.server.app.bridge.snapshot()
            payload["run_dir"] = str(self.server.app.run_dir.resolve())
            self._json(payload)
            return
        if path == "/api/events":
            self._events()
            return
        static_name = "index.html" if path == "/" else path.lstrip("/")
        if static_name not in {"index.html", "styles.css", "app.js"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        destination = STATIC_DIR / static_name
        if not destination.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        payload = destination.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", CONTENT_TYPES[destination.suffix])
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _events(self) -> None:
        """Keep one server-sent event stream alive until the client disconnects."""
        subscriber = self.server.app.hub.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                try:
                    event = subscriber.get(timeout=15.0)
                    payload = json.dumps(event, separators=(",", ":"))
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.server.app.hub.unsubscribe(subscriber)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/command":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            command = str(payload.get("command", "")).upper()
            supported = set(SUPPORTED_COMMANDS.values())
            if command not in supported:
                raise ValueError("unsupported command")
            self.server.app.bridge.command(command)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"accepted": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._json({"accepted": True, "command": command}, HTTPStatus.ACCEPTED)


class DashboardServer(ThreadingHTTPServer):
    """Threaded local server carrying the application state explicitly."""
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: DashboardApp) -> None:
        super().__init__(address, DashboardHandler)
        self.app = app


def main() -> int:
    parser = argparse.ArgumentParser(description="Node Learning hardware demonstration dashboard")
    parser.add_argument("--port", default="COM7", help="VSN serial port")
    parser.add_argument("--asn-debug-port", default="COM8")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir or default_run_dir()
    hub = EventHub()
    bridge = SerialBridge(args.port, args.baud, run_dir, hub, args.asn_debug_port)
    app = DashboardApp(bridge, hub, run_dir)
    server = DashboardServer((args.host, args.http_port), app)
    url = f"http://{args.host}:{args.http_port}"
    print(f"Dashboard: {url}")
    print(f"Run data: {run_dir.resolve()}")
    bridge.start()
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
