"""Exercise the model-loading gate over real HTTP rather than mocks.

The gate exists to tell a loaded agent apart from one serving stubs, and both
answer 200. Stubbing the transport would test the branch and skip the part that
actually goes wrong, so these spin up a real server on a loopback port.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_gate():
    spec = importlib.util.spec_from_file_location(
        "verify_model_loading", ROOT / "scripts" / "verify_model_loading.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_model_loading"] = module
    spec.loader.exec_module(module)
    return module


def serve(payload: dict | None, status: int = 200):
    """Run a one-endpoint /health server on an ephemeral port."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            body = b"" if payload is None else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep pytest output readable
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture
def gate_against():
    servers: list[HTTPServer] = []

    def _run(monkeypatch, payload: dict | None, status: int = 200) -> int:
        module = load_gate()
        server = serve(payload, status)
        servers.append(server)
        port = server.server_address[1]
        monkeypatch.setattr(module, "AGENT_PORTS", {"fraud-detection": port})
        monkeypatch.setattr(
            sys, "argv", ["verify_model_loading.py", "--base-url", "http://127.0.0.1", "--timeout", "5"]
        )
        return module.main()

    yield _run
    for server in servers:
        server.shutdown()
        server.server_close()


def test_passes_when_the_agent_reports_a_loaded_model(monkeypatch, gate_against):
    assert gate_against(monkeypatch, {"status": "healthy", "model_loaded": True}) == 0


def test_fails_when_the_agent_is_serving_stubs(monkeypatch, gate_against):
    # The case the gate exists for: healthy, responsive, and unable to score.
    assert gate_against(monkeypatch, {"status": "healthy", "model_loaded": False}) == 1


def test_fails_when_health_omits_the_flag(monkeypatch, gate_against):
    assert gate_against(monkeypatch, {"status": "healthy"}) == 1


def test_fails_on_a_non_200_health_response(monkeypatch, gate_against):
    assert gate_against(monkeypatch, {"model_loaded": True}, status=503) == 1


def test_fails_on_unparseable_health_body(monkeypatch, gate_against):
    assert gate_against(monkeypatch, None) == 1


def test_fails_when_the_agent_is_unreachable(monkeypatch):
    module = load_gate()
    # Port 1 on loopback refuses immediately rather than hanging the suite.
    monkeypatch.setattr(module, "AGENT_PORTS", {"fraud-detection": 1})
    monkeypatch.setattr(
        sys, "argv", ["verify_model_loading.py", "--base-url", "http://127.0.0.1", "--timeout", "2"]
    )
    assert module.main() == 1


def test_refuses_a_non_http_base_url(monkeypatch):
    module = load_gate()
    monkeypatch.setattr(sys, "argv", ["verify_model_loading.py", "--base-url", "file:///etc"])
    assert module.main() == 2
