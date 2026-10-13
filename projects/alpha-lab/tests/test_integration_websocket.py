"""Integration tests for WebSocket connections."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import alpha_lab.server as srv
from alpha_lab.server import app


@pytest.fixture(autouse=True)
def setup_ws_server(tmp_workspace: str) -> None:
    """Reset server globals before each test."""
    srv._workspace = tmp_workspace
    srv.event_history.clear()
    srv.connected_websockets.clear()
    srv.manager = srv.DashboardState()
    srv.manager.workspace = tmp_workspace
    srv.manager.db = None


def get_client():
    from fastapi.testclient import TestClient
    return TestClient(app)


class TestWebSocketConnect:
    def test_connect_and_disconnect(self, tmp_workspace: str) -> None:
        client = get_client()
        with client.websocket_connect("/ws") as ws:
            # Connection should succeed — just disconnect cleanly
            pass
        # No error means success


class TestWebSocketHistory:
    def test_receives_history_on_connect(self, tmp_workspace: str) -> None:
        # Pre-populate event_history
        srv.event_history.extend([
            {"type": "status", "status": "starting", "detail": "Agent starting"},
            {"type": "status", "status": "thinking", "detail": "Calling API..."},
            {"type": "api_request", "instructions": "Internal payload is not replayed"},
            {"type": "agent_text", "delta": "Ignored token delta", "full_text": "Ignored"},
            {"type": "tool_result", "name": "inspect", "output": "Done"},
        ])

        client = get_client()
        with client.websocket_connect("/ws") as ws:
            received = []
            # Receive all 3 history events
            for _ in range(3):
                data = ws.receive_json()
                received.append(data)

            assert len(received) == 3
            assert received[0]["type"] == "status"
            assert received[0]["status"] == "starting"
            assert received[2]["type"] == "tool_result"


class TestWebSocketCommands:
    def test_viewer_ignores_stop_but_generates_status(self, tmp_workspace: str) -> None:
        client = get_client()
        from unittest.mock import patch
        with patch("alpha_lab.output_generator.OutputGenerator") as generator:
            generator.return_value.generate_status_report.return_value={"check":"report"}
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type":"stop"})
                ws.send_json({"type":"generate_status"})
                result=ws.receive_json()
                assert result["type"]=="status_report" and result["check"]=="report"
                generator.return_value.generate_status_report.assert_called_once()
