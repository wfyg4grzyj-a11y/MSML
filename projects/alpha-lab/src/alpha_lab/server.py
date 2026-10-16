"""FastAPI + WebSocket server for alpha-lab web dashboard.

Pure viewer mode: point at a --workspace to browse files and monitor a
running or completed analysis.  The pipeline is started separately via
``python run.py`` — this server never starts or stops it.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import heapq
import json
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from alpha_lab.events import AgentEvent, FileChangedEvent
from alpha_lab.metrics import MetricsCollector

logger = logging.getLogger("alpha_lab.server")


# ---------------------------------------------------------------------------
# Dashboard State
# ---------------------------------------------------------------------------


class DashboardState:
    """Holds async event queue and references used by read-only endpoints."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[AgentEvent] | None = None
        self.workspace: str = ""
        self.db: Any = None  # ExperimentDB, loaded lazily from workspace

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue = asyncio.Queue()

    def _try_load_db(self) -> None:
        """Load experiment DB from workspace if it exists."""
        if self.db is not None or not self.workspace:
            return
        db_path = os.path.join(self.workspace, "experiments.db")
        if os.path.exists(db_path):
            from alpha_lab.experiment_db import ExperimentDB
            self.db = ExperimentDB(db_path)


# ---------------------------------------------------------------------------
# Log Tailer — streams live JSONL events from the pipeline into the WS queue
# ---------------------------------------------------------------------------


class LogTailer:
    """Tails all .jsonl files in {workspace}/logs/ and pushes new events."""

    def __init__(
        self,
        workspace: str,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
    ) -> None:
        self.logs_dir = Path(workspace) / "logs"
        self._loop = loop
        self._queue = queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Track file positions: {path: offset}
        self._offsets: dict[str, int] = {}
        self._identities: dict[str, tuple[int, int]] = {}
        self._checkpoints: dict[str, bytes] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            if self._stop_event.is_set():
                raise RuntimeError("Previous log tailer is still stopping")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._tail, daemon=True, name="alpha-lab-log-tailer")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if (self._thread is not None and self._thread.ident is not None
                and self._thread is not threading.current_thread()):
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                logger.warning("Log tailer did not stop within two seconds")

    def _tail(self) -> None:
        """Poll for new lines in all .jsonl files every 0.5s."""
        while not self._stop_event.is_set():
            try:
                for log_path in self.logs_dir.glob("*.jsonl"):
                    if self._stop_event.is_set():
                        break
                    self._read_new_lines(str(log_path))
            except OSError:
                logger.debug("Cannot scan dashboard log directory")
            self._stop_event.wait(0.5)

    def _read_new_lines(self, path: str, emit=None) -> None:
        """Consume complete UTF-8 lines; preserve a partial line for the next poll.

        Byte cursors agree with filesystem sizes on Windows. File identity and
        a small checkpoint detect replacement/truncation between polls. Startup
        history uses this same reader/cursor, avoiding a load-to-tail gap.
        """
        offset = self._offsets.get(path, 0)
        try:
            with open(path, 'rb') as f:
                stat = os.fstat(f.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if (offset>stat.st_size or (path in self._identities
                                          and self._identities[path]!=identity)):
                    offset=0
                checkpoint=self._checkpoints.get(path)
                if offset and checkpoint is not None:
                    f.seek(max(0,offset-len(checkpoint)))
                    if f.read(len(checkpoint))!=checkpoint:
                        offset=0
                f.seek(offset)
                while True:
                    line_start=f.tell()
                    line=f.readline()
                    if not line:
                        break
                    if not line.endswith(b'\n'):
                        f.seek(line_start)
                        break
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if not isinstance(data,dict):
                        continue
                    if emit is not None:
                        emit(data)
                    elif not self._stop_event.is_set() and not self._loop.is_closed():
                        self._loop.call_soon_threadsafe(self._queue.put_nowait, _DictEvent(data))
                self._offsets[path] = f.tell()
                self._identities[path] = identity
                f.seek(max(0,self._offsets[path]-64))
                self._checkpoints[path] = f.read(min(64,self._offsets[path]))
        except (OSError, RuntimeError):
            logger.debug("Dashboard log unavailable or event loop closed")


class _DictEvent:
    """Thin wrapper so pre-parsed dicts work with the broadcast loop."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


# ---------------------------------------------------------------------------
# File Watcher
# ---------------------------------------------------------------------------


class FileWatcher:
    """Watches workspace for file changes using watchfiles."""

    def __init__(
        self,
        workspace: str,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[AgentEvent],
    ) -> None:
        self.workspace = workspace
        self._loop = loop
        self._queue = queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            if self._stop_event.is_set():
                raise RuntimeError("Previous file watcher is still stopping")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch, daemon=True, name="alpha-lab-file-watcher")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if (self._thread is not None and self._thread.ident is not None
                and self._thread is not threading.current_thread()):
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                logger.warning("File watcher did not stop within two seconds")

    def _watch(self) -> None:
        try:
            from watchfiles import watch, Change

            for changes in watch(
                self.workspace,
                stop_event=self._stop_event,
                debounce=500,
                rust_timeout=1000,
                recursive=True,
            ):
                for change_type, path_str in changes:
                    if self._stop_event.is_set() or self._loop.is_closed():
                        return
                    rel_path = os.path.relpath(path_str, self.workspace)

                    if any(
                        part.startswith(".") or part == "__pycache__"
                        for part in Path(rel_path).parts
                    ):
                        continue

                    change_name = {
                        Change.added: "added",
                        Change.modified: "modified",
                        Change.deleted: "deleted",
                    }.get(change_type, "modified")

                    event = FileChangedEvent(
                        change=change_name,
                        path=rel_path,
                    )
                    self._loop.call_soon_threadsafe(
                        self._queue.put_nowait, event
                    )
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"File watcher error: {e}")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    global _runtime_active
    if _runtime_active:
        raise RuntimeError("Dashboard already has an active lifespan")
    _runtime_active = True
    try:
        await startup()
        yield
    finally:
        try:
            await shutdown()
        finally:
            _runtime_active = False


app = FastAPI(title="Alpha Lab", lifespan=lifespan)

_server_start_time: float = 0.0
_metrics_collector = MetricsCollector()

manager = DashboardState()
file_watcher: FileWatcher | None = None
log_tailer: LogTailer | None = None
connected_websockets: set[WebSocket] = set()
event_history: list[dict] = []  # All events, replayed to new WS connections
MAX_EVENT_HISTORY = 10_000  # Cap to prevent OOM on long runs
_broadcast_task: asyncio.Task | None = None  # prevent GC of broadcast loop
_status_report_task: asyncio.Task | None = None  # prevent GC of timer
_manual_status_tasks: set[asyncio.Task] = set()
_runtime_active = False

# Set by serve_main.
_workspace: str = ""
_research_root: str | None = None
_adapter_cache: dict = {}  # {"adapter": DomainAdapter | None}
_chat_config_path: Path | None = None
_chat_session: Any = None
_chat_configuration_error: str | None = None
_chat_request_tasks: set[asyncio.Task] = set()


def _load_jsonl_history(tailer: LogTailer) -> None:
    """Keep only the most recent bounded history and reuse its byte cursors."""
    logs_dir = Path(_workspace) / "logs"
    if not logs_dir.exists():
        return
    recent: list[tuple[float, int, dict]] = []
    total_loaded = 0

    def remember(data):
        nonlocal total_loaded
        timestamp=data.get('timestamp',0)
        if (not isinstance(timestamp,(int,float)) or isinstance(timestamp,bool)
                or not math.isfinite(timestamp)):
            timestamp=0
        total_loaded+=1
        entry=(timestamp,total_loaded,data)
        if len(recent)<MAX_EVENT_HISTORY:
            heapq.heappush(recent,entry)
        else:
            heapq.heappushpop(recent,entry)

    for log_path in sorted(logs_dir.glob("*.jsonl")):
        tailer._read_new_lines(str(log_path),emit=remember)
    event_history[:] = [entry[2] for entry in sorted(recent)]
    if total_loaded:
        logger.info(f"Loaded {len(event_history)} recent events from {total_loaded} complete records")


async def startup() -> None:
    global file_watcher, log_tailer, _broadcast_task, _status_report_task, _server_start_time
    global _chat_session, _chat_configuration_error

    _server_start_time = time.time()
    loop = asyncio.get_running_loop()
    manager.set_loop(loop)
    manager.workspace = _workspace
    manager.db = None
    event_history.clear()
    connected_websockets.clear()
    _adapter_cache.clear()
    _chat_session = None
    _chat_configuration_error = None
    if _chat_config_path is not None:
        try:
            from dotenv import load_dotenv
            from alpha_lab.dashboard_chat import PublicChatSession
            load_dotenv()
            _chat_session = PublicChatSession(_chat_config_path)
        except Exception as error:
            _chat_configuration_error = "External research chat configuration is invalid or its run directory already exists."
            logger.warning("External chat disabled: %s",type(error).__name__)

    _broadcast_task = asyncio.create_task(_broadcast_loop())
    _status_report_task = asyncio.create_task(_status_report_timer())

    # Try to load adapter for metric-aware endpoints
    if _workspace:
        try:
            from alpha_lab.adapter_loader import resolve_adapter
            _adapter_cache["adapter"] = resolve_adapter(_workspace)
        except Exception as e:
            logger.warning(f"Failed to load adapter at startup: {e}")
            _adapter_cache["adapter"] = None

    # Load experiment DB if it exists
    manager._try_load_db()

    # Start log tailer — streams live JSONL events from the pipeline
    if _workspace:
        log_tailer = LogTailer(_workspace, loop, manager._queue)
        _load_jsonl_history(log_tailer)
        log_tailer.start()

    # Always start file watcher if workspace exists
    if _workspace and Path(_workspace).exists():
        file_watcher = FileWatcher(_workspace, loop, manager._queue)
        file_watcher.start()


async def shutdown() -> None:
    """Release only viewer-owned resources; pipeline jobs run independently."""
    global file_watcher, log_tailer, _broadcast_task, _status_report_task, _server_start_time
    global _chat_session, _chat_configuration_error
    if _chat_session is not None:
        try:
            await asyncio.to_thread(_chat_session.close)
        except Exception as error:
            logger.warning("Chat cleanup failed: %s",type(error).__name__)
    owners=[owner for owner in (file_watcher,log_tailer) if owner is not None]
    for owner in owners:
        owner._stop_event.set()
    if owners:
        await asyncio.gather(*(asyncio.to_thread(owner.stop) for owner in owners))
    tasks=set(_manual_status_tasks)
    tasks.update(_chat_request_tasks)
    tasks.update(task for task in (_broadcast_task,_status_report_task) if task is not None)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks,return_exceptions=True)
    for ws in list(connected_websockets):
        try:
            await asyncio.wait_for(ws.close(code=1001),timeout=2)
        except Exception:
            pass
    connected_websockets.clear()
    _manual_status_tasks.clear()
    _chat_request_tasks.clear()
    _chat_session = None
    _chat_configuration_error = None
    file_watcher = log_tailer = None
    _broadcast_task = _status_report_task = None
    _server_start_time = 0.
    manager._queue = manager._loop = manager.db = None
    event_history.clear()
    _adapter_cache.clear()


def _remember_event(data: dict) -> None:
    event_history.append(data)
    if len(event_history)>MAX_EVENT_HISTORY:
        del event_history[:max(1,MAX_EVENT_HISTORY//5)]


async def _broadcast_loop() -> None:
    """Consume events from the queue and broadcast to all WebSockets."""
    while True:
        if manager._queue is None:
            await asyncio.sleep(0.1)
            continue

        try:
            event = await asyncio.wait_for(manager._queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        data = event.to_dict()
        _remember_event(data)

        dead: set[WebSocket] = set()
        for ws in list(connected_websockets):  # snapshot to avoid set-changed-during-iteration
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        connected_websockets.difference_update(dead)


async def _generate_and_broadcast_status() -> None:
    """Generate a status report and broadcast it to all connected clients."""
    if not _workspace:
        return
    try:
        from alpha_lab.output_generator import OutputGenerator
        gen = OutputGenerator(_workspace, adapter=_adapter_cache.get("adapter"))
        report = gen.generate_status_report()
        data = {"type": "status_report", "timestamp": time.time(), **report}
        _remember_event(data)

        dead: set[WebSocket] = set()
        for ws in list(connected_websockets):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        connected_websockets.difference_update(dead)
    except Exception as e:
        logger.error(f"Status report broadcast failed: {e}")


_STATUS_REPORT_INTERVAL = 600  # seconds (10 minutes)


async def _status_report_timer() -> None:
    """Auto-generate status reports every 10 minutes while agent is running."""
    while True:
        await asyncio.sleep(_STATUS_REPORT_INTERVAL)
        if connected_websockets:
            await _generate_and_broadcast_status()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


# Event types the frontend actually renders (skip bloated internal events)
_REPLAY_TYPES = {
    "status", "tool_call", "tool_result", "phase", "experiment",
    "board_summary", "question", "error", "file_changed", "status_report",
}

# Max size (chars) for a single replayed event payload
_REPLAY_MAX_CHARS = 20_000


def _slim_event(data: dict) -> dict | None:
    """Return a trimmed copy of *data* suitable for replay, or None to skip."""
    etype = data.get("type", "")

    # Skip event types the frontend ignores (api_request, api_response, etc.)
    if etype not in _REPLAY_TYPES:
        return None

    # Truncate oversized string fields (e.g. tool outputs)
    slim = dict(data)
    for key in ("output", "arguments", "content"):
        val = slim.get(key)
        if isinstance(val, str) and len(val) > _REPLAY_MAX_CHARS:
            slim[key] = val[:_REPLAY_MAX_CHARS] + "\n··· truncated for replay ···"
    return slim


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()

    # Add client first so it receives live events during replay
    connected_websockets.add(ws)

    # Replay a trimmed snapshot of history (skip bloated internal events)
    history_snapshot = list(event_history)
    for data in history_snapshot:
        slim = _slim_event(data)
        if slim is None:
            continue
        try:
            await ws.send_json(slim)
        except Exception:
            connected_websockets.discard(ws)
            return

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "generate_status":
                task=asyncio.create_task(_generate_and_broadcast_status())
                _manual_status_tasks.add(task)
                task.add_done_callback(_manual_status_tasks.discard)

    except WebSocketDisconnect:
        pass
    finally:
        connected_websockets.discard(ws)


# ---------------------------------------------------------------------------
# REST API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/status")
async def get_status() -> JSONResponse:
    return JSONResponse({
        "workspace": _workspace,
    })


@app.get("/api/files")
async def get_files() -> JSONResponse:
    """Return workspace directory tree as nested JSON."""
    if not _workspace or not Path(_workspace).exists():
        return JSONResponse({"children": []})

    def build_tree(dir_path: Path, rel_prefix: str = "") -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        try:
            entries = sorted(dir_path.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        except PermissionError:
            return items

        for entry in entries:
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue

            rel = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name

            if entry.is_dir():
                items.append({
                    "name": entry.name,
                    "path": rel,
                    "type": "directory",
                    "children": build_tree(entry, rel),
                })
            else:
                items.append({
                    "name": entry.name,
                    "path": rel,
                    "type": "file",
                })

        return items

    tree = build_tree(Path(_workspace))
    return JSONResponse({"children": tree})


@app.get("/api/file/{path:path}")
async def get_file(path: str) -> JSONResponse:
    """Return file content. Text as JSON, images as base64."""
    if not _workspace:
        return JSONResponse({"error": "No workspace"}, status_code=400)

    full_path = (Path(_workspace) / path).resolve()
    workspace_resolved = Path(_workspace).resolve()
    try:
        full_path.relative_to(workspace_resolved)
    except ValueError:
        return JSONResponse({"error": "Path traversal denied"}, status_code=403)

    if not full_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    if not full_path.is_file():
        return JSONResponse({"error": "Not a file"}, status_code=400)

    suffix = full_path.suffix.lower()
    image_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
    }

    if suffix in image_types:
        import base64
        data = base64.b64encode(full_path.read_bytes()).decode("ascii")
        return JSONResponse({
            "type": "image",
            "media_type": image_types[suffix],
            "data": data,
            "name": full_path.name,
        })

    try:
        content = full_path.read_text(errors="replace")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({
        "type": "text",
        "content": content,
        "name": full_path.name,
    })


# ---------------------------------------------------------------------------
# Health & Metrics endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def get_health() -> JSONResponse:
    """Return structured health status of all components."""
    uptime = time.time() - _server_start_time if _server_start_time else 0.0

    # Database status
    manager._try_load_db()
    db_status: dict[str, Any] = {"status": "not_configured"}
    if manager.db is not None:
        try:
            board = manager.db.board_summary()
            db_status = {"status": "ok", "board": board}
        except Exception as e:
            db_status = {"status": "error", "error": str(e)}

    overall = "ok"
    if db_status.get("status") == "error":
        overall = "degraded"

    return JSONResponse({
        "status": overall,
        "uptime_seconds": round(uptime, 2),
        "components": {
            "database": db_status,
            "metrics": _metrics_collector.snapshot(),
        },
    })


@app.get("/api/metrics")
async def get_metrics() -> JSONResponse:
    """Return MetricsCollector snapshot."""
    return JSONResponse(_metrics_collector.snapshot())


# ---------------------------------------------------------------------------
# Experiment API endpoints (Phase 3)
# ---------------------------------------------------------------------------


@app.get("/api/experiments")
async def get_experiments() -> JSONResponse:
    """Return all experiments from the board."""
    manager._try_load_db()
    if manager.db is None:
        return JSONResponse({"experiments": []})
    experiments = manager.db.list_all()
    return JSONResponse({
        "experiments": [
            {
                "id": exp.id,
                "name": exp.name,
                "description": exp.description,
                "hypothesis": exp.hypothesis,
                "status": exp.status,
                "config_json": exp.config_json,
                "worker_id": exp.worker_id,
                "slurm_job_id": exp.slurm_job_id,
                "results_json": exp.results_json,
                "error": exp.error,
                "debrief_path": exp.debrief_path,
                "created_at": exp.created_at,
                "updated_at": exp.updated_at,
                "started_at": exp.started_at,
                "finished_at": exp.finished_at,
            }
            for exp in experiments
        ],
        "summary": manager.db.board_summary(),
    })


@app.get("/api/experiments/{exp_id}")
async def get_experiment(exp_id: int) -> JSONResponse:
    """Return a single experiment's details."""
    manager._try_load_db()
    if manager.db is None:
        return JSONResponse({"error": "No experiment database"}, status_code=404)
    exp = manager.db.get(exp_id)
    if exp is None:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)
    return JSONResponse({
        "id": exp.id,
        "name": exp.name,
        "description": exp.description,
        "hypothesis": exp.hypothesis,
        "status": exp.status,
        "config_json": exp.config_json,
        "worker_id": exp.worker_id,
        "slurm_job_id": exp.slurm_job_id,
        "results_json": exp.results_json,
        "error": exp.error,
        "debrief_path": exp.debrief_path,
        "created_at": exp.created_at,
        "updated_at": exp.updated_at,
        "started_at": exp.started_at,
        "finished_at": exp.finished_at,
    })


@app.post("/api/experiments/{exp_id}/cancel")
async def cancel_experiment(exp_id: int) -> JSONResponse:
    """Prune idle work; active processes remain owned by the research runner."""
    manager._try_load_db()
    if manager.db is None:
        return JSONResponse({"error": "No experiment database"}, status_code=404)
    exp = manager.db.get(exp_id)
    if exp is None:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)
    terminal = {"done", "cancelled", "analyzed"}
    if exp.status in terminal:
        return JSONResponse(
            {"error": f"Cannot cancel experiment in '{exp.status}' state"},
            status_code=409,
        )
    if exp.status in {"queued", "running"} or exp.worker_id is not None:
        return JSONResponse(
            {"error": "Work is already in progress. Cancel it from the research runner; the viewer cannot stop it."},
            status_code=409,
        )
    try:
        if not manager.db.cancel_if_idle(exp_id):
            return JSONResponse(
                {"error": "Experiment state changed. Refresh the board and cancel active work from the research runner."},
                status_code=409,
            )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"status": "cancelled", "id": exp_id})


@app.get("/api/status-report")
async def get_status_report() -> JSONResponse:
    """Generate and return a structured status report."""
    if not _workspace:
        return JSONResponse({"error": "No workspace"}, status_code=400)
    try:
        from alpha_lab.output_generator import OutputGenerator
        gen = OutputGenerator(_workspace, adapter=_adapter_cache.get("adapter"))
        report = gen.generate_status_report()
        return JSONResponse(report)
    except Exception as e:
        logger.error(f"Status report generation failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/leaderboard")
async def get_leaderboard() -> JSONResponse:
    """Return top experiments by primary metric."""
    manager._try_load_db()
    if manager.db is None:
        return JSONResponse({"leaderboard": []})
    import json as _json

    # Use adapter metric if available
    _metric = "sharpe"
    _direction = "maximize"
    _display_name = "Sharpe"
    _adapter = _adapter_cache.get("adapter")
    if _adapter is not None:
        _metric = _adapter.metric.primary_metric
        _direction = _adapter.metric.direction
        _display_name = _adapter.metric.display_name

    leaders = manager.db.leaderboard(_metric, 20)

    def _safe_metrics(results_json: str | None) -> dict:
        if not results_json:
            return {}
        try:
            return _json.loads(results_json)
        except (ValueError, TypeError):
            return {}

    return JSONResponse({
        "metric": _metric,
        "direction": _direction,
        "display_name": _display_name,
        "leaderboard": [
            {
                "id": exp.id,
                "name": exp.name,
                "status": exp.status,
                "metrics": _safe_metrics(exp.results_json),
            }
            for exp in leaders
        ],
    })


# ---------------------------------------------------------------------------
# Explicit public research-question chat
# ---------------------------------------------------------------------------


@app.get("/api/chat-config")
async def get_chat_config() -> JSONResponse:
    if _chat_session is None:
        return JSONResponse({"enabled":False,"workspace_context":False,
                             "reason":_chat_configuration_error or "External research chat is not configured."})
    return JSONResponse(_chat_session.describe())


@app.post("/api/chat")
async def chat_query(request: Request) -> JSONResponse:
    """One consented public question; no automatic workspace context or history."""
    raw=await request.body()
    if len(raw)>131_072:
        return JSONResponse({"error":"Question request exceeds the HTTP body limit."},status_code=413)
    try:
        body=json.loads(raw)
    except (ValueError,UnicodeDecodeError):
        return JSONResponse({"error":"Expected a public-question JSON request."},status_code=400)
    if not isinstance(body,dict) or set(body)!={"question","consent_to_send"}:
        return JSONResponse({"error":"Only question and consent_to_send are accepted; workspace context is disabled."},status_code=400)
    if body["consent_to_send"] is not True:
        return JSONResponse({"error":"Confirm sending this public question to the configured endpoint."},status_code=400)
    session=_chat_session
    if session is None:
        return JSONResponse({"error":"External research chat is disabled. Configure its model and limits explicitly."},status_code=503)
    from alpha_lab.dashboard_chat import ChatError
    task=asyncio.create_task(asyncio.to_thread(session.answer,body["question"]))
    _chat_request_tasks.add(task)
    try:
        answer=await task
        return JSONResponse({"answer":answer,"model":session.settings.model,
                             "reasoning_effort":session.settings.reasoning_effort,"workspace_context":False})
    except ChatError as error:
        return JSONResponse({"error":str(error)},status_code=error.status_code)
    except Exception as error:
        logger.warning("Research chat failed: %s",type(error).__name__)
        return JSONResponse({"error":"Research chat failed; no automatic retry."},status_code=502)
    finally:
        _chat_request_tasks.discard(task)


# ---------------------------------------------------------------------------
# Static files (must be last)
# ---------------------------------------------------------------------------

from alpha_lab.research.web import research_router
app.include_router(research_router(lambda: _research_root or _workspace))

_frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def serve_main() -> None:
    """CLI entry point for the web dashboard server.

    This server is a passive viewer — it does not start or stop the pipeline.
    Run the pipeline separately via ``python run.py``, then point this server
    at the same workspace to monitor progress.
    """
    parser = argparse.ArgumentParser(
        prog="alpha-lab-serve",
        description="Alpha Lab Web Dashboard — passive viewer for a running or completed pipeline",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Workspace directory path to monitor",
    )
    parser.add_argument(
        "--chat-config",
        type=Path,
        help="Optional explicit public-question chat controls; disabled when omitted",
    )
    parser.add_argument('--research-root',type=Path,help='Optional saved US equity study library for /research')
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    global _workspace, _chat_config_path, _research_root
    _workspace = os.path.abspath(args.workspace)
    _chat_config_path = args.chat_config.resolve() if args.chat_config else None
    _research_root = str(args.research_root.resolve()) if args.research_root else None
    manager.workspace = _workspace

    print(f"Workspace: {_workspace}")

    Path(_workspace).mkdir(parents=True, exist_ok=True)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    serve_main()
