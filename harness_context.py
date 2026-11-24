"""Ask the agy harness for each model's context window, and cache the answer.

``agy`` knows the window: its Agent Client Protocol surface reports it. A
``session/new`` response carries ``models.availableModels`` and
``currentModelId``, and every model switch emits

    {"sessionUpdate": "usage_update", "used": 0, "size": 1048576}

where ``size`` is that model's context window. Verified against agy 1.5.15 on
2026-11-23: gemini-4-neon-* → 1,048,576, claude-opus-5-5-* → 1,000,000, both
matching the published model cards.

Discovery costs no tokens and no conversation: it opens a session against an
isolated home (the same one the workers use), walks the model list with
``session/set_model``, closes the session and exits. It runs in a background
thread the first time a declared window is needed and the cache is stale, so a
turn never waits on it; until it lands, :mod:`agy_context` serves its static
fallback.

Cache: ``$HERMES_HOME/cache/agy_context/windows.json``, keyed by model id, with
the agy version that reported it. A new agy build invalidates it.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 7 * 24 * 3600
_DISCOVERY_TIMEOUT = 60.0
_STEP_TIMEOUT = 8.0

_refresh_lock = threading.Lock()
_refreshing = False
_memo: dict[str, Any] = {"fetched_at": 0.0, "windows": {}, "mtime": None, "path": None}


def _hermes_home() -> str:
    try:
        from hermes_constants import get_hermes_home

        return str(get_hermes_home())
    except ImportError:
        return os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")


def cache_path() -> Path:
    override = os.environ.get("HERMES_ANTIGRAVITY_WINDOWS_CACHE") or os.environ.get("ANTIGRAVITY_WINDOWS_CACHE")
    if override:
        return Path(override)
    return Path(_hermes_home()) / "cache" / "agy_context" / "windows.json"


def agy_version() -> str:
    try:
        from process import resolve_agy_command  # plugin package import

        cmd = resolve_agy_command()
    except Exception:  # pragma: no cover - standalone/dev layouts
        cmd = os.environ.get("ANTIGRAVITY_COMMAND") or "agy"
    try:
        out = subprocess.run(
            [cmd, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
        return (out.stdout or out.stderr or "").strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #


def _read_cache_file(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("windows"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def load_cached_windows(*, max_age: float = CACHE_TTL_SECONDS) -> dict[str, int]:
    """Windows from the cache file, if it is fresh and from this agy build.

    Re-read only when the file's mtime changes; ``get_model_context_length`` can
    run on every turn, so this must stay a stat plus a memo hit.
    """
    path = cache_path()
    try:
        st = path.stat()
    except OSError:
        return {}
    if _memo.get("path") != str(path) or _memo.get("mtime") != st.st_mtime:
        data = _read_cache_file(path)
        windows = {}
        for model, value in (data.get("windows") or {}).items():
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                windows[str(model).strip().lower()] = value
        _memo.update(
            fetched_at=float(data.get("fetched_at") or 0.0),
            windows=windows,
            mtime=st.st_mtime,
            path=str(path),
            agy=str(data.get("agy_version") or ""),
        )
    if _memo.get("agy") and _memo["agy"] != agy_version():
        return {}
    if max_age and _memo.get("fetched_at") and time.time() - float(_memo["fetched_at"]) > max_age:
        return {}
    return dict(_memo.get("windows") or {})


def save_cached_windows(windows: dict[str, int]) -> None:
    path = cache_path()
    payload = {
        "fetched_at": time.time(),
        "agy_version": agy_version(),
        "source": "agy acp session/set_model usage_update.size",
        "windows": {str(k).lower(): int(v) for k, v in windows.items() if int(v) > 0},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:  # pragma: no cover - disk problems are not worth a turn
        logger.debug("could not write the agy window cache: %s", exc)


# --------------------------------------------------------------------------- #
# ACP discovery
# --------------------------------------------------------------------------- #


class _AcpSession:
    """Minimal ACP client: request/response by id, plus session/update events."""

    def __init__(self, command: str, env: dict[str, str], cwd: str, timeout: float = _DISCOVERY_TIMEOUT):
        self._events: list[dict[str, Any]] = []
        self._pending: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._next_id = 0
        self.proc = subprocess.Popen(
            [command, "acp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.deadline = time.monotonic() + timeout

    def _read_loop(self) -> None:
        proc = self.proc
        if not proc.stdout:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                with self._lock:
                    waiter = self._pending.pop(msg["id"], None)
                if waiter is not None:
                    waiter["response"] = msg
                    waiter["done"].set()
                continue
            if msg.get("method") == "session/update":
                params = msg.get("params") or {}
                update = params.get("update") or {}
                if isinstance(update, dict):
                    self._events.append({"sessionId": params.get("sessionId"), **update})

    def request(self, method: str, params: dict[str, Any], timeout: float = _STEP_TIMEOUT) -> Optional[dict[str, Any]]:
        if time.monotonic() > self.deadline:
            return None
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            waiter = {"done": threading.Event(), "response": None}
            self._pending[request_id] = waiter
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            return None
        if not waiter["done"].wait(timeout):
            return None
        response = waiter["response"] or {}
        return response.get("result") if isinstance(response, dict) else None

    def wait_usage_update(self, session_id: str, timeout: float = 3.0) -> Optional[int]:
        """Block until a usage_update for *session_id* arrives; its ``size``.

        The update is an asynchronous session event, not the response to
        ``session/set_model``, so it can land just before or just after the
        response. Polling the drained list is the honest way to pair them.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                for index, event in enumerate(self._events):
                    if event.get("sessionUpdate") != "usage_update":
                        continue
                    if event.get("sessionId") != session_id:
                        continue
                    raw = event.get("size")
                    del self._events[: index + 1]
                    try:
                        value = int(raw)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        return None
                    return value if value > 0 else None
            time.sleep(0.05)
        return None

    def drain_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = self._events[:]
            self._events.clear()
        return events

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            with __import__("contextlib").suppress(Exception):
                self.proc.kill()


_EFFORT_SUFFIXES = ("-xhigh", "-high", "-medium", "-low")


def _split_effort(model_id: str) -> tuple[str, str]:
    """(base, effort) for an agy model id; ``xhigh`` folds into ``high``."""
    for suffix in _EFFORT_SUFFIXES:
        if model_id.endswith(suffix):
            effort = suffix[1:]
            return model_id[: -len(suffix)], ("high" if effort == "xhigh" else effort)
    return model_id, ""


def _session_size(events: list[dict[str, Any]], session_id: str) -> Optional[int]:
    """Last usage_update size reported for *session_id*."""
    size = None
    for event in events:
        if event.get("sessionUpdate") == "usage_update" and event.get("sessionId") == session_id:
            raw = event.get("size")
            if raw is None:
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                size = value
    return size


def discover_windows(timeout: float = _DISCOVERY_TIMEOUT) -> dict[str, int]:
    """Ask the harness for every model's window. Returns {} on any failure."""
    try:
        from process import build_child_env, resolve_agy_command, setup_isolated_home, sync_isolated_token
    except ImportError:  # pragma: no cover - standalone layouts
        try:
            from .process import build_child_env, resolve_agy_command, setup_isolated_home, sync_isolated_token
        except ImportError:
            logger.debug("agy window discovery skipped: process helpers unavailable")
            return {}

    command = resolve_agy_command()
    isolated_home = None
    try:
        isolated_home, _gemini_dir = setup_isolated_home(Path(os.environ.get("TMPDIR", "/tmp")))
        sync_isolated_token(isolated_home)
        env = build_child_env(isolated_home)
    except Exception as exc:
        logger.debug("agy window discovery skipped: isolated home failed (%s)", exc)
        return {}

    session = None
    windows: dict[str, int] = {}
    try:
        session = _AcpSession(command, env, str(isolated_home), timeout=timeout)
        if not session.request(
            "initialize",
            {"protocolVersion": 1, "clientCapabilities": {}},
            timeout=min(_STEP_TIMEOUT, timeout),
        ):
            logger.debug("agy window discovery: initialize failed")
            return {}
        created = session.request("session/new", {"cwd": str(isolated_home), "mcpServers": []})
        if not created:
            logger.debug("agy window discovery: session/new failed")
            return {}
        session_id = str(created.get("sessionId") or "")
        models = (created.get("models") or {}) if isinstance(created.get("models"), dict) else {}
        available = [str(m.get("modelId")) for m in (models.get("availableModels") or []) if isinstance(m, dict) and m.get("modelId")]
        current = str(models.get("currentModelId") or "")
        current_size = session.wait_usage_update(session_id, timeout=5.0)
        # agy announces the window when a switch changes it and stays silent when
        # it does not, so discovery walks BASES (one representative variant each)
        # and carries the last announced size across a silent switch. Every
        # variant of a base shares its window; the effort suffix is not a model.
        last_size = current_size
        if current and current_size:
            windows[current] = current_size
            windows[_split_effort(current)[0]] = current_size
        bases: dict[str, str] = {}
        for model_id in available:
            base, effort = _split_effort(model_id)
            if base not in bases or (effort == "high"):
                bases[base] = model_id
        if current:
            current_base, _ = _split_effort(current)
            bases.pop(current_base, None)
        for base, representative in bases.items():
            if time.monotonic() > session.deadline:
                break
            session.drain_events()
            session.request("session/set_model", {"sessionId": session_id, "modelId": representative})
            size = session.wait_usage_update(session_id, timeout=3.0)
            if size:
                last_size = size
            elif last_size:
                size = last_size
            if size:
                windows[base] = size
                windows[representative] = size
        session.request("session/close", {"sessionId": session_id}, timeout=3)
    except Exception as exc:  # pragma: no cover - never break a turn
        logger.debug("agy window discovery failed: %s", exc)
    finally:
        if session is not None:
            session.close()
        if isolated_home is not None:
            try:
                import shutil

                shutil.rmtree(isolated_home, ignore_errors=True)
            except Exception:
                pass
    # A variant's value covers its base: `gemini-4-neon-high` (1M) is also the
    # window of `gemini-4-neon`, which may never be the current model itself.
    for model_id, size in list(windows.items()):
        base, _effort = _split_effort(model_id)
        windows.setdefault(base, size)
    if windows:
        logger.info("agy reported context windows for %d models: %s", len(windows), dict(sorted(windows.items())[:4]))
    return windows


def refresh_in_background(*, force: bool = False) -> None:
    """Kick one discovery run without blocking the caller."""
    global _refreshing
    if not force and load_cached_windows():
        return
    with _refresh_lock:
        if _refreshing:
            return
        _refreshing = True

    def _run() -> None:
        global _refreshing
        try:
            windows = discover_windows()
            if windows:
                save_cached_windows(windows)
        finally:
            with _refresh_lock:
                _refreshing = False

    threading.Thread(target=_run, name="agy-window-discovery", daemon=True).start()


def lookup(model: str) -> Optional[int]:
    """Cached harness window for *model* (longest prefix), or None."""
    name = (model or "").strip().lower()
    if not name:
        return None
    windows = load_cached_windows()
    if not windows:
        return None
    if name in windows:
        return windows[name]
    # An effort variant shares its base model's window; try the base early so a
    # cold cache keyed only by another variant still answers.
    base, _effort = _split_effort(name)
    if base != name and base in windows:
        return windows[base]
    best: tuple[int, int] | None = None
    for cached, size in windows.items():
        if name.startswith(cached) and (best is None or len(cached) > best[0]):
            best = (len(cached), size)
    if best is not None:
        return best[1]
    # A base id asked for before any variant of it was the current model: take
    # the window of the closest variant that extends the name.
    variants = [(cached, size) for cached, size in windows.items() if cached.startswith(name)]
    if variants:
        shortest = min(variants, key=lambda item: len(item[0]))
        return shortest[1]
    return None


def reset_memo() -> None:
    """Forget the in-process cache view (tests)."""
    _memo.update(fetched_at=0.0, windows={}, mtime=None, path=None)
