"""Hermes setup and picker hooks for the Antigravity provider.

Hermes asks an external-process provider two questions outside inference:
"is it installed and signed in, and what does this account have?" (``setup_status``)
and "what can I offer in the model picker?" (``discover_models``). Both are
answered here from the CLI itself, with no inference request and no tokens
spent: ``agy models`` for the catalog, and either the OAuth token file or
``agy usage`` (a free read) for the sign-in check.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)

INSTALL_HINT = "Install the Antigravity CLI (`agy`) and sign in once to use this provider."
LOGIN_HINT = "Run `agy` once in a terminal and finish the Google sign-in, then retry."


def _helpers():
    try:
        from .process import _no_window, resolve_agy_command, resolve_real_token_path
    except ImportError:
        from process import _no_window, resolve_agy_command, resolve_real_token_path  # type: ignore[no-redef]
    return _no_window, resolve_agy_command, resolve_real_token_path


def _usage_probe(timeout: float) -> Optional[str]:
    """Run the CLI's free `usage` read; return its raw response or None."""
    _no_window, resolve_agy_command, _token = _helpers()
    try:
        run = subprocess.run(
            [resolve_agy_command(), "usage"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            **_no_window(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (run.stdout or "").strip()
    if not text:
        return None
    # A signed-out CLI answers with an auth error rather than a usage table.
    lowered = text.lower()
    if "not logged" in lowered or "sign in" in lowered or "authentication" in lowered:
        return None
    return text


def _quota_detail(raw: str) -> str:
    """One line from the usage table, e.g. `Gemini (5h): 62% remaining`."""
    best = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if "remaining" in line.lower():
            best = line.replace("\t", " ").strip()
            if "5h" in line.lower() or "5 hour" in line.lower():
                return best
    return best


def setup_status(command: Any = None, env: Optional[dict[str, str]] = None, timeout: float = 15.0) -> dict[str, Any]:
    """``{available, logged_in, plan, detail, login_command}`` from the CLI itself."""
    _no_window, resolve_agy_command, resolve_real_token_path = _helpers()
    try:
        command = command or resolve_agy_command()
    except Exception:
        command = None
    if not command:
        return {
            "available": False,
            "logged_in": False,
            "plan": "",
            "detail": INSTALL_HINT,
            "login_command": None,
        }
    # File-based sign-in is the norm on a headless host; a keyring-only token
    # returns None here and is confirmed by the free usage read below.
    logged_in = False
    try:
        logged_in = bool(resolve_real_token_path())
    except Exception:
        logged_in = False
    detail = ""
    if not logged_in:
        raw = _usage_probe(timeout)
        if raw:
            logged_in = True
            detail = _quota_detail(raw)
    else:
        raw = _usage_probe(timeout)
        if raw:
            detail = _quota_detail(raw)
        else:
            logged_in = False
            detail = LOGIN_HINT
    return {
        "available": True,
        "logged_in": logged_in,
        "plan": "Antigravity",
        "detail": detail if logged_in else (detail or LOGIN_HINT),
        # The CLI signs in interactively; there is no non-interactive login verb.
        "login_command": None,
    }


def discover_models(timeout: float = 15.0) -> Optional[list[dict[str, str]]]:
    """The account's live picker as ``[{id, label, note}]``, or None."""
    try:
        from .models import load_catalog_rows
    except ImportError:
        from models import load_catalog_rows  # type: ignore[no-redef]
    try:
        rows = load_catalog_rows(timeout=timeout)
    except Exception as exc:
        logger.debug("Antigravity model discovery failed: %s", exc)
        return None
    return rows or None


def resolve_aux_model(timeout: float = 15.0) -> str:
    """Cheapest live model id for auxiliary work, or "" when unknown.

    Auxiliary calls (titles, summaries, compaction helpers) do not need a
    frontier model. Prefer the small flash tier, then the low Neon tier, and
    only then anything Gemini 4 — always as ``<base>-<effort>``, the form agy
    accepts on ``--model``.
    """
    try:
        from .models import load_catalog_rows
    except ImportError:
        from models import load_catalog_rows  # type: ignore[no-redef]
    try:
        rows = load_catalog_rows(timeout=timeout)
    except Exception:
        return ""
    by_id = {row["id"]: row for row in rows}
    for base in ("gemini-3.8-flash", "gemini-4-neon", "gemini-4-hydrogen"):
        row = by_id.get(base)
        if row is None:
            continue
        efforts = [e.strip() for e in (row.get("note") or "").split(",") if e.strip()]
        if "low" in efforts:
            return f"{base}-low"
        return base
    for row in rows:
        if row["id"].startswith("gemini-4-"):
            return row["id"]
    return ""


def default_vision_model(timeout: float = 15.0) -> Optional[str]:
    """A model id that accepts image attachments natively, or None."""
    try:
        from .models import load_catalog_rows
    except ImportError:
        from models import load_catalog_rows  # type: ignore[no-redef]
    try:
        rows = load_catalog_rows(timeout=timeout)
    except Exception:
        return None
    ids = [row["id"] for row in rows]
    for candidate in ("gemini-4-neon", "gemini-4-hydrogen", "gemini-4-carbon", "gemini-3.8-flash"):
        if candidate in ids:
            return candidate
    return next((model for model in ids if model.startswith("gemini")), None)


def auto_compact_override() -> str:
    """Value for ``AGY_CLI_AUTO_COMPACT_OVERRIDE`` unless the user set one.

    Hermes compacts at its own threshold (half the declared window at the
    default 0.5, with a 0.75 floor below 1M), which lands well before 90% of
    the model's window. Keeping agy's internal compaction above Hermes's means
    agy never discards history behind Hermes's back; it stays the backstop for
    a session Hermes could not compact in time.
    """
    explicit = os.environ.get("AGY_CLI_AUTO_COMPACT_OVERRIDE")
    if explicit:
        return explicit
    return os.environ.get("ANTIGRAVITY_AUTO_COMPACT") or "90%"
