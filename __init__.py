"""Antigravity Subscription DirectSDK provider plugin for Hermes Agent."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import fields
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

try:
    from .models import (
        _FALLBACK_MODELS,
        _MODEL_ALIASES,
        _cached_catalog,
        _heuristic_efforts,
        load_catalog,
        model_efforts,
        resolve_model_alias,
        split_effort_suffix,
    )
except ImportError:
    # Loaded outside a package (e.g. a flat source tree under test):
    # the absolute name is the same module.
    from models import (
        _FALLBACK_MODELS,
        _MODEL_ALIASES,
        _cached_catalog,
        _heuristic_efforts,
        load_catalog,
        model_efforts,
        resolve_model_alias,
        split_effort_suffix,
    )

logger = logging.getLogger(__name__)

# Context accounting lives in agy_context: the single-prompt-frame cap is a
# measured BYTE limit (191,583 bytes on agy 1.5.15), not a per-family token
# guess, and continuation turns send only a delta through --conversation. The
# declared window is therefore the trajectory checkpoint for every family, and
# the frame cap is enforced where the prompt is built (prompt.py).
try:  # plugin package import
    from .agy_context import TRAJECTORY_CONTEXT as _DEFAULT_CONTEXT_LENGTH
except ImportError:  # standalone import (tests, plugin dev)
    from agy_context import TRAJECTORY_CONTEXT as _DEFAULT_CONTEXT_LENGTH  # type: ignore[no-redef]
_CONTEXT_LENGTH_OVERRIDE_VARS = (
    "HERMES_ANTIGRAVITY_CONTEXT_LENGTHS",
    "ANTIGRAVITY_CONTEXT_LENGTHS",
)


def _context_length_overrides() -> dict[str, int]:
    """Parse per-model context overrides from ``*_CONTEXT_LENGTHS`` JSON env vars.

    ``HERMES_ANTIGRAVITY_CONTEXT_LENGTHS`` wins over ``ANTIGRAVITY_CONTEXT_LENGTHS``.
    Values must be positive integers; malformed entries are ignored rather than
    failing the request path.
    """
    for var in _CONTEXT_LENGTH_OVERRIDE_VARS:
        raw = os.environ.get(var)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except ValueError:
            logger.warning("Ignoring malformed %s (not valid JSON)", var)
            continue
        if isinstance(parsed, dict):
            out: dict[str, int] = {}
            for name, val in parsed.items():
                if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                    out[str(name).strip().lower()] = val
            if out:
                return out
    return {}


def resolve_context_length(model: str) -> int:
    """Declared context for *model*: the multi-turn trajectory checkpoint.

    One number for every family, because the binding limit is not per-model any
    more. A live conversation persists through ``agy --conversation`` and each
    turn sends only its delta, so the limit a session can grow to is agy's
    compiled trajectory checkpoint (239,616 tokens) less the 16k output
    reserve. The single-prompt-frame byte cap (191,583 bytes measured on agy
    1.5.15) is enforced in :mod:`prompt` by fitting each frame to a budget, so a
    full send can no longer be silently cut by the transport.

    Overrides still win, exact name first and then the effort-stripped family
    name, so a pinned number in ``*_CONTEXT_LENGTHS`` overrides this.
    """
    base, _effort = split_effort_suffix(model.strip().lower())
    overrides = _context_length_overrides()
    for name in (model.strip().lower(), base):
        if name and name in overrides:
            return overrides[name]
    return _DEFAULT_CONTEXT_LENGTH


class AntigravitySubscriptionDirectSDKProfile(ProviderProfile):
    """Google Antigravity Subscription DirectSDK provider profile."""

    def create_client(self, **client_kwargs: Any) -> Any:
        """Create the Antigravity client facade."""
        from .client import AntigravityClient

        return AntigravityClient(**client_kwargs)

    def supported_reasoning_efforts(
        self, model: str | None
    ) -> tuple[str, ...] | None:
        """Declared reasoning-effort vocabulary for models on this provider.

        Enables Hermes /model picker and /reasoning commands to offer appropriate
        thinking effort options for reasoning-capable models. The levels come from
        `agy models`, so they follow what agy accepts per model: Gemini Flash and
        Claude take low/medium/high, Gemini Pro low/high, gpt-oss medium only.
        """
        name = (model or "").strip()
        catalog = _cached_catalog()
        resolved_name = resolve_model_alias(name, catalog)
        base, suffix_effort = split_effort_suffix(resolved_name)
        # No process spawn here: use the cached catalog and the built-in table.
        efforts = model_efforts(base, refresh=False)
        if efforts is None:
            efforts = _heuristic_efforts(base, suffix_effort is not None)
        return efforts

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract reasoning effort and forward as top-level api_kwargs to AntigravityClient."""
        effort = None
        if isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                effort = "low"
            else:
                effort = reasoning_config.get("effort")
        top_level: dict[str, Any] = {}
        if effort:
            top_level["reasoning_effort"] = str(effort).strip().lower()
        return {}, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 15.0,
    ) -> list[str] | None:
        """Query `agy models` and normalize to clean, deduplicated base models.

        Separates model families from thinking efforts (e.g. `gemini-3.8-flash-{low,medium,high}`
        or `claude-opus-5-5-{low,medium,high}` becomes one entry), allowing Hermes' native
        reasoning effort picker to handle the thinking depth cleanly. The result also warms
        the effort table that `resolve_model_and_effort` reads.
        """
        catalog = load_catalog(timeout=timeout)
        if catalog:
            return list(catalog)
        return list(_FALLBACK_MODELS)

    def get_model_context_length(self, model: str) -> int | None:
        """Declared context window for Antigravity CLI.

        Returns the trajectory checkpoint (239,000 tokens) for every model: a
        live conversation sends only deltas through ``agy --conversation``, so
        that checkpoint, not the model's advertised window and not a
        per-family guess, is what Hermes should compact against. The transport
        does cap a *single* prompt frame at a measured 191,583 bytes, and that
        is handled where the prompt is assembled (:mod:`prompt` fits every
        frame to a byte budget and learns the real cap from any truncation agy
        reports), not by declaring a smaller window.

        Overrides: ``HERMES_ANTIGRAVITY_CONTEXT_LENGTH`` /
        ``ANTIGRAVITY_CONTEXT_LENGTH`` set one global value;
        ``HERMES_ANTIGRAVITY_CONTEXT_LENGTHS`` /
        ``ANTIGRAVITY_CONTEXT_LENGTHS`` take a JSON object mapping a model
        name or family prefix (e.g. ``"gemini-4-argon"``) to a token count.
        The global override wins over the per-model table. Prompt sizing:
        ``HERMES_ANTIGRAVITY_PROMPT_BYTES`` / ``ANTIGRAVITY_PROMPT_BYTES``.
        """
        env_val = os.environ.get("HERMES_ANTIGRAVITY_CONTEXT_LENGTH") or os.environ.get("ANTIGRAVITY_CONTEXT_LENGTH")
        if env_val:
            try:
                val = int(env_val.strip())
                if val > 0:
                    return val
            except ValueError:
                pass
        return resolve_context_length(model)

    def classify_api_error(
        self,
        error: Exception,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        message: str = "",
        body: Any = None,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        return _classify_antigravity_error(
            error,
            status_code=status_code,
            error_code=error_code,
            message=message,
            body=body,
            model=model,
        )

    def fetch_account_usage(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Fetch subscription quota snapshot for /usage, status bar, and desktop."""
        try:
            from .usage import get_account_usage_snapshot
        except ImportError:
            from usage import get_account_usage_snapshot

        return get_account_usage_snapshot(force_refresh=force_refresh)


def _classify_antigravity_error(
    error: Exception,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
    message: str = "",
    body: Any = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Classify agy CLI specific runtime errors so Hermes' smart failover /
    recovery pipeline triggers auto-compression, billing failover, or permanent auth abort.
    """
    err_str = f"{error} {message}".lower()
    if any(
        pattern in err_str
        for pattern in (
            "subscriber fell behind updates",
            "stalled for 5s",
            "empty result (status='success')",
            "empty result (status=\"success\")",
            "max_trajectory_tokens",
            "max trajectory tokens",
        )
    ):
        return {
            "reason": "context_overflow",
            "retryable": True,
            "should_compress": True,
        }
    if "exceeded the output token limit" in err_str or "previous response was cut off" in err_str:
        return {
            "reason": "server_error",
            "retryable": True,
            "should_compress": False,
        }
    if any(
        pattern in err_str
        for pattern in (
            "ai credits balance is too low",
            "credits balance is too low",
            "daily or billing cap",
            "billing cap",
        )
    ):
        return {
            "reason": "billing",
            "retryable": False,
            "should_fallback": True,
            "should_compress": False,
        }
    if any(
        pattern in err_str
        for pattern in (
            "verify your account to continue",
            "appeal a terms of service block",
            "terms of service block",
            "terms of service violation",
        )
    ):
        return {
            "reason": "auth_permanent",
            "retryable": False,
            "should_fallback": False,
            "should_compress": False,
        }
    return None


# ProviderProfile optional field opt-ins.
_PROFILE_EXTRA_FIELDS: dict[str, Any] = {}


class _LazyModelCapabilities(dict):
    """dict answering exact-model capability lookups without enumerating the catalog.

    The core reads this mapping only via ``.get(model, {})``
    (``agent/models_dev._provider_model_capabilities``), so a dynamic ``get``
    lets the plugin declare per-model vision on demand. Declaring
    ``modalities.input`` here makes ``decide_image_input_mode`` resolve
    ``native`` deterministically — offline, with no models.dev fetch and
    WITHOUT depending on the agy catalog: the family is visible from the
    model id's own prefix (``gemini-*`` / ``claude-*``), so a cold catalog
    cache (first turn after gateway start, TTL expiry, agy unavailable)
    cannot flip the verdict to ``text``. Effort-suffixed ids
    (``gemini-3.8-flash-high``) and future families (``gemini-4-*``,
    ``claude-5-*``) are covered by the same prefix rule. Alias ids are
    resolved as a best-effort second pass. Without this declaration the
    verdict flip-flops between native and text on identical turns depending
    on catalog cache state, which is exactly the multimodality flakiness
    this plugin must not have.
    """

    def get(self, model: Any, default: Any = None) -> Any:
        name = str(model or "").strip()
        if not name:
            return default
        verdict = self._declare(name.lower())
        if verdict is not None:
            return verdict
        # Alias ids ("flash", "opus"): resolve through the live catalog when
        # available and re-check the resolved family. Failure keeps the
        # fail-closed default — the core's own probes decide.
        try:
            resolved = resolve_model_alias(name, _cached_catalog()).lower()
            if resolved != name.lower():
                return self._declare(resolved)
        except Exception:
            pass
        return default

    @staticmethod
    def _declare(base: str) -> dict[str, Any] | None:
        base, _suffix = split_effort_suffix(base)
        if base.startswith("gemini"):
            # Canonical override schema (agent/models_dev._override_to_catalog_shape):
            # ``supports_vision`` is the key the core's vision derivation honors;
            # ``modalities`` is kept for raw readers.
            return {
                "supports_vision": True,
                "modalities": {"input": ["text", "image", "video"], "output": ["text"]},
            }
        if base.startswith("claude"):
            return {
                "supports_vision": True,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
            }
        # gpt-oss and unknown families: no declaration -> the core's own probes decide.
        return None


_PROFILE_FIELD_NAMES = {f.name for f in fields(ProviderProfile)}
if "supports_streaming" in _PROFILE_FIELD_NAMES:
    _PROFILE_EXTRA_FIELDS["supports_streaming"] = True
if "models_dev_id" in _PROFILE_FIELD_NAMES:
    _PROFILE_EXTRA_FIELDS["models_dev_id"] = "google"
if "model_capabilities" in _PROFILE_FIELD_NAMES:
    _PROFILE_EXTRA_FIELDS["model_capabilities"] = _LazyModelCapabilities()

antigravity_profile = AntigravitySubscriptionDirectSDKProfile(
    name="antigravity-subscription-directsdk",
    aliases=("antigravity", "agy", "antigravity-directsdk"),
    display_name="Antigravity Subscription DirectSDK",
    description="Use your Antigravity / Gemini subscription via the official agy CLI",
    base_url="agy://local",
    api_mode="chat_completions",
    auth_type="external_process",
    process_command="agy",
    process_args=("--output-format", "stream-json", "--disable-slash-commands"),
    process_command_env_vars=("ANTIGRAVITY_COMMAND", "AGY_CLI_PATH", "ANTIGRAVITY_CLI_PATH"),
    process_args_env_var="ANTIGRAVITY_ARGS",
    default_aux_model="gemini-3.8-flash",
    fallback_models=(),  # Account-scoped models: discovered dynamically via fetch_models() (agy models)
    supports_vision=True,
    unsupported_response_formats=("json_schema",),
    classify_api_error=_classify_antigravity_error,
    **_PROFILE_EXTRA_FIELDS,
)

register_provider(antigravity_profile)
