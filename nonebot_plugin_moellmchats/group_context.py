"""Optional runtime context supplied by other NoneBot plugins.

Providers are synchronous snapshots: they must not perform network or disk I/O.
The chat plugin owns prompt assembly; providers own their scope and facts.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

GroupContextProvider = Callable[[Any, Any], Optional[str]]
_providers: dict[str, GroupContextProvider] = {}
_logger = logging.getLogger(__name__)


def register_group_context_provider(name: str, provider: GroupContextProvider) -> None:
    """Register or replace a named provider (safe on plugin reload)."""
    _providers[name] = provider


def collect_group_context(bot: Any, event: Any) -> str:
    """Collect scoped context without allowing a broken provider to break chat."""
    if getattr(event, "group_id", None) is None:
        return ""
    parts = []
    for name, provider in tuple(_providers.items()):
        try:
            context = provider(bot, event)
        except Exception:
            _logger.exception("Group context provider %s failed", name)
            continue
        if isinstance(context, str) and context.strip():
            parts.append(context.strip()[:8192])
    return "\n\n".join(parts)
