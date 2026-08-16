from collections.abc import Awaitable
from typing import Any, Callable, Optional

from nonebot.log import logger
from nonebot.plugin import get_loaded_plugins, get_plugin

_PLUGIN_NAME = "group_superuser_gate"
_EXPECTED_INTERFACE_VERSION = 2
_group_gate: Optional[Callable[[Any, Any], Awaitable[bool]]] = None


def _find_gate_plugin():
    plugin = get_plugin(_PLUGIN_NAME)
    if plugin is not None:
        return plugin
    return next(
        (
            candidate
            for candidate in get_loaded_plugins()
            if candidate.name == _PLUGIN_NAME or candidate.module_name.rsplit(".", 1)[-1] == _PLUGIN_NAME
        ),
        None,
    )


def configure_group_gate(mode="auto"):
    global _group_gate

    configured_mode = str(mode or "auto").strip().lower()
    if configured_mode not in {"auto", "required", "off"}:
        raise RuntimeError(f"Invalid moellmchats group_gate_mode: {configured_mode!r}")

    _group_gate = None
    if configured_mode == "off":
        logger.info("Moellmchats group gate is disabled")
        return

    plugin = _find_gate_plugin()
    if plugin is None:
        if configured_mode == "required":
            raise RuntimeError("Moellmchats requires the group_superuser_gate plugin, but it is not loaded")
        logger.info("Moellmchats group gate is unavailable; continuing in auto mode")
        return

    interface_version = getattr(plugin.module, "GROUP_SUPERUSER_GATE_INTERFACE_VERSION", None)
    gate = getattr(plugin.module, "event_access_allowed", None)
    if interface_version != _EXPECTED_INTERFACE_VERSION or not callable(gate):
        raise RuntimeError(
            "Loaded group_superuser_gate has an incompatible interface "
            f"(expected {_EXPECTED_INTERFACE_VERSION}, got {interface_version!r})"
        )

    _group_gate = gate
    logger.success(f"Moellmchats group gate active: provider={plugin.module_name}, interface={interface_version}")


async def group_has_superuser(bot, event):
    if not hasattr(event, "group_id"):
        return False
    return await event_access_allowed(bot, event)


async def event_access_allowed(bot, event):
    if _group_gate is None:
        return True
    return bool(await _group_gate(bot, event))


__all__ = ["configure_group_gate", "event_access_allowed", "group_has_superuser"]
