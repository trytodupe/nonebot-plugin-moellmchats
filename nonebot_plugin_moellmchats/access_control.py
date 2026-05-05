from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nonebot import get_driver
from nonebot.log import logger
from nonebot.plugin import get_loaded_plugins, get_plugin, get_plugin_by_module_name, load_plugin


ACCESS_CAPABILITY = "moellmchats.private_chat"
ACCESS_REQUEST_KEYWORD = "申请"
ACCESS_REQUEST_PROMPT = '发送“申请”来申请开启私聊功能。'


@dataclass(slots=True)
class AccessDecision:
    allowed: bool
    handled: bool = False
    reply_text: str | None = None


_access_request_service: Any | None = None


def _is_superuser(user_id: int | str) -> bool:
    driver = get_driver()
    superusers = getattr(driver.config, "superusers", set()) or set()
    return str(user_id) in {str(item).strip() for item in superusers if str(item).strip()}


def is_private_acl_exempt_user(user_id: int | str) -> bool:
    return _is_superuser(user_id)


def _is_access_request_loaded() -> bool:
    if get_plugin("access_request") is not None:
        return True
    return any(plugin.name == "access_request" for plugin in get_loaded_plugins())


def _resolve_access_request_service() -> Any | None:
    global _access_request_service

    if _access_request_service is not None:
        return _access_request_service

    try:
        plugin = get_plugin("access_request")
        if plugin is None:
            if not _is_access_request_loaded():
                plugin = load_plugin("access_request")
            if plugin is None:
                plugin = get_plugin("access_request")
        if plugin is None:
            plugin = get_plugin_by_module_name("src.plugins.access_request")
        if plugin is None:
            return None

        service_module = __import__(f"{plugin.module_name}.service", fromlist=["service"])
        _access_request_service = getattr(service_module, "service", None)
        if _access_request_service is None:
            logger.warning("access_request.service loaded but service singleton is missing")
        return _access_request_service
    except Exception:
        logger.info("access_request plugin is unavailable for moellmchats private ACL", exc_info=True)
        return None


def is_access_request_plugin_available() -> bool:
    return _resolve_access_request_service() is not None


async def evaluate_private_access(bot: Any, event: Any, plain_text: str) -> AccessDecision:
    user_id = int(getattr(event, "user_id"))
    access_request_service = _resolve_access_request_service()
    if access_request_service is None:
        return AccessDecision(allowed=False, handled=True, reply_text="当前环境未启用私聊访问控制插件。")

    if access_request_service.is_allowed(user_id, ACCESS_CAPABILITY):
        return AccessDecision(allowed=True)

    normalized_text = str(plain_text or "").strip()
    if normalized_text != ACCESS_REQUEST_KEYWORD:
        return AccessDecision(
            allowed=False,
            handled=True,
            reply_text=ACCESS_REQUEST_PROMPT,
        )

    request_record = access_request_service.request_access(
        user_id=user_id,
        request_text=normalized_text,
        capability=ACCESS_CAPABILITY,
    )
    if request_record.status == "approved":
        return AccessDecision(
            allowed=True,
            handled=True,
            reply_text="已开通私聊功能。",
        )

    sender = getattr(event, "sender", None)
    requester_name = (
        getattr(sender, "card", None)
        or getattr(sender, "nickname", None)
        or str(user_id)
    )
    await access_request_service.notify_primary_superuser(bot, request_record, requester_name)
    return AccessDecision(
        allowed=False,
        handled=True,
        reply_text="申请已提交，请等待审核。",
    )
