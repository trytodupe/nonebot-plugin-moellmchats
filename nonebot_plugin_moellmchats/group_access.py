from collections import OrderedDict
from typing import Any

from nonebot import get_driver
from nonebot.adapters.onebot.v11 import Bot, MessageEvent
from nonebot.log import logger

_EVENT_CACHE_LIMIT = 1024
_event_access_cache: OrderedDict[tuple[str, int, int], bool] = OrderedDict()


def _configured_superusers() -> set[str]:
    configured = getattr(get_driver().config, "superusers", set()) or set()
    return {str(user_id).strip() for user_id in configured if str(user_id).strip()}


def _event_cache_key(bot: Bot, event: Any) -> tuple[str, int, int] | None:
    message_id = getattr(event, "message_id", None)
    if message_id is None:
        return None
    return str(bot.self_id), int(event.group_id), int(message_id)


def _cache_result(key: tuple[str, int, int] | None, allowed: bool) -> None:
    if key is None:
        return
    _event_access_cache[key] = allowed
    _event_access_cache.move_to_end(key)
    while len(_event_access_cache) > _EVENT_CACHE_LIMIT:
        _event_access_cache.popitem(last=False)


async def group_has_superuser(bot: Bot, event: MessageEvent) -> bool:
    if not hasattr(event, "group_id"):
        return False

    cache_key = _event_cache_key(bot, event)
    if cache_key in _event_access_cache:
        return _event_access_cache[cache_key]

    superusers = _configured_superusers()
    if not superusers:
        _cache_result(cache_key, False)
        return False

    try:
        members = await bot.get_group_member_list(group_id=event.group_id)
    except Exception:
        logger.warning(f"Failed to check superuser membership for group {event.group_id}")
        _cache_result(cache_key, False)
        return False

    member_ids = {str(member.get("user_id")).strip() for member in members if member.get("user_id") is not None}
    allowed = bool(superusers & member_ids)
    _cache_result(cache_key, allowed)
    return allowed
