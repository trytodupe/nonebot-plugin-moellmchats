from collections.abc import Iterable
from typing import Any


def is_allowed_group(group_id: Any, allowed_group_ids: Iterable[Any]) -> bool:
    if group_id is None:
        return False
    normalized_ids = {str(value).strip() for value in allowed_group_ids if str(value).strip()}
    return str(group_id).strip() in normalized_ids


def contains_doubao_help_trigger(text: str) -> bool:
    return "豆包" in text.lstrip()[:5]


def should_trigger_group_chat(has_at_mention: bool, text: str) -> bool:
    return has_at_mention or contains_doubao_help_trigger(text)
