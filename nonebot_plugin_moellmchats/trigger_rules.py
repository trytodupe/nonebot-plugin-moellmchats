def contains_doubao_help_trigger(text: str) -> bool:
    return "豆包" in text.lstrip()[:5]


def should_trigger_group_chat(has_at_mention: bool, text: str) -> bool:
    return has_at_mention or contains_doubao_help_trigger(text)
