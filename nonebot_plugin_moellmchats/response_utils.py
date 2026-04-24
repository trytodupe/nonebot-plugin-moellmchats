import json
import unicodedata

VISUAL_LINE_WIDTH = 62.0
FORWARD_LINE_LIMIT = 45


def normalize_image_summary(summary: str) -> str:
    return " ".join(summary.strip().split())


def build_image_reference(summary: str) -> str:
    return f"[image:{normalize_image_summary(summary)}]"


def replace_image_placeholders(
    text: str, replacements: list[str], placeholder: str = "[图片]"
) -> str:
    if placeholder not in text:
        if replacements:
            return f"{text}\n" + "\n".join(replacements)
        return text

    parts = text.split(placeholder)
    result = [parts[0]]
    for index, replacement in enumerate(replacements):
        result.append(replacement)
        if index + 1 < len(parts):
            result.append(parts[index + 1])

    if len(replacements) + 1 < len(parts):
        result.extend(parts[len(replacements) + 1 :])
    return "".join(result)


def build_image_context_replacements(images: list[dict]) -> list[str]:
    replacements = []
    for image in images:
        if summary := image.get("summary"):
            replacements.append(build_image_reference(summary))
        else:
            replacements.append("[image]")
    return replacements


def format_text_with_image_context(text: str, images: list[dict]) -> str:
    return replace_image_placeholders(text, build_image_context_replacements(images))


def _char_visual_width(char: str) -> float:
    if char == "\n":
        return VISUAL_LINE_WIDTH
    if unicodedata.east_asian_width(char) in {"F", "W"}:
        return VISUAL_LINE_WIDTH / 15
    if char in "ilI.,'`|![](){}:; ":
        return 1.0
    if char.isascii():
        return VISUAL_LINE_WIDTH / 27
    return VISUAL_LINE_WIDTH / 20


def wrap_visual_lines(text: str, line_width: float = VISUAL_LINE_WIDTH) -> list[str]:
    lines = []
    current = []
    current_width = 0.0
    for char in text:
        if char == "\n":
            lines.append("".join(current))
            current = []
            current_width = 0.0
            continue
        char_width = _char_visual_width(char)
        if current and current_width + char_width > line_width:
            lines.append("".join(current))
            current = [char]
            current_width = char_width
        else:
            current.append(char)
            current_width += char_width
    lines.append("".join(current))
    return lines


def is_long_message(text: str) -> bool:
    return len(wrap_visual_lines(text)) > FORWARD_LINE_LIMIT


def split_long_message(text: str, line_limit: int = FORWARD_LINE_LIMIT) -> list[str]:
    lines = wrap_visual_lines(text)
    return ["\n".join(lines[index : index + line_limit]) for index in range(0, len(lines), line_limit)]


def extract_response_output_text(response: dict) -> str:
    for item in response.get("output", []):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        texts = []
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
        if texts:
            return "".join(texts).strip()
    return (response.get("output_text") or "").strip()


def extract_image_generation_calls(response: dict) -> list[dict]:
    calls = []
    for item in response.get("output", []):
        if item.get("type") != "image_generation_call":
            continue

        result = item.get("result")
        if isinstance(result, str) and result.strip():
            calls.append(
                {
                    "result": result,
                    "image_id": item.get("id") or item.get("image_id"),
                    "action": item.get("action"),
                }
            )
            continue

        if not isinstance(result, list):
            continue

        for entry in result:
            if isinstance(entry, str) and entry.strip():
                calls.append(
                    {
                        "result": entry,
                        "image_id": item.get("id") or item.get("image_id"),
                        "action": item.get("action"),
                    }
                )
    return calls


def parse_response_json_text(response: dict) -> dict:
    text = extract_response_output_text(response)
    if not text:
        return {}
    return json.loads(text)


def detect_image_media_type(
    image_bytes: bytes, fallback_media_type: str | None = None
) -> str | None:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    if (
        len(image_bytes) >= 12
        and image_bytes[0:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"

    if fallback_media_type:
        media_type = fallback_media_type.split(";", 1)[0].strip().lower()
        if media_type.startswith("image/"):
            return media_type
    return None
