import json


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


def parse_sse_event_chunk(chunk: bytes) -> list[dict]:
    events = []
    if not chunk:
        return events

    text = chunk.decode("utf-8", errors="ignore")
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event_name = None
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        data_text = "\n".join(data_lines).strip()
        if not data_text or data_text == "[DONE]":
            continue
        try:
            payload = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        events.append({"event": event_name, "data": payload})
    return events


def is_image_generation_sse_event(event_name: str | None, payload: dict) -> bool:
    if event_name and "image_generation_call" in event_name:
        return True
    event_type = payload.get("type")
    if isinstance(event_type, str) and "image_generation_call" in event_type:
        return True
    return False


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
