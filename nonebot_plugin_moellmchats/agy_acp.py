import asyncio
import base64
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Awaitable, Callable

try:
    from nonebot.log import logger
except ImportError:  # pragma: no cover - fallback for standalone unit tests
    import logging

    logger = logging.getLogger(__name__)


DEFAULT_AGY_MODEL = "gemini-4-hydrogen-medium"
DEFAULT_ACP_TIMEOUT_SECONDS = 300.0
_DATA_URL_RE = re.compile(r"^data:([^;,]+);base64,(.+)$", re.DOTALL)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

_ALLOWED_DIRECT_TOOLS = {
    "search_web",
    "generate_image",
    "image-generator",
    "image_generator",
}
_ALLOWED_IMAGE_SUBAGENTS = {
    "image-generator",
    "image_generator",
    "generate_image",
}
_DENIED_TOOLS = {
    "run_command",
    "read_file",
    "write_file",
    "edit_file",
    "manage_task",
    "run_workflow",
    "swarm",
    "schedule",
    "delete_file",
    "view_file",
    "list_dir",
    "grep_search",
    "find_by_name",
}


@dataclass
class AcpTurnResult:
    text: str
    images: list[bytes] = field(default_factory=list)
    session_id: str | None = None


def get_agy_runtime_dir() -> Path:
    shm_dir = Path("/dev/shm")
    uid = os.getuid() if hasattr(os, "getuid") else 0
    if shm_dir.exists() and os.access(shm_dir, os.W_OK):
        return shm_dir / f"moellmchats_agy_acp_{uid}"
    return Path(tempfile.gettempdir()) / f"moellmchats_agy_acp_{uid}"


def prepare_agy_runtime(
    runtime_dir: Path | None = None,
    oauth_token_source: Path | None = None,
) -> tuple[Path, Path, Path]:
    base_dir = Path(runtime_dir) if runtime_dir is not None else get_agy_runtime_dir()
    gemini_dir = base_dir / "gemini"
    app_data_dir = gemini_dir / "antigravity-cli"
    cwd_dir = base_dir / "cwd"
    app_data_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir.mkdir(parents=True, exist_ok=True)

    src_token = (
        Path(oauth_token_source)
        if oauth_token_source is not None
        else Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
    )
    dst_token = app_data_dir / "antigravity-oauth-token"
    if src_token.exists() and not dst_token.exists() and not dst_token.is_symlink():
        try:
            os.symlink(src_token, dst_token)
        except FileExistsError:
            pass

    return base_dir, gemini_dir, cwd_dir


def cleanup_session_files(gemini_dir: Path, session_id: str | None) -> None:
    if not session_id:
        return
    app_data_dir = Path(gemini_dir) / "antigravity-cli"
    conversations_dir = app_data_dir / "conversations"
    for suffix in (".db", ".db-wal", ".db-shm"):
        db_file = conversations_dir / f"{session_id}{suffix}"
        if db_file.exists() or db_file.is_symlink():
            try:
                db_file.unlink()
            except OSError:
                pass

    brain_dir = app_data_dir / "brain" / session_id
    if brain_dir.exists():
        shutil.rmtree(brain_dir, ignore_errors=True)


def data_url_to_acp_image(data_url: str) -> dict[str, str] | None:
    if not isinstance(data_url, str):
        return None
    match = _DATA_URL_RE.match(data_url.strip())
    if not match:
        return None
    mime_type, b64_data = match.group(1).strip(), match.group(2).strip()
    if not mime_type or not b64_data:
        return None
    return {
        "type": "image",
        "mimeType": mime_type,
        "data": b64_data,
    }


def responses_content_to_acp_blocks(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        text = content.strip()
        return [{"type": "text", "text": content}] if text else []

    blocks: list[dict[str, Any]] = []
    if not isinstance(content, list):
        return blocks

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in ("input_text", "text"):
            text = str(item.get("text") or "")
            if text:
                blocks.append({"type": "text", "text": text})
        elif item_type in ("input_image", "image"):
            if "image_url" in item:
                image_block = data_url_to_acp_image(str(item.get("image_url") or ""))
                if image_block:
                    blocks.append(image_block)
            elif item.get("data") and item.get("mimeType"):
                blocks.append(
                    {
                        "type": "image",
                        "mimeType": str(item["mimeType"]),
                        "data": str(item["data"]),
                    }
                )
    return blocks


def _normalize_tool_identifier(raw_name: Any) -> str:
    if not isinstance(raw_name, str):
        return ""
    cleaned = raw_name.strip().lower()
    if not cleaned:
        return ""
    # Handle titles like "search_web: query" or "run_command (ls)"
    token = re.split(r"[\s:(\[]+", cleaned, maxsplit=1)[0]
    return token


def _extract_subagent_target(tool_call: dict[str, Any], params: dict[str, Any]) -> str:
    raw_input = tool_call.get("rawInput")
    candidates: list[Any] = []
    if isinstance(raw_input, dict):
        for key in ("TypeName", "type_name", "typeName", "subagent", "agent", "name", "tool"):
            if key in raw_input:
                candidates.append(raw_input.get(key))
    for key in ("subagent", "TypeName", "type_name", "typeName"):
        if key in tool_call:
            candidates.append(tool_call.get(key))
        if key in params:
            candidates.append(params.get(key))
    for candidate in candidates:
        normalized = _normalize_tool_identifier(candidate)
        if normalized:
            return normalized
    return ""


def _pick_permission_option_id(options: Any, *, allow: bool) -> str:
    if not isinstance(options, list):
        return "allow_once" if allow else "reject_once"

    preferred_kinds = (
        ("allow_once", "allow_always") if allow else ("reject_once", "reject_always")
    )
    for target_kind in preferred_kinds:
        for opt in options:
            if isinstance(opt, dict) and opt.get("kind") == target_kind:
                option_id = opt.get("optionId")
                if option_id:
                    return str(option_id)

    prefixes = ("allow",) if allow else ("reject", "deny")
    for opt in options:
        if isinstance(opt, dict):
            option_id = str(opt.get("optionId") or "")
            if option_id.lower().startswith(prefixes):
                return option_id

    return "allow_once" if allow else "reject_once"


def evaluate_permission_request(params: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(params, dict):
        return False, "reject_once"

    tool_call = params.get("toolCall") if isinstance(params.get("toolCall"), dict) else {}
    raw_names = [
        tool_call.get("toolName"),
        tool_call.get("name"),
        tool_call.get("tool"),
        tool_call.get("title"),
        params.get("toolName"),
        params.get("name"),
        params.get("tool"),
    ]
    tool_name = ""
    for raw in raw_names:
        normalized = _normalize_tool_identifier(raw)
        if normalized:
            tool_name = normalized
            break

    allowed = False
    if tool_name in _DENIED_TOOLS:
        allowed = False
    elif tool_name in _ALLOWED_DIRECT_TOOLS:
        allowed = True
    elif tool_name == "invoke_subagent":
        target = _extract_subagent_target(tool_call, params)
        allowed = target in _ALLOWED_IMAGE_SUBAGENTS

    option_id = _pick_permission_option_id(params.get("options"), allow=allowed)
    return allowed, option_id


def _extract_images_from_dir(
    root_dir: Path,
    *,
    ignore_paths: set[Path] | None = None,
    ignore_dir_names: set[str] | None = None,
) -> tuple[list[bytes], list[Path]]:
    if not root_dir.exists():
        return [], []
    images: list[bytes] = []
    found_paths: list[Path] = []
    ignored = ignore_paths or set()
    ignored_dirs = ignore_dir_names or set()
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        if ignored_dirs and any(part in ignored_dirs for part in path.relative_to(root_dir).parts[:-1]):
            continue
        if path in ignored:
            continue
        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        try:
            images.append(path.read_bytes())
            found_paths.append(path)
        except OSError:
            continue
    return images, found_paths


class AgyAcpClient:
    def __init__(
        self,
        *,
        runtime_dir: Path | None = None,
        oauth_token_source: Path | None = None,
        process_factory: Callable[..., Any] | None = None,
        timeout_seconds: float = DEFAULT_ACP_TIMEOUT_SECONDS,
    ):
        self.runtime_dir = runtime_dir
        self.oauth_token_source = oauth_token_source
        self._process_factory = process_factory
        self.timeout_seconds = timeout_seconds

        self._lock = asyncio.Lock()
        self._proc: Any = None
        self._reader_task: asyncio.Task | None = None
        self._next_id = 1
        self._pending: dict[int | str, asyncio.Future] = {}
        self._session_text_chunks: dict[str, list[str]] = {}
        self._session_images: dict[str, list[bytes]] = {}
        self._session_image_subagent_called: dict[str, bool] = {}
        self._initialized = False
        self._gemini_dir: Path | None = None
        self._cwd_dir: Path | None = None

    async def _spawn_process(self) -> Any:
        _, gemini_dir, cwd_dir = prepare_agy_runtime(
            runtime_dir=self.runtime_dir,
            oauth_token_source=self.oauth_token_source,
        )
        self._gemini_dir = gemini_dir
        self._cwd_dir = cwd_dir

        cmd = [
            "agy",
            "acp",
            "--disable-telemetry",
            "--gemini-dir",
            str(gemini_dir),
            "--app-data-dir",
            "antigravity-cli",
            "--log-file",
            "/dev/null",
        ]

        if self._process_factory is not None:
            return await self._process_factory(cmd, cwd=str(cwd_dir))

        try:
            return await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(cwd_dir),
                limit=16 * 1024 * 1024,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("未找到 agy 命令，无法使用大豆包（agy_acp）。") from exc

    async def _ensure_started(self) -> None:
        if self._proc is not None and getattr(self._proc, "returncode", None) is None and self._initialized:
            return

        await self._reset_process()
        self._proc = await self._spawn_process()
        self._reader_task = asyncio.create_task(self._reader_loop())

        await self._send_request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {
                    "name": "nonebot-plugin-moellmchats",
                    "version": "1.0.0",
                },
            },
        )
        self._initialized = True

    async def _reset_process(self) -> None:
        self._initialized = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

        if self._proc is not None:
            try:
                if getattr(self._proc, "returncode", None) is None:
                    self._proc.terminate()
                    try:
                        await asyncio.wait_for(self._proc.wait(), timeout=3.0)
                    except asyncio.TimeoutError:
                        self._proc.kill()
                        await self._proc.wait()
            except Exception:
                pass
            self._proc = None

        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("agy acp 进程已退出。"))
        self._pending.clear()

    async def close(self) -> None:
        async with self._lock:
            await self._reset_process()

    async def _write_json(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("agy acp 标准输入不可用。")
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()

    async def _send_request(self, method: str, params: dict[str, Any]) -> Any:
        req_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[req_id] = future
        try:
            await self._write_json(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": method,
                    "params": params,
                }
            )
            return await asyncio.wait_for(future, timeout=self.timeout_seconds)
        finally:
            self._pending.pop(req_id, None)

    async def _reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                raw_line = await self._proc.stdout.readline()
                if not raw_line:
                    break
                text = raw_line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(f"agy acp 返回了非 JSON 行: {text[:200]}")
                    continue

                if "id" in msg and "method" in msg:
                    await self._handle_server_request(msg)
                elif "id" in msg:
                    self._handle_response(msg)
                elif msg.get("method") == "session/update":
                    self._handle_session_update(msg.get("params") or {})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"agy acp reader loop error: {exc}")
        finally:
            self._initialized = False
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("agy acp 连接已断开。"))

    def _handle_response(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        future = self._pending.get(req_id)
        if future is None or future.done():
            return
        if "error" in msg and msg["error"] is not None:
            err = msg["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            future.set_exception(RuntimeError(f"agy acp 请求失败: {message}"))
        else:
            future.set_result(msg.get("result"))

    async def _handle_server_request(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            allowed, option_id = evaluate_permission_request(params)
            tool_call = params.get("toolCall") or {}
            tool_label = tool_call.get("title") or tool_call.get("name")
            if allowed and _normalize_tool_identifier(tool_label) != "search_web":
                req_sid = str(params.get("sessionId") or "")
                if req_sid:
                    self._session_image_subagent_called[req_sid] = True
                else:
                    for active_sid in self._session_text_chunks:
                        self._session_image_subagent_called[active_sid] = True
            logger.info(
                {
                    "event": "agy_acp_permission",
                    "allowed": allowed,
                    "option_id": option_id,
                    "tool": tool_label,
                }
            )
            await self._write_json(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "outcome": {
                            "outcome": "selected",
                            "optionId": option_id,
                        }
                    },
                }
            )
            return

        await self._write_json(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not supported: {method}",
                },
            }
        )

    def _handle_session_update(self, params: dict[str, Any]) -> None:
        session_id = str(params.get("sessionId") or "")
        update = params.get("update")
        if not session_id or not isinstance(update, dict):
            return

        update_type = update.get("sessionUpdate") or update.get("type")
        if update_type in ("tool_call", "tool_call_update"):
            raw_title = str(update.get("title") or update.get("name") or "").lower()
            raw_input = str(update.get("rawInput") or "").lower()
            if any(
                marker in raw_title or marker in raw_input
                for marker in ("image-generator", "image_generator", "generate_image")
            ):
                self._session_image_subagent_called[session_id] = True
            return

        if update_type not in ("agent_message_chunk", "agent_message"):
            return

        content = update.get("content")
        items = content if isinstance(content, list) else [content]
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text")
                if isinstance(text, str) and text:
                    self._session_text_chunks.setdefault(session_id, []).append(text)
            elif item_type == "image":
                b64_data = item.get("data")
                if isinstance(b64_data, str) and b64_data:
                    try:
                        self._session_images.setdefault(session_id, []).append(
                            base64.b64decode(b64_data)
                        )
                    except Exception:
                        pass

    async def run_turn(
        self,
        *,
        prompt_blocks: list[dict[str, Any]],
        model_info: dict[str, Any] | None = None,
        on_interim_text: Callable[[str], Awaitable[None]] | None = None,
    ) -> AcpTurnResult:
        model_info = model_info or {}
        async with self._lock:
            await self._ensure_started()
            assert self._gemini_dir is not None and self._cwd_dir is not None

            existing_cwd_files = {
                p for p in self._cwd_dir.rglob("*") if p.is_file()
            }
            brain_root = self._gemini_dir / "antigravity-cli" / "brain"
            conv_root = self._gemini_dir / "antigravity-cli" / "conversations"
            existing_brain_dirs = (
                {p for p in brain_root.iterdir() if p.is_dir()}
                if brain_root.exists()
                else set()
            )
            existing_brain_files = (
                {p for p in brain_root.rglob("*") if p.is_file()}
                if brain_root.exists()
                else set()
            )
            existing_conv_entries = (
                set(conv_root.iterdir()) if conv_root.exists() else set()
            )

            session_res = await self._send_request(
                "session/new",
                {
                    "cwd": str(self._cwd_dir),
                    "mcpServers": [],
                },
            )
            session_id = str((session_res or {}).get("sessionId") or "")
            if not session_id:
                raise RuntimeError("agy acp 未返回有效的 sessionId。")

            self._session_text_chunks[session_id] = []
            self._session_images[session_id] = []
            self._session_image_subagent_called[session_id] = False

            try:
                model_value = str(model_info.get("model") or DEFAULT_AGY_MODEL).strip()
                if model_value:
                    await self._send_request(
                        "session/set_config_option",
                        {
                            "sessionId": session_id,
                            "configId": "model",
                            "value": model_value,
                        },
                    )
                for config_key in ("thought_level", "mode"):
                    config_val = model_info.get(config_key)
                    if config_val:
                        await self._send_request(
                            "session/set_config_option",
                            {
                                "sessionId": session_id,
                                "configId": config_key,
                                "value": str(config_val),
                            },
                        )

                await self._send_request(
                    "session/prompt",
                    {
                        "sessionId": session_id,
                        "prompt": prompt_blocks,
                    },
                )

                input_hashes: set[str] = set()
                for block in prompt_blocks:
                    if isinstance(block, dict) and block.get("type") == "image":
                        b64_data = block.get("data")
                        if isinstance(b64_data, str) and b64_data:
                            try:
                                input_hashes.add(
                                    hashlib.sha256(base64.b64decode(b64_data)).hexdigest()
                                )
                            except Exception:
                                pass

                def _collect_turn_images() -> tuple[list[bytes], list[Path]]:
                    found: list[bytes] = []
                    hashes = set(input_hashes)
                    for img in self._session_images.get(session_id, []):
                        digest = hashlib.sha256(img).hexdigest()
                        if digest not in hashes:
                            hashes.add(digest)
                            found.append(img)
                    b_imgs, _ = _extract_images_from_dir(
                        brain_root,
                        ignore_paths=existing_brain_files,
                        ignore_dir_names={".user_uploaded", ".system_generated"},
                    )
                    c_imgs, c_paths = _extract_images_from_dir(
                        self._cwd_dir,
                        ignore_paths=existing_cwd_files,
                    )
                    for img in b_imgs + c_imgs:
                        digest = hashlib.sha256(img).hexdigest()
                        if digest not in hashes:
                            hashes.add(digest)
                            found.append(img)
                    return found, c_paths

                collected_images, cwd_image_paths = _collect_turn_images()
                initial_text = "".join(self._session_text_chunks.get(session_id, []))

                if self._session_image_subagent_called.get(session_id) and not collected_images:
                    if on_interim_text is not None and initial_text.strip():
                        try:
                            await on_interim_text(initial_text)
                        except Exception:
                            logger.warning("agy_acp interim text callback failed")
                    self._session_text_chunks[session_id] = []
                    loop = asyncio.get_running_loop()
                    deadline = loop.time() + min(self.timeout_seconds, 90.0)
                    image_seen_at: float | None = None
                    last_text_len = 0
                    text_stable_since: float | None = None
                    while loop.time() < deadline:
                        await asyncio.sleep(0.4)
                        collected_images, cwd_image_paths = _collect_turn_images()
                        cur_text_len = sum(
                            len(t) for t in self._session_text_chunks.get(session_id, [])
                        )
                        now = loop.time()
                        if collected_images:
                            if image_seen_at is None:
                                image_seen_at = now
                            if cur_text_len > 0:
                                if cur_text_len != last_text_len:
                                    last_text_len = cur_text_len
                                    text_stable_since = now
                                elif text_stable_since is not None and (now - text_stable_since) >= 0.8:
                                    break
                            elif (now - image_seen_at) >= 10.0:
                                break

                text_reply = "".join(self._session_text_chunks.get(session_id, [])) or initial_text

                for p in cwd_image_paths:
                    try:
                        p.unlink()
                    except OSError:
                        pass

                return AcpTurnResult(
                    text=text_reply,
                    images=collected_images,
                    session_id=session_id,
                )
            finally:
                self._session_text_chunks.pop(session_id, None)
                self._session_images.pop(session_id, None)
                self._session_image_subagent_called.pop(session_id, None)
                try:
                    if self._proc is not None and getattr(self._proc, "returncode", None) is None:
                        await self._send_request(
                            "session/close",
                            {"sessionId": session_id},
                        )
                except Exception:
                    pass
                cleanup_session_files(self._gemini_dir, session_id)
                if brain_root.exists():
                    for p in brain_root.iterdir():
                        if p.is_dir() and p not in existing_brain_dirs:
                            shutil.rmtree(p, ignore_errors=True)
                if conv_root.exists():
                    for p in conv_root.iterdir():
                        if p not in existing_conv_entries:
                            if p.is_dir():
                                shutil.rmtree(p, ignore_errors=True)
                            else:
                                try:
                                    p.unlink()
                                except OSError:
                                    pass


_shared_acp_client: AgyAcpClient | None = None


def get_shared_acp_client() -> AgyAcpClient:
    global _shared_acp_client
    if _shared_acp_client is None:
        _shared_acp_client = AgyAcpClient()
    return _shared_acp_client


async def close_shared_acp_client() -> None:
    global _shared_acp_client
    if _shared_acp_client is not None:
        await _shared_acp_client.close()
        _shared_acp_client = None
