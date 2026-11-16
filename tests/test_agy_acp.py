import asyncio
import base64
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
import unittest


module_path = Path(__file__).resolve().parents[1] / "nonebot_plugin_moellmchats" / "agy_acp.py"
spec = spec_from_file_location("nonebot_plugin_moellmchats.agy_acp", module_path)
agy_acp = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(agy_acp)


class _FakeStdin:
    def __init__(self, process: "_FakeAcpProcess"):
        self._process = process
        self._buffer = bytearray()

    def write(self, data: bytes) -> None:
        self._buffer.extend(data)
        while b"\n" in self._buffer:
            line, _, rest = self._buffer.partition(b"\n")
            self._buffer = bytearray(rest)
            text = line.decode("utf-8").strip()
            if text:
                self._process.handle_client_message(json.loads(text))

    async def drain(self) -> None:
        return None


class _FakeAcpProcess:
    def __init__(
        self,
        *,
        runtime_dir: Path,
        session_id: str = "sess-1",
        emit_permission_checks: bool = False,
        create_brain_image: bool = False,
    ):
        self.runtime_dir = runtime_dir
        self.session_id = session_id
        self.emit_permission_checks = emit_permission_checks
        self.create_brain_image = create_brain_image
        self.returncode: int | None = None
        self.stdin = _FakeStdin(self)
        self.stdout = asyncio.StreamReader()
        self.received_methods: list[str] = []
        self.config_options_set: list[tuple[str, str]] = []
        self.permission_outcomes: list[tuple[str, str]] = []
        self._pending_prompt_id: int | str | None = None

    def _push(self, payload: dict) -> None:
        self.stdout.feed_data((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))

    def handle_client_message(self, msg: dict) -> None:
        if "method" in msg:
            method = msg["method"]
            req_id = msg.get("id")
            params = msg.get("params") or {}
            self.received_methods.append(method)

            if method == "initialize":
                self._push({"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": 1}})
            elif method == "session/new":
                self._push({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": self.session_id}})
            elif method == "session/set_config_option":
                self.config_options_set.append((params["configId"], params["value"]))
                self._push({"jsonrpc": "2.0", "id": req_id, "result": {"configOptions": []}})
            elif method == "session/prompt":
                if self.emit_permission_checks:
                    self._pending_prompt_id = req_id
                    self._push(
                        {
                            "jsonrpc": "2.0",
                            "id": 9001,
                            "method": "session/request_permission",
                            "params": {
                                "sessionId": self.session_id,
                                "toolCall": {"title": "search_web"},
                                "options": [
                                    {"optionId": "allow_once", "kind": "allow_once"},
                                    {"optionId": "reject_once", "kind": "reject_once"},
                                ],
                            },
                        }
                    )
                else:
                    self._finish_prompt(req_id)
            elif method == "session/close":
                self._push({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif "id" in msg:
            req_id = msg["id"]
            outcome = ((msg.get("result") or {}).get("outcome") or {}).get("optionId", "")
            if req_id == 9001:
                self.permission_outcomes.append(("search_web", outcome))
                self._push(
                    {
                        "jsonrpc": "2.0",
                        "id": 9002,
                        "method": "session/request_permission",
                        "params": {
                            "sessionId": self.session_id,
                            "toolCall": {"title": "run_command"},
                            "options": [
                                {"optionId": "allow_once", "kind": "allow_once"},
                                {"optionId": "reject_once", "kind": "reject_once"},
                            ],
                        },
                    }
                )
            elif req_id == 9002:
                self.permission_outcomes.append(("run_command", outcome))
                assert self._pending_prompt_id is not None
                self._finish_prompt(self._pending_prompt_id)

    def _finish_prompt(self, req_id: int | str) -> None:
        self._push(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": self.session_id,
                    "update": {
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"type": "text", "text": "internal thinking"},
                    },
                },
            }
        )
        self._push(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": self.session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {
                            "type": "text",
                            "text": '{"assistant_reply":"你好","image_memories":[]}',
                        },
                    },
                },
            }
        )
        if self.create_brain_image:
            brain_dir = (
                self.runtime_dir
                / "gemini"
                / "antigravity-cli"
                / "brain"
                / self.session_id
            )
            brain_dir.mkdir(parents=True, exist_ok=True)
            (brain_dir / "generated.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
            conv_dir = self.runtime_dir / "gemini" / "antigravity-cli" / "conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)
            (conv_dir / f"{self.session_id}.db").write_bytes(b"sqlite")

        self._push({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}})

    def terminate(self) -> None:
        if self.returncode is None:
            self.returncode = 0
            self.stdout.feed_eof()

    def kill(self) -> None:
        self.terminate()

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class AgyAcpTest(unittest.TestCase):
    def test_permission_allow_and_deny_rules(self):
        options = [
            {"optionId": "allow_once", "kind": "allow_once"},
            {"optionId": "reject_once", "kind": "reject_once"},
        ]

        allowed, opt = agy_acp.evaluate_permission_request(
            {"toolCall": {"title": "search_web"}, "options": options}
        )
        self.assertTrue(allowed)
        self.assertEqual(opt, "allow_once")

        for img_tool in ("image-generator", "generate_image", "image_generator"):
            allowed, opt = agy_acp.evaluate_permission_request(
                {"toolCall": {"name": img_tool}, "options": options}
            )
            self.assertTrue(allowed, msg=img_tool)
            self.assertEqual(opt, "allow_once")

        allowed, opt = agy_acp.evaluate_permission_request(
            {
                "toolCall": {
                    "name": "invoke_subagent",
                    "rawInput": {"TypeName": "image-generator"},
                },
                "options": options,
            }
        )
        self.assertTrue(allowed)
        self.assertEqual(opt, "allow_once")

        allowed, opt = agy_acp.evaluate_permission_request(
            {
                "toolCall": {
                    "name": "invoke_subagent",
                    "rawInput": {"TypeName": "research"},
                },
                "options": options,
            }
        )
        self.assertFalse(allowed)
        self.assertEqual(opt, "reject_once")

        for denied_tool in (
            "run_command",
            "read_file",
            "write_file",
            "edit_file",
            "manage_task",
            "run_workflow",
            "swarm",
            "schedule",
            "unknown_custom_tool",
        ):
            allowed, opt = agy_acp.evaluate_permission_request(
                {"toolCall": {"title": denied_tool}, "options": options}
            )
            self.assertFalse(allowed, msg=denied_tool)
            self.assertEqual(opt, "reject_once")

    def test_data_url_and_responses_content_conversion(self):
        raw_b64 = base64.b64encode(b"hello-png").decode("ascii")
        block = agy_acp.data_url_to_acp_image(f"data:image/png;base64,{raw_b64}")
        self.assertEqual(
            block,
            {"type": "image", "mimeType": "image/png", "data": raw_b64},
        )
        self.assertIsNone(agy_acp.data_url_to_acp_image("https://example.com/a.png"))

        blocks = agy_acp.responses_content_to_acp_blocks(
            [
                {"type": "input_text", "text": "看这张图"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{raw_b64}"},
            ]
        )
        self.assertEqual(
            blocks,
            [
                {"type": "text", "text": "看这张图"},
                {"type": "image", "mimeType": "image/jpeg", "data": raw_b64},
            ],
        )

    def test_runtime_setup_and_session_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            token_src = tmp_path / "source_token"
            token_src.write_text("oauth-secret", encoding="utf-8")
            runtime_dir = tmp_path / "runtime"

            _, gemini_dir, cwd_dir = agy_acp.prepare_agy_runtime(
                runtime_dir=runtime_dir,
                oauth_token_source=token_src,
            )
            self.assertTrue(cwd_dir.is_dir())
            token_dst = gemini_dir / "antigravity-cli" / "antigravity-oauth-token"
            self.assertTrue(token_dst.is_symlink())
            self.assertEqual(token_dst.read_text(encoding="utf-8"), "oauth-secret")

            sid = "abc-123"
            conv_dir = gemini_dir / "antigravity-cli" / "conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)
            db_file = conv_dir / f"{sid}.db"
            db_file.write_bytes(b"db")
            brain_dir = gemini_dir / "antigravity-cli" / "brain" / sid
            brain_dir.mkdir(parents=True, exist_ok=True)
            (brain_dir / "out.png").write_bytes(b"png")

            agy_acp.cleanup_session_files(gemini_dir, sid)
            self.assertFalse(db_file.exists())
            self.assertFalse(brain_dir.exists())

    def test_client_lifecycle_permissions_images_and_restart(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as tmp:
                runtime_dir = Path(tmp) / "runtime"
                spawned: list[_FakeAcpProcess] = []

                async def factory(cmd, cwd):
                    idx = len(spawned) + 1
                    proc = _FakeAcpProcess(
                        runtime_dir=runtime_dir,
                        session_id=f"sess-{idx}",
                        emit_permission_checks=(idx == 1),
                        create_brain_image=(idx == 1),
                    )
                    spawned.append(proc)
                    return proc

                client = agy_acp.AgyAcpClient(
                    runtime_dir=runtime_dir,
                    process_factory=factory,
                    timeout_seconds=5.0,
                )
                try:
                    res1 = await client.run_turn(
                        prompt_blocks=[{"type": "text", "text": "hello"}],
                        model_info={"thought_level": "medium", "mode": "default"},
                    )
                    self.assertEqual(res1.text, '{"assistant_reply":"你好","image_memories":[]}')
                    self.assertEqual(res1.images, [b"\x89PNG\r\n\x1a\nfake"])
                    self.assertEqual(
                        spawned[0].received_methods,
                        [
                            "initialize",
                            "session/new",
                            "session/set_config_option",
                            "session/set_config_option",
                            "session/set_config_option",
                            "session/prompt",
                            "session/close",
                        ],
                    )
                    self.assertEqual(
                        spawned[0].config_options_set,
                        [
                            ("model", "gemini-4-hydrogen-medium"),
                            ("thought_level", "medium"),
                            ("mode", "default"),
                        ],
                    )
                    self.assertEqual(
                        spawned[0].permission_outcomes,
                        [("search_web", "allow_once"), ("run_command", "reject_once")],
                    )
                    # Session files should be cleaned up
                    self.assertFalse(
                        (runtime_dir / "gemini" / "antigravity-cli" / "brain" / "sess-1").exists()
                    )
                    self.assertFalse(
                        (
                            runtime_dir
                            / "gemini"
                            / "antigravity-cli"
                            / "conversations"
                            / "sess-1.db"
                        ).exists()
                    )

                    # Simulate process exit; next turn must auto-restart
                    spawned[0].terminate()
                    res2 = await client.run_turn(
                        prompt_blocks=[{"type": "text", "text": "second"}],
                        model_info={"model": "gemini-4-carbon"},
                    )
                    self.assertEqual(len(spawned), 2)
                    self.assertEqual(res2.text, '{"assistant_reply":"你好","image_memories":[]}')
                finally:
                    await client.close()

        asyncio.run(run_test())

    def test_ignores_user_uploaded_and_input_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain_dir = Path(tmp) / "brain" / "sess-x"
            (brain_dir / ".user_uploaded").mkdir(parents=True)
            (brain_dir / ".system_generated").mkdir(parents=True)
            (brain_dir / ".user_uploaded" / "input.png").write_bytes(b"user-input")
            (brain_dir / ".system_generated" / "sys.png").write_bytes(b"sys-img")
            (brain_dir / "generated.png").write_bytes(b"gen-img")

            images, _ = agy_acp._extract_images_from_dir(
                brain_dir,
                ignore_dir_names={".user_uploaded", ".system_generated"},
            )
            self.assertEqual(images, [b"gen-img"])


if __name__ == "__main__":
    unittest.main()
