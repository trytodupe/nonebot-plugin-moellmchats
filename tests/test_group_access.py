from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

nonebot_log_module = sys.modules.get("nonebot.log") or ModuleType("nonebot.log")
nonebot_log_module.logger = SimpleNamespace(
    info=lambda *args, **kwargs: None,
    success=lambda *args, **kwargs: None,
)
sys.modules["nonebot.log"] = nonebot_log_module

nonebot_plugin_module = sys.modules.get("nonebot.plugin") or ModuleType("nonebot.plugin")
nonebot_plugin_module.get_plugin = lambda _name: None
nonebot_plugin_module.get_loaded_plugins = lambda: set()
sys.modules["nonebot.plugin"] = nonebot_plugin_module

module_path = Path(__file__).resolve().parents[1] / "nonebot_plugin_moellmchats" / "group_access.py"
spec = spec_from_file_location("nonebot_plugin_moellmchats.group_access", module_path)
group_access = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(group_access)


def gate_plugin(gate, version=1):
    return SimpleNamespace(
        name="group_superuser_gate",
        module_name="group_superuser_gate",
        module=SimpleNamespace(
            GROUP_SUPERUSER_GATE_INTERFACE_VERSION=version,
            group_has_superuser=gate,
        ),
    )


class GroupAccessTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        group_access._group_gate = None

    async def test_auto_mode_uses_loaded_gate(self):
        gate = AsyncMock(return_value=False)
        with patch.object(group_access, "get_plugin", return_value=gate_plugin(gate)):
            group_access.configure_group_gate("auto")

        bot = object()
        event = SimpleNamespace(group_id=200)
        assert not await group_access.group_has_superuser(bot, event)
        gate.assert_awaited_once_with(bot, event)

    async def test_auto_mode_allows_groups_without_gate(self):
        with (
            patch.object(group_access, "get_plugin", return_value=None),
            patch.object(group_access, "get_loaded_plugins", return_value=set()),
        ):
            group_access.configure_group_gate("auto")

        assert await group_access.group_has_superuser(object(), SimpleNamespace(group_id=200))

    def test_required_mode_rejects_missing_gate(self):
        with (
            patch.object(group_access, "get_plugin", return_value=None),
            patch.object(group_access, "get_loaded_plugins", return_value=set()),
        ):
            error_message = None
            try:
                group_access.configure_group_gate("required")
            except RuntimeError as error:
                error_message = str(error)
            assert error_message is not None
            assert "requires the group_superuser_gate" in error_message

    def test_incompatible_loaded_gate_is_rejected(self):
        with patch.object(group_access, "get_plugin", return_value=gate_plugin(AsyncMock(), version=2)):
            error_message = None
            try:
                group_access.configure_group_gate("auto")
            except RuntimeError as error:
                error_message = str(error)
            assert error_message is not None
            assert "incompatible interface" in error_message

    async def test_private_events_remain_rejected_by_group_rule(self):
        gate = AsyncMock(return_value=True)
        with patch.object(group_access, "get_plugin", return_value=gate_plugin(gate)):
            group_access.configure_group_gate("required")

        assert not await group_access.group_has_superuser(object(), SimpleNamespace())
        gate.assert_not_awaited()
