from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

nonebot_module = sys.modules.get("nonebot") or ModuleType("nonebot")
nonebot_module.get_driver = lambda: SimpleNamespace(config=SimpleNamespace(superusers=set()))
sys.modules["nonebot"] = nonebot_module
sys.modules.setdefault(
    "nonebot.adapters.onebot.v11",
    SimpleNamespace(Bot=object, MessageEvent=object),
)
sys.modules.setdefault(
    "nonebot.log",
    SimpleNamespace(logger=SimpleNamespace(warning=lambda *args, **kwargs: None)),
)

module_path = Path(__file__).resolve().parents[1] / "nonebot_plugin_moellmchats" / "group_access.py"
spec = spec_from_file_location("nonebot_plugin_moellmchats.group_access", module_path)
group_access = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(group_access)


class FakeGroupMessageEvent:
    def __init__(self, group_id=200, message_id=300):
        self.group_id = group_id
        self.message_id = message_id


class GroupAccessTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        group_access._event_access_cache.clear()
        self.bot = SimpleNamespace(
            self_id=100,
            get_group_member_list=AsyncMock(),
        )

    async def _check(self, event=None, superusers=None):
        event = event or FakeGroupMessageEvent()
        superusers = {"10", "20"} if superusers is None else superusers
        with patch.object(
            group_access,
            "get_driver",
            return_value=SimpleNamespace(config=SimpleNamespace(superusers=superusers)),
        ):
            return await group_access.group_has_superuser(self.bot, event)

    async def test_allows_group_with_configured_superuser_member(self):
        self.bot.get_group_member_list.return_value = [
            {"user_id": 9},
            {"user_id": 20},
        ]

        assert await self._check()

    async def test_rejects_group_without_configured_superuser_member(self):
        self.bot.get_group_member_list.return_value = [{"user_id": 9}]

        assert not await self._check()

    async def test_rejects_silently_when_member_query_fails(self):
        self.bot.get_group_member_list.side_effect = RuntimeError("unsupported")

        assert not await self._check()

    async def test_rejects_without_configured_superusers_without_query(self):
        assert not await self._check(superusers=set())
        self.bot.get_group_member_list.assert_not_awaited()

    async def test_reuses_result_for_same_message_event(self):
        self.bot.get_group_member_list.return_value = [{"user_id": 10}]
        event = FakeGroupMessageEvent()

        assert await self._check(event)
        assert await self._check(event)
        self.bot.get_group_member_list.assert_awaited_once_with(group_id=200)
