import asyncio
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


nonebot_module = ModuleType("nonebot")
nonebot_module.get_driver = lambda: SimpleNamespace(config=SimpleNamespace(superusers=set()))
nonebot_log_module = ModuleType("nonebot.log")
nonebot_log_module.logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
nonebot_plugin_module = ModuleType("nonebot.plugin")
nonebot_plugin_module.get_loaded_plugins = lambda: []
nonebot_plugin_module.get_plugin = lambda name: None
nonebot_plugin_module.get_plugin_by_module_name = lambda name: None
nonebot_plugin_module.load_plugin = lambda name: None
sys.modules.setdefault("nonebot", nonebot_module)
sys.modules.setdefault("nonebot.log", nonebot_log_module)
sys.modules.setdefault("nonebot.plugin", nonebot_plugin_module)

module_path = Path(__file__).resolve().parents[1] / "nonebot_plugin_moellmchats" / "access_control.py"
spec = spec_from_file_location("nonebot_plugin_moellmchats.access_control", module_path)
access_control = module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = access_control
spec.loader.exec_module(access_control)


class FakeAccessRequestService:
    def __init__(self, *, allowed=False):
        self.allowed = allowed
        self.requests = []
        self.notifications = []

    def is_allowed(self, user_id, capability):
        return self.allowed

    def request_access(self, *, user_id, request_text, capability):
        self.requests.append(
            {
                "user_id": user_id,
                "request_text": request_text,
                "capability": capability,
            }
        )
        return SimpleNamespace(status="pending")

    async def notify_primary_superuser(self, bot, record, requester_name):
        self.notifications.append(
            {
                "bot": bot,
                "record": record,
                "requester_name": requester_name,
            }
        )
        return True


class AccessControlTest(unittest.TestCase):
    def test_unapproved_non_request_private_message_is_silent(self):
        service = FakeAccessRequestService()
        event = SimpleNamespace(user_id=10001)

        with patch.object(access_control, "_resolve_access_request_service", return_value=service):
            decision = asyncio.run(access_control.evaluate_private_access(object(), event, "hello"))

        self.assertFalse(decision.allowed)
        self.assertTrue(decision.handled)
        self.assertIsNone(decision.reply_text)
        self.assertEqual(service.requests, [])
        self.assertEqual(service.notifications, [])

    def test_exact_request_keyword_submits_silently(self):
        service = FakeAccessRequestService()
        event = SimpleNamespace(
            user_id=10001,
            sender=SimpleNamespace(card="", nickname="Tester"),
        )

        with patch.object(access_control, "_resolve_access_request_service", return_value=service):
            decision = asyncio.run(access_control.evaluate_private_access(object(), event, "申请"))

        self.assertFalse(decision.allowed)
        self.assertTrue(decision.handled)
        self.assertIsNone(decision.reply_text)
        self.assertEqual(
            service.requests,
            [
                {
                    "user_id": 10001,
                    "request_text": "申请",
                    "capability": "moellmchats.private_chat",
                }
            ],
        )
        self.assertEqual(len(service.notifications), 1)
        self.assertEqual(service.notifications[0]["requester_name"], "Tester")

    def test_approved_user_can_chat_without_application_reply(self):
        service = FakeAccessRequestService(allowed=True)
        event = SimpleNamespace(user_id=10001)

        with patch.object(access_control, "_resolve_access_request_service", return_value=service):
            decision = asyncio.run(access_control.evaluate_private_access(object(), event, "申请"))

        self.assertTrue(decision.allowed)
        self.assertFalse(decision.handled)
        self.assertIsNone(decision.reply_text)
        self.assertEqual(service.requests, [])


if __name__ == "__main__":
    unittest.main()
