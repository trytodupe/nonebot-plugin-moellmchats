import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "nonebot_plugin_moellmchats/group_context.py"
spec = spec_from_file_location("group_context", path)
group_context = module_from_spec(spec)
spec.loader.exec_module(group_context)


class GroupContextTest(unittest.TestCase):
    def setUp(self):
        group_context._providers.clear()

    def tearDown(self):
        group_context._providers.clear()

    def test_provider_can_scope_context_to_one_group(self):
        group_context.register_group_context_provider(
            "cabbage", lambda bot, event: "runtime facts" if event.group_id == 136312506 else None
        )
        self.assertEqual(group_context.collect_group_context(None, SimpleNamespace(group_id=136312506)), "runtime facts")
        self.assertEqual(group_context.collect_group_context(None, SimpleNamespace(group_id=42)), "")
        self.assertEqual(group_context.collect_group_context(None, SimpleNamespace()), "")

    def test_provider_failure_does_not_break_other_context(self):
        def broken(bot, event):
            raise ValueError("broken provider")

        group_context.register_group_context_provider("broken", broken)
        group_context.register_group_context_provider("working", lambda bot, event: "working")
        with patch.object(group_context._logger, "exception") as log:
            self.assertEqual(group_context.collect_group_context(None, SimpleNamespace(group_id=1)), "working")
            log.assert_called_once()

    def test_registration_replaces_provider_and_bounds_output(self):
        group_context.register_group_context_provider("cabbage", lambda bot, event: "old")
        group_context.register_group_context_provider("cabbage", lambda bot, event: "x" * 10000)
        self.assertEqual(len(group_context.collect_group_context(None, SimpleNamespace(group_id=1))), 8192)
