from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


module_path = Path(__file__).resolve().parents[1] / "nonebot_plugin_moellmchats" / "trigger_rules.py"
spec = spec_from_file_location("nonebot_plugin_moellmchats.trigger_rules", module_path)
trigger_rules = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(trigger_rules)


class AllowedGroupTest(unittest.TestCase):
    def test_allows_configured_group(self):
        assert trigger_rules.is_allowed_group(1015880675, [1015880675])

    def test_normalizes_string_group_ids(self):
        assert trigger_rules.is_allowed_group(1015880675, [" 1015880675 "])

    def test_rejects_other_groups(self):
        assert not trigger_rules.is_allowed_group(725601182, [1015880675])

    def test_rejects_missing_group(self):
        assert not trigger_rules.is_allowed_group(None, [1015880675])

    def test_empty_allowlist_rejects_all_groups(self):
        assert not trigger_rules.is_allowed_group(1015880675, [])
