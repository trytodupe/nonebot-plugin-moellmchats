import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


module_path = Path(__file__).resolve().parents[1] / "nonebot_plugin_moellmchats" / "trigger_rules.py"
spec = spec_from_file_location("nonebot_plugin_moellmchats.trigger_rules", module_path)
trigger_rules = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(trigger_rules)


class GroupTriggerTest(unittest.TestCase):
    def test_contains_doubao_help_trigger_accepts_match_within_first_five_chars(self):
        self.assertTrue(trigger_rules.contains_doubao_help_trigger("豆包帮我看看"))
        self.assertTrue(trigger_rules.contains_doubao_help_trigger("  豆包 帮我查一下"))
        self.assertTrue(trigger_rules.contains_doubao_help_trigger("@豆包 帮我看看"))
        self.assertTrue(trigger_rules.contains_doubao_help_trigger("哈 豆包帮我"))
        self.assertTrue(trigger_rules.contains_doubao_help_trigger("豆包"))

    def test_contains_doubao_help_trigger_rejects_matches_after_first_five_chars(self):
        self.assertFalse(trigger_rules.contains_doubao_help_trigger("帮我叫一下豆包"))
        self.assertFalse(trigger_rules.contains_doubao_help_trigger("12345豆包帮我"))
        self.assertFalse(trigger_rules.contains_doubao_help_trigger("帮我看看"))

    def test_should_trigger_group_chat_accepts_at_mention(self):
        self.assertTrue(trigger_rules.should_trigger_group_chat(True, "随便说点什么"))

    def test_should_trigger_group_chat_accepts_doubao_prefix_without_at(self):
        self.assertTrue(trigger_rules.should_trigger_group_chat(False, "豆包帮我查一下"))

    def test_should_trigger_group_chat_rejects_unrelated_group_text(self):
        self.assertFalse(trigger_rules.should_trigger_group_chat(False, "随便聊聊"))

    def test_contains_big_doubao_trigger_accepts_match_within_first_six_chars(self):
        self.assertTrue(trigger_rules.contains_big_doubao_trigger("大豆包帮我看看"))
        self.assertTrue(trigger_rules.contains_big_doubao_trigger("  大豆包 帮我查一下"))
        self.assertTrue(trigger_rules.contains_big_doubao_trigger("@大豆包 帮我看看"))
        self.assertTrue(trigger_rules.contains_big_doubao_trigger("哈 大豆包帮我"))
        self.assertTrue(trigger_rules.contains_big_doubao_trigger("123大豆包"))
        self.assertTrue(trigger_rules.contains_big_doubao_trigger("大豆包"))

    def test_contains_big_doubao_trigger_rejects_regular_doubao_and_late_matches(self):
        self.assertFalse(trigger_rules.contains_big_doubao_trigger("豆包帮我看看"))
        self.assertFalse(trigger_rules.contains_big_doubao_trigger("帮我叫一下大豆包"))
        self.assertFalse(trigger_rules.contains_big_doubao_trigger("1234大豆包帮我"))

    def test_should_trigger_group_chat_accepts_big_doubao_prefix_without_at(self):
        self.assertTrue(trigger_rules.should_trigger_group_chat(False, "123大豆包帮我查一下"))


if __name__ == "__main__":
    unittest.main()
