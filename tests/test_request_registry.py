import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


module_path = Path(__file__).resolve().parents[1] / "nonebot_plugin_moellmchats" / "request_registry.py"
spec = spec_from_file_location("nonebot_plugin_moellmchats.request_registry", module_path)
request_registry_module = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(request_registry_module)

RequestRegistry = request_registry_module.RequestRegistry


class RequestRegistryTest(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.registry = RequestRegistry(clock=lambda: self.now)

    def begin(self, user_id="1", scope="group:10", allow_concurrent=False):
        return self.registry.begin(
            user_id=user_id,
            user_name=f"user-{user_id}",
            scope=scope,
            prompt_preview="a prompt",
            allow_concurrent=allow_concurrent,
        )

    def test_begin_and_finish(self):
        result = self.begin()
        self.assertTrue(result.started)
        self.assertEqual(len(self.registry.snapshot("group:10")), 1)

        self.registry.finish(result.request.request_id)
        self.assertEqual(self.registry.snapshot("group:10"), [])

    def test_same_user_requires_confirmation_but_different_user_does_not(self):
        first = self.begin()
        conflict = self.begin()
        other_user = self.begin(user_id="2")

        self.assertFalse(conflict.started)
        self.assertEqual(conflict.active_request, first.request)
        self.assertTrue(other_user.started)

    def test_confirmed_request_can_run_concurrently(self):
        self.begin()
        second = self.begin(allow_concurrent=True)

        self.assertTrue(second.started)
        self.assertEqual(len(self.registry.snapshot("group:10")), 2)

    def test_confirmation_is_bound_to_user_scope_and_reply(self):
        payload = object()
        self.registry.add_pending(user_id="1", scope="group:10", prompt_preview="next", payload=payload)
        self.registry.bind_confirmation("1", 99)

        self.assertFalse(self.registry.has_valid_confirmation(user_id="2", scope="group:10", reply_message_id=99))
        self.assertFalse(self.registry.has_valid_confirmation(user_id="1", scope="group:11", reply_message_id=99))
        self.assertFalse(self.registry.has_valid_confirmation(user_id="1", scope="group:10", reply_message_id=98))
        confirmed = self.registry.take_confirmed(user_id="1", scope="group:10", reply_message_id=99)
        self.assertIs(confirmed.payload, payload)

    def test_pending_confirmation_expires(self):
        self.registry.add_pending(user_id="1", scope="group:10", prompt_preview="next", payload=object())
        self.registry.bind_confirmation("1", 99)
        self.now += 120

        self.assertFalse(self.registry.has_valid_confirmation(user_id="1", scope="group:10", reply_message_id=99))

    def test_existing_pending_request_is_not_replaced_by_begin(self):
        self.begin()
        self.registry.add_pending(user_id="1", scope="group:10", prompt_preview="second", payload=object())

        conflict = self.begin()

        self.assertTrue(conflict.pending_exists)

    def test_add_pending_does_not_replace_existing_request(self):
        first_payload = object()
        self.registry.add_pending(user_id="1", scope="group:10", prompt_preview="second", payload=first_payload)

        result = self.registry.add_pending(user_id="1", scope="group:10", prompt_preview="third", payload=object())
        self.registry.bind_confirmation("1", 99)
        confirmed = self.registry.take_confirmed(user_id="1", scope="group:10", reply_message_id=99)

        self.assertIsNone(result)
        self.assertIs(confirmed.payload, first_payload)

    def test_snapshot_is_scope_isolated(self):
        self.begin(user_id="1", scope="group:10")
        self.begin(user_id="2", scope="group:11")
        self.begin(user_id="3", scope="private:3")

        self.assertEqual([item.user_id for item in self.registry.snapshot("group:10")], ["1"])
        self.assertEqual([item.user_id for item in self.registry.snapshot("private:3")], ["3"])


if __name__ == "__main__":
    unittest.main()
