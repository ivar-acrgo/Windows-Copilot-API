"""Tests for multi-turn session key derivation."""

import unittest

from server.prompt import messages_session_key, messages_store_key, turn_prompt
from server.schemas import ChatMessage
from server.sessions import ConversationSessionStore


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


class TurnPromptTests(unittest.TestCase):
    def test_single_user_message(self):
        messages = [_msg("user", "Hello")]
        self.assertEqual(turn_prompt(messages), "Hello")

    def test_system_plus_last_user_only(self):
        messages = [
            _msg("system", "Be brief."),
            _msg("user", "I am Ada."),
            _msg("assistant", "Hi Ada!"),
            _msg("user", "What is my name?"),
        ]
        self.assertEqual(turn_prompt(messages), "Be brief.\n\nWhat is my name?")


class SessionKeyTests(unittest.TestCase):
    def test_first_turn_has_no_lookup_key(self):
        messages = [_msg("user", "Hello")]
        self.assertIsNone(messages_session_key(messages))
        self.assertEqual(messages_store_key(messages), messages_store_key(messages))

    def test_second_turn_lookup_uses_prior_user_only(self):
        turn1 = [_msg("user", "I am Ada.")]
        turn2 = [
            _msg("user", "I am Ada."),
            _msg("assistant", "client-side wording may differ"),
            _msg("user", "What is my name?"),
        ]
        self.assertEqual(messages_session_key(turn2), messages_store_key(turn1))

    def test_third_turn_lookup_uses_first_two_users(self):
        turn2 = [
            _msg("user", "I am Ada."),
            _msg("assistant", "Hi!"),
            _msg("user", "Remember the number 42."),
        ]
        turn3 = turn2 + [_msg("assistant", "OK."), _msg("user", "What number?")]
        self.assertEqual(messages_session_key(turn3), messages_store_key(turn2))


class ConversationSessionStoreTests(unittest.TestCase):
    def test_put_get_round_trip(self):
        store = ConversationSessionStore(ttl_seconds=60)
        store.put("key", "conv-123")
        self.assertEqual(store.get("key"), "conv-123")

    def test_disabled_when_ttl_zero(self):
        store = ConversationSessionStore(ttl_seconds=0)
        store.put("key", "conv-123")
        self.assertIsNone(store.get("key"))


if __name__ == "__main__":
    unittest.main()
