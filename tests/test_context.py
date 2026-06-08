import unittest

from offline_llm_app.context import SlidingContextWindow, estimate_tokens


class SlidingContextWindowTests(unittest.TestCase):
    def test_preserves_system_prompt_when_trimming(self):
        window = SlidingContextWindow(max_tokens=18, system_prompt="system rules")

        window.add_user_message("first message that should be removed")
        window.add_assistant_message("second message that should also fit poorly")
        window.add_user_message("latest")

        messages = window.messages()
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "latest"})
        self.assertNotIn("first message that should be removed", [m["content"] for m in messages])

    def test_estimate_tokens_uses_word_or_character_floor(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreaterEqual(estimate_tokens("one two three"), 3)
        self.assertGreaterEqual(estimate_tokens("abcdefghijklmnop"), 4)


if __name__ == "__main__":
    unittest.main()
