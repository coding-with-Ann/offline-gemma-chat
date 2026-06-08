import unittest
from unittest.mock import Mock

from offline_llm_app import AppConfig, OfflineLLMApp


class OfflineLLMAppTests(unittest.TestCase):
    def test_ask_adds_prompt_and_response_to_context(self):
        client = Mock()
        client.chat.return_value = "Local answer"
        app = OfflineLLMApp(
            config=AppConfig(max_context_tokens=512),
            client=client,
        )

        response = app.ask("  hello  ")

        self.assertEqual(response, "Local answer")
        sent_messages = client.chat.call_args.args[0]
        self.assertEqual(sent_messages[-1], {"role": "user", "content": "hello"})
        self.assertEqual(app.context.messages()[-1], {"role": "assistant", "content": "Local answer"})

    def test_empty_prompt_is_rejected(self):
        app = OfflineLLMApp(client=Mock())

        with self.assertRaises(ValueError):
            app.ask("   ")


if __name__ == "__main__":
    unittest.main()
