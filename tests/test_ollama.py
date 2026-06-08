import io
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from offline_llm_app.ollama import OllamaClient, OllamaError


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class OllamaClientTests(unittest.TestCase):
    @patch("offline_llm_app.ollama.urlopen")
    def test_chat_posts_messages_to_local_ollama(self, urlopen_mock):
        urlopen_mock.return_value = FakeResponse({"message": {"content": "hello"}})
        client = OllamaClient(host="http://127.0.0.1:11434", model="gemma2")

        response = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(response, "hello")
        request = urlopen_mock.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(request.get_method(), "POST")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "gemma2")
        self.assertFalse(payload["stream"])

    @patch("offline_llm_app.ollama.urlopen")
    def test_network_error_is_wrapped(self, urlopen_mock):
        urlopen_mock.side_effect = URLError("offline")
        client = OllamaClient()

        with self.assertRaises(OllamaError):
            client.chat([{"role": "user", "content": "hi"}])

    @patch("offline_llm_app.ollama.subprocess.run")
    def test_model_check_uses_ollama_process_boundary(self, run_mock):
        run_mock.return_value = Mock(returncode=0, stdout="NAME ID SIZE MODIFIED\ngemma2:latest abc 1GB now\n")
        client = OllamaClient(model="gemma2")

        client.ensure_model_available()

        run_mock.assert_called_once_with(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )

    @patch("offline_llm_app.ollama.subprocess.run")
    def test_model_check_fails_when_model_missing(self, run_mock):
        run_mock.return_value = Mock(returncode=0, stdout="NAME ID SIZE MODIFIED\nllama3 abc 1GB now\n")
        client = OllamaClient(model="gemma2")

        with self.assertRaises(OllamaError):
            client.ensure_model_available()


if __name__ == "__main__":
    unittest.main()
