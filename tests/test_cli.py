import unittest
from unittest.mock import Mock, patch

from offline_llm_app.cli import main


class CliTests(unittest.TestCase):
    @patch("offline_llm_app.cli.OfflineLLMApp")
    def test_one_shot_prompt_prints_response(self, app_cls):
        app = Mock()
        app.ask.return_value = "answer"
        app_cls.return_value = app

        with patch("builtins.print") as print_mock:
            exit_code = main(["--prompt", "hi", "--skip-ready-check"])

        self.assertEqual(exit_code, 0)
        app.ensure_ready.assert_not_called()
        app.ask.assert_called_once_with("hi")
        print_mock.assert_called_once_with("answer")

    @patch("offline_llm_app.cli.OfflineLLMApp")
    def test_ready_check_runs_by_default(self, app_cls):
        app = Mock()
        app.ask.return_value = "answer"
        app_cls.return_value = app

        exit_code = main(["--prompt", "hi"])

        self.assertEqual(exit_code, 0)
        app.ensure_ready.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
