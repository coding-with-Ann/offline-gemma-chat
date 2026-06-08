from __future__ import annotations

from .config import AppConfig
from .context import SlidingContextWindow
from .ollama import OllamaClient


class OfflineLLMApp:
    """Coordinates local model calls with a bounded conversation context."""

    def __init__(
        self,
        config: AppConfig | None = None,
        client: OllamaClient | None = None,
        context: SlidingContextWindow | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.client = client or OllamaClient(
            host=self.config.host,
            model=self.config.model,
            timeout_seconds=self.config.timeout_seconds,
        )
        self.context = context or SlidingContextWindow(
            max_tokens=self.config.max_context_tokens,
            system_prompt=self.config.system_prompt,
        )

    def ask(self, prompt: str) -> str:
        cleaned = prompt.strip()
        if not cleaned:
            raise ValueError("Prompt cannot be empty.")

        self.context.add_user_message(cleaned)
        reply = self.client.chat(self.context.messages())
        self.context.add_assistant_message(reply)
        return reply

    def ensure_ready(self) -> None:
        self.client.ensure_server_available()
        self.client.ensure_model_available()
