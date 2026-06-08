from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def as_ollama(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class SlidingContextWindow:
    """Maintains recent chat history within an approximate token budget."""

    def __init__(self, max_tokens: int, system_prompt: str | None = None) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        self.max_tokens = max_tokens
        self._messages: list[ChatMessage] = []
        if system_prompt:
            self._messages.append(ChatMessage("system", system_prompt.strip()))

    def add_user_message(self, content: str) -> None:
        self._append(ChatMessage("user", content.strip()))

    def add_assistant_message(self, content: str) -> None:
        self._append(ChatMessage("assistant", content.strip()))

    def messages(self) -> list[dict[str, str]]:
        return [message.as_ollama() for message in self._messages]

    def _append(self, message: ChatMessage) -> None:
        if not message.content:
            raise ValueError("message content cannot be empty.")
        self._messages.append(message)
        self._trim_to_budget()

    def _trim_to_budget(self) -> None:
        while self._token_count() > self.max_tokens and len(self._messages) > 1:
            remove_index = 1 if self._messages[0].role == "system" else 0
            del self._messages[remove_index]

    def _token_count(self) -> int:
        return sum(estimate_tokens(message.content) + 4 for message in self._messages)


def estimate_tokens(text: str) -> int:
    """Approximate tokens without depending on a tokenizer package."""

    if not text:
        return 0
    words = text.split()
    character_estimate = max(1, len(text) // 4)
    return max(len(words), character_estimate)
