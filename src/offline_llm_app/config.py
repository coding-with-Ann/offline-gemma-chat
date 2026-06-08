from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SYSTEM_PROMPT = (
    "You are a local, privacy-first assistant. Keep answers useful, concise, "
    "and avoid requesting sensitive data unless it is necessary."
)


@dataclass(frozen=True)
class AppConfig:
    model: str = "gemma2"
    host: str = "http://127.0.0.1:11434"
    max_context_tokens: int = 4096
    timeout_seconds: float = 120.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if self.max_context_tokens < 256:
            raise ValueError("max_context_tokens must be at least 256.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if not self.model.strip():
            raise ValueError("model cannot be empty.")
        if not self.host.startswith(("http://", "https://")):
            raise ValueError("host must start with http:// or https://.")
