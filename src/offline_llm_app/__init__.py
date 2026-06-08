"""Privacy-first offline LLM application package."""

from .app import OfflineLLMApp
from .config import AppConfig
from .context import ChatMessage, SlidingContextWindow
from .ollama import OllamaClient, OllamaError

__all__ = [
    "AppConfig",
    "ChatMessage",
    "OfflineLLMApp",
    "OllamaClient",
    "OllamaError",
    "SlidingContextWindow",
]
