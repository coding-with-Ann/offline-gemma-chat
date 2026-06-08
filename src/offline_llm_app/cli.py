from __future__ import annotations

import argparse
import sys

from .app import OfflineLLMApp
from .config import AppConfig
from .ollama import OllamaError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="offline-llm",
        description="Privacy-first local Gemma 2 chat through Ollama.",
    )
    parser.add_argument("--prompt", help="Run one prompt and exit.")
    parser.add_argument("--model", default="gemma2", help="Ollama model name.")
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Local Ollama host URL.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=4096,
        help="Approximate context token budget.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP timeout for local Ollama calls.",
    )
    parser.add_argument(
        "--skip-ready-check",
        action="store_true",
        help="Skip Ollama server and model readiness checks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AppConfig(
        model=args.model,
        host=args.host,
        max_context_tokens=args.max_context_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    app = OfflineLLMApp(config)

    try:
        if not args.skip_ready_check:
            app.ensure_ready()

        if args.prompt:
            print(app.ask(args.prompt))
            return 0

        return _interactive(app)
    except (OllamaError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _interactive(app: OfflineLLMApp) -> int:
    print("Offline LLM ready. Type /exit to quit.")
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            print()
            return 0

        if prompt.strip().lower() in {"/exit", "/quit"}:
            return 0
        if not prompt.strip():
            continue

        print(app.ask(prompt))
