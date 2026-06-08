from __future__ import annotations

import json
import subprocess
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    """Raised when the local Ollama boundary fails."""


class OllamaClient:
    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "gemma2",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.host = host.rstrip("/") + "/"
        self.model = model
        self.timeout_seconds = timeout_seconds

    def ensure_server_available(self) -> None:
        try:
            self._request("api/tags", method="GET")
        except OllamaError as exc:
            raise OllamaError(
                "Ollama is not reachable locally. Start it with `ollama serve`."
            ) from exc

    def ensure_model_available(self) -> None:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise OllamaError("Could not run `ollama list`; verify Ollama is installed.")

        installed = {
            line.split()[0].split(":")[0]
            for line in result.stdout.splitlines()[1:]
            if line.split()
        }
        if self.model.split(":")[0] not in installed:
            raise OllamaError(
                f"Model `{self.model}` is not installed. Run `ollama pull {self.model}`."
            )

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        response = self._request("api/chat", payload=payload)
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise OllamaError("Ollama returned an unexpected chat response.") from exc
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama returned an empty response.")
        return content

    def _request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            urljoin(self.host, path),
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise OllamaError(f"Ollama network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OllamaError("Ollama request timed out.") from exc

        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise OllamaError("Ollama returned a non-object JSON response.")
        return parsed
