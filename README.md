# Offline Gemma Chat

Privacy-first local chat application for Ollama and Gemma 2. Prompts and model
responses stay on the local machine; the app talks only to a local Ollama HTTP
endpoint.

## Features

- Local-only Ollama client using the standard library.
- Default model: `gemma2`.
- Sliding context window to keep recent conversation inside a configurable
  token budget.
- CLI for interactive chat and one-shot prompts.
- Unit tests mock process and network boundaries.

## Requirements

- Python 3.10+
- Ollama installed locally
- Gemma 2 pulled into Ollama:

```powershell
ollama pull gemma2
```

No Python runtime packages are required.

Install the local package in editable mode:

```powershell
python -m pip install -r requirements.txt
```

## Run

From this directory:

```powershell
offline-llm --prompt "Summarize why local LLMs help privacy."
```

Interactive mode:

```powershell
offline-llm
```

Optional settings:

```powershell
offline-llm --model gemma2 --host http://127.0.0.1:11434 --max-context-tokens 4096
```

## Test

```powershell
python -m pytest
```

The tests do not call a real Ollama server.
