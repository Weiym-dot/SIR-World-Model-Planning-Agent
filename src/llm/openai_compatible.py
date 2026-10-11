from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_env_file(path: str | None = None) -> None:
    """Tiny .env loader so the project has no provider SDK dependency.

    Existing environment variables always win. This intentionally supports the
    simple KEY=VALUE format used by `.env.example`.
    """
    env_path = Path(path or ".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class LLMSettings:
    """Settings for any OpenAI-compatible `/chat/completions` endpoint."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 45.0
    temperature: float = 0.0
    extra_headers: dict[str, str] | None = None

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "LLMSettings":
        load_env_file(env_file)
        base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()

        missing = [
            name
            for name, value in (
                ("LLM_BASE_URL", base_url),
                ("LLM_API_KEY", api_key),
                ("LLM_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise ValueError("Missing LLM configuration: " + ", ".join(missing))

        timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

        headers_raw = os.getenv("LLM_EXTRA_HEADERS_JSON", "").strip()
        extra_headers: dict[str, str] | None = None
        if headers_raw:
            parsed = json.loads(headers_raw)
            if not isinstance(parsed, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
            ):
                raise ValueError("LLM_EXTRA_HEADERS_JSON must be a JSON object of string:string pairs")
            extra_headers = parsed

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            extra_headers=extra_headers,
        )


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("LLM returned an empty response")

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON response must be an object")
        return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        parsed = json.loads(fenced.group(1))
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON response must be an object")
        return parsed

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON response must be an object")
        return parsed

    raise ValueError(f"Could not extract a JSON object from LLM response: {text[:200]!r}")


class OpenAICompatibleStructuredClient:
    """SDK-free adapter for OpenAI-compatible Chat Completions APIs.

    Provider switching is configuration-only as long as the provider exposes
    `/chat/completions` with the common OpenAI response shape.
    """

    def __init__(self, settings: LLMSettings):
        self.settings = settings

    @property
    def endpoint(self) -> str:
        return f"{self.settings.base_url}/chat/completions"

    def complete_json(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one valid JSON object and no markdown or extra text. "
                        "Follow the user's requested keys and value constraints."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.extra_headers:
            headers.update(self.settings.extra_headers)

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach LLM API at {self.endpoint}: {exc.reason}") from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected OpenAI-compatible response shape: {body}") from exc

        if not isinstance(content, str):
            raise ValueError("Expected string content from LLM chat completion")
        return _extract_json_object(content)


def build_llm_client_from_env(env_file: str | None = None) -> OpenAICompatibleStructuredClient | None:
    """Return a real client only when `LLM_ENABLED=true`; otherwise use fallback."""
    load_env_file(env_file)
    enabled = os.getenv("LLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None
    return OpenAICompatibleStructuredClient(LLMSettings.from_env(env_file))
