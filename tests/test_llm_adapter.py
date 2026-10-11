import json

import pytest

from src.llm.openai_compatible import LLMSettings, _extract_json_object, build_llm_client_from_env


def test_extract_clean_json():
    assert _extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_fenced_json():
    assert _extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_from_surrounding_text():
    assert _extract_json_object('Result: {"a": 1}') == {"a": 1}


def test_build_client_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LLM_ENABLED", raising=False)
    assert build_llm_client_from_env() is None


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_MODEL", "demo-model")
    monkeypatch.setenv("LLM_EXTRA_HEADERS_JSON", json.dumps({"X-Test": "yes"}))
    settings = LLMSettings.from_env()
    assert settings.base_url == "https://example.test/v1"
    assert settings.model == "demo-model"
    assert settings.extra_headers == {"X-Test": "yes"}


def test_endpoint_is_composed_from_base_url():
    from src.llm.openai_compatible import OpenAICompatibleStructuredClient
    settings = LLMSettings(
        base_url="https://example.test/v1",
        api_key="secret",
        model="demo-model",
    )
    client = OpenAICompatibleStructuredClient(settings)
    assert client.endpoint == "https://example.test/v1/chat/completions"
