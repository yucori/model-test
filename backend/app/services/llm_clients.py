"""
Unified async LLM client for generation models (Anthropic, OpenAI, Ollama, Google Gemini).

Note: this module is ONLY for text generation.
Embedding is handled separately in vector_store.py.
"""
import time
from dataclasses import dataclass

from app.config import settings
from app.schemas import LLMModelInfo, LLMProvider


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


# ── Provider detection ────────────────────────────────────────────────────────

def _provider_for(model_id: str) -> str:
    if model_id.startswith("claude"):
        return "anthropic"
    if model_id.startswith(("gpt-", "o1", "o3", "text-")):
        return "openai"
    if model_id.startswith("gemini"):
        return "google"
    return "ollama"


# ── Anthropic ─────────────────────────────────────────────────────────────────

async def _call_anthropic(model_id: str, system: str, messages: list[dict]) -> LLMResponse:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    t0 = time.perf_counter()
    r = await client.messages.create(
        model=model_id, max_tokens=1024, system=system, messages=messages,
    )
    return LLMResponse(
        content=r.content[0].text if r.content else "",
        prompt_tokens=r.usage.input_tokens,
        completion_tokens=r.usage.output_tokens,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


# ── OpenAI ────────────────────────────────────────────────────────────────────

async def _call_openai(model_id: str, system: str, messages: list[dict]) -> LLMResponse:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    t0 = time.perf_counter()
    r = await client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system}] + messages,
        max_tokens=1024,
    )
    usage = r.usage
    return LLMResponse(
        content=r.choices[0].message.content or "",
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


# ── Google Gemini ─────────────────────────────────────────────────────────────

async def _call_gemini(model_id: str, system: str, messages: list[dict]) -> LLMResponse:
    from google import genai
    from google.genai import types as genai_types
    client = genai.Client(api_key=settings.gemini_api_key)
    user_content = messages[-1]["content"] if messages else ""
    t0 = time.perf_counter()
    r = await client.aio.models.generate_content(
        model=model_id,
        contents=user_content,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1024,
        ),
    )
    elapsed = (time.perf_counter() - t0) * 1000
    usage = r.usage_metadata
    return LLMResponse(
        content=r.text or "",
        prompt_tokens=usage.prompt_token_count if usage else 0,
        completion_tokens=usage.candidates_token_count if usage else 0,
        latency_ms=elapsed,
    )


# ── Ollama ────────────────────────────────────────────────────────────────────

async def _call_ollama(model_id: str, system: str, messages: list[dict]) -> LLMResponse:
    import httpx
    t0 = time.perf_counter()
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=120) as c:
        r = await c.post("/api/chat", json={
            "model": model_id,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
        })
        if not r.is_success:
            # Surface Ollama's actual error message instead of a generic HTTP error
            try:
                detail = r.json().get("error", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(f"Ollama {r.status_code}: {detail}")
    data = r.json()
    usage = data.get("usage", {})
    return LLMResponse(
        content=data.get("message", {}).get("content", ""),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


# ── Public API ────────────────────────────────────────────────────────────────

async def call_model(model_id: str, system: str, question: str) -> LLMResponse:
    messages = [{"role": "user", "content": question}]
    provider = _provider_for(model_id)
    if provider == "anthropic":
        return await _call_anthropic(model_id, system, messages)
    if provider == "openai":
        return await _call_openai(model_id, system, messages)
    if provider == "google":
        return await _call_gemini(model_id, system, messages)
    return await _call_ollama(model_id, system, messages)


# ── Model catalogue ───────────────────────────────────────────────────────────

# Ollama models that are embedding-only and must not appear in the LLM list
_OLLAMA_EMBEDDING_MODELS = {
    "nomic-embed-text",
    "qwen3-embedding",
    "embeddinggemma",
    "bge-m3",
    "mxbai-embed-large",
    "all-minilm",
    "snowflake-arctic-embed",
}


def _is_ollama_embedding_only(model_name: str) -> bool:
    """Return True if this Ollama model is embedding-only (not a chat/LLM model)."""
    base = model_name.split(":")[0].lower()
    return base in _OLLAMA_EMBEDDING_MODELS


async def _ollama_model_ids() -> list[str]:
    try:
        import httpx
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=5) as c:
            r = await c.get("/api/tags")
            return [
                m["name"] for m in r.json().get("models", [])
                if not _is_ollama_embedding_only(m["name"])
            ]
    except Exception:
        return []


async def get_available_llm_models() -> list[LLMModelInfo]:
    models: list[LLMModelInfo] = []

    anthropic_ok = bool(settings.anthropic_api_key)
    for mid, name in [
        ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
        ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
        ("claude-opus-4-6", "Claude Opus 4.6"),
    ]:
        models.append(LLMModelInfo(
            id=mid, name=name, provider=LLMProvider.ANTHROPIC,
            available=anthropic_ok,
            description="" if anthropic_ok else "ANTHROPIC_API_KEY 미설정",
        ))

    gemini_ok = bool(settings.gemini_api_key)
    for mid, name in [
        ("gemini-2.0-flash", "Gemini 2.0 Flash"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ]:
        models.append(LLMModelInfo(
            id=mid, name=name, provider=LLMProvider.GOOGLE,
            available=gemini_ok,
            description="" if gemini_ok else "GEMINI_API_KEY 미설정",
        ))

    for mid in await _ollama_model_ids():
        models.append(LLMModelInfo(
            id=mid, name=f"{mid} (Ollama)",
            provider=LLMProvider.OLLAMA, available=True,
        ))

    return models
