"""
LLM-as-Judge evaluator for chatbot responses.
Uses a judge model to score responses 0-10
across four dimensions relevant to Korean CS chatbots.

Supports Anthropic, OpenAI, and Ollama judge models.
Small Ollama models (≤7B) may produce irregular JSON; _extract_json()
applies multiple fallback strategies before giving up.
"""
import json
import re
from app.schemas import EvaluationScores
from app.config import settings

# Instruct the model to output JSON only, with an explicit example.
# The example is important for smaller Ollama models that need a concrete template.
_JUDGE_PROMPT = """You are an expert evaluator for a Korean agricultural e-commerce CS chatbot.

Score the chatbot response below on each criterion from 0 to 10 (decimals allowed).
Be strict: reserve 9-10 for truly excellent responses.

Criteria:
1. relevance        — How directly does the response address the customer question?
2. accuracy         — How factually correct is the information given the reference material?
3. helpfulness      — Does the customer receive concrete, actionable help?
4. korean_fluency   — Is the Korean natural, polite, and appropriate for customer service?
5. source_citation  — Does the response correctly identify WHERE in the document the answer comes from?
                      Score 0: no source mentioned at all.
                      Score 3-5: vaguely references a document or policy without specifics.
                      Score 6-8: correctly names the relevant policy document or section/table.
                      Score 9-10: pinpoints the exact section, clause, or table row the information came from.
                      NOTE: Both prose-structured (조/항) and table-structured documents must be treated equally.
                      A response citing a table row or table header correctly is as good as citing an article number.
6. overall          — Overall CS response quality combining all criteria above.

Customer question:
{question}

Reference material (RAG retrieval):
{context}

Chatbot response:
{response}

Output ONLY the following JSON object. No explanation, no markdown fences:
{{"relevance": 7.5, "accuracy": 8, "helpfulness": 7, "korean_fluency": 9, "source_citation": 6, "overall": 7.5, "reasoning": "..."}}"""


async def evaluate_response(
    question: str,
    context: list[str],
    response: str,
    judge_model: str,
) -> EvaluationScores | None:
    """Return EvaluationScores or None if evaluation fails."""
    try:
        context_text = "\n---\n".join(context) if context else "(no reference material)"
        prompt = _JUDGE_PROMPT.format(
            question=question,
            context=context_text,
            response=response,
        )

        raw = await _call_judge(judge_model, prompt)
        data = _extract_json(raw)
        if data is None:
            print(f"[evaluator] Could not extract JSON from judge output: {raw[:200]!r}")
            return None

        return EvaluationScores(
            relevance=_clamp(data.get("relevance", 0)),
            accuracy=_clamp(data.get("accuracy", 0)),
            helpfulness=_clamp(data.get("helpfulness", 0)),
            korean_fluency=_clamp(data.get("korean_fluency", 0)),
            source_citation=_clamp(data.get("source_citation", 0)),
            overall=_clamp(data.get("overall", 0)),
            reasoning=str(data.get("reasoning", "")),
        )
    except Exception as exc:
        print(f"[evaluator] Evaluation failed ({judge_model}): {exc}")
        return None


# ── JSON extraction ────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict | None:
    """Try several strategies to extract a JSON object from raw model output.

    Smaller Ollama models often wrap JSON in prose or markdown fences.
    Strategy order (cheapest → most forgiving):
      1. Direct json.loads on stripped text
      2. Strip markdown code fences (```json ... ```)
      3. Find first {...} block with regex
      4. Extract individual numeric fields with regex (last resort)
    """
    text = raw.strip()

    # Strategy 1: plain JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    fenced = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    fenced = re.sub(r'\s*```$', '', fenced, flags=re.MULTILINE).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    # Strategy 3: extract first {...} block
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: regex field extraction — works even when JSON is malformed
    fields = {
        "relevance":       r'"?relevance"?\s*[:=]\s*([\d.]+)',
        "accuracy":        r'"?accuracy"?\s*[:=]\s*([\d.]+)',
        "helpfulness":     r'"?helpfulness"?\s*[:=]\s*([\d.]+)',
        "korean_fluency":  r'"?korean_fluency"?\s*[:=]\s*([\d.]+)',
        "source_citation": r'"?source_citation"?\s*[:=]\s*([\d.]+)',
        "overall":         r'"?overall"?\s*[:=]\s*([\d.]+)',
        "reasoning":       r'"?reasoning"?\s*[:=]\s*"([^"]*)"',
    }
    result: dict = {}
    for key, pattern in fields.items():
        hit = re.search(pattern, text, re.IGNORECASE)
        if hit:
            result[key] = hit.group(1)

    # Accept partial extraction only if all six numeric fields are present
    required = {"relevance", "accuracy", "helpfulness", "korean_fluency", "source_citation", "overall"}
    if required.issubset(result.keys()):
        return result

    return None


def _clamp(value) -> float:
    """Convert to float and clamp to [0, 10]."""
    try:
        return max(0.0, min(10.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


# ── Provider routing ───────────────────────────────────────────────────────────

async def _call_judge(model_id: str, prompt: str) -> str:
    if model_id.startswith("gemini"):
        return await _call_gemini_judge(model_id, prompt)
    return await _call_ollama_judge(model_id, prompt)


async def _call_ollama_judge(model_id: str, prompt: str) -> str:
    import httpx
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=120) as c:
        r = await c.post("/api/chat", json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
        })
        r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


async def _call_gemini_judge(model_id: str, prompt: str) -> str:
    from google import genai
    from google.genai import types as genai_types
    client = genai.Client(api_key=settings.gemini_api_key)
    r = await client.aio.models.generate_content(
        model=model_id,
        contents=prompt,
        config=genai_types.GenerateContentConfig(max_output_tokens=512),
    )
    return r.text or ""
