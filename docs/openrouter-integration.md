# OpenRouter 통합 작업 기록

> 작업일: 2026-04-23  
> 목적: 사용자가 OpenRouter를 통해 GPT-5 Nano 등 OpenAI 호환 모델을 LLM으로 사용할 수 있도록 지원 추가

---

## 배경

기존 `llm_clients.py`에 `_call_openai()`가 존재했지만 `call_model()`에서 라우팅되지 않아 사실상 미사용 상태였다. 사용자가 OpenRouter + GPT-5 Nano를 사용 중임을 확인해 정식 통합을 수행했다.

OpenRouter는 OpenAI-compatible API(`https://openrouter.ai/api/v1`)를 제공하므로 `openai.AsyncOpenAI(base_url=..., api_key=...)`만으로 연결 가능하다.

---

## 변경 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `backend/.env` | `OPENROUTER_API_KEY` 플레이스홀더 추가 |
| `backend/app/config.py` | `openrouter_api_key`, `openrouter_base_url` 필드 추가 |
| `backend/app/schemas.py` | `LLMProvider.OPENROUTER = "openrouter"` 추가 |
| `backend/app/services/llm_clients.py` | `_call_openrouter()` 구현, `_provider_for()` 분기 추가, `call_model()` 라우팅, `_OPENROUTER_MODELS` 카탈로그 추가 |
| `backend/app/routers/pipeline.py` | `_generate_one_case()`에 `openrouter:` 분기 추가 (테스트케이스 자동생성 judge 지원) |
| `frontend/src/types/index.ts` | `LLMProvider`에 `'openrouter'` 추가 |
| `frontend/src/components/common/ModelBadge.tsx` | OpenRouter 뱃지 색상(주황) 및 라벨 추가 |
| `frontend/src/pages/EmbeddingEvalPage.tsx` | 자동생성 judge 모델 드롭다운 추가 (Gemini / OpenRouter 옵션), `genJudge` state 추가 |

---

## 모델 ID 규칙

```
openrouter:<provider/model>

예시:
  openrouter:openai/gpt-5-nano
  openrouter:openai/gpt-4.1-nano
  openrouter:openai/gpt-4o-mini
  openrouter:anthropic/claude-haiku-4-5
```

`_call_openrouter()`에서 `model_id.removeprefix("openrouter:")`로 실제 모델명을 추출해 API에 전달한다.

---

## 설정 방법

`backend/.env`에 실제 키 입력:

```env
OPENROUTER_API_KEY=sk-or-v1-...
```

키가 설정되면:
- Run 페이지 LLM 목록에 OpenRouter 모델 자동 노출 (`available=True`)
- 키 미설정 시 `available=False`, "OPENROUTER_API_KEY 미설정" 설명 표시

---

## 카탈로그에 포함된 모델

| 모델 ID | 표시명 |
|---------|--------|
| `openrouter:openai/gpt-5-nano` | GPT-5 Nano |
| `openrouter:openai/gpt-4.1-nano` | GPT-4.1 Nano |
| `openrouter:openai/gpt-4.1-mini` | GPT-4.1 Mini |
| `openrouter:openai/gpt-4.1` | GPT-4.1 |
| `openrouter:openai/gpt-4o-mini` | GPT-4o Mini |
| `openrouter:openai/gpt-4o` | GPT-4o |
| `openrouter:anthropic/claude-haiku-4-5` | Claude Haiku 4.5 |
| `openrouter:anthropic/claude-sonnet-4-5` | Claude Sonnet 4.5 |
| `openrouter:meta-llama/llama-3.1-8b-instruct` | Llama 3.1 8B |
| `openrouter:meta-llama/llama-3.3-70b-instruct` | Llama 3.3 70B |

새 모델 추가 시 `llm_clients.py`의 `_OPENROUTER_MODELS` 리스트에만 추가하면 된다.

---

## 테스트케이스 자동생성 judge로 사용 시

`EmbeddingEvalPage`의 judge 드롭다운에서 OpenRouter 모델 선택 가능.  
`pipeline.py`의 `_generate_one_case()`가 `openrouter:` 접두사를 감지해 OpenRouter API로 라우팅한다.
