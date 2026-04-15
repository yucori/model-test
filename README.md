# model-test

FarmOS 농산물 직판매 쇼핑몰 CS 챗봇 구성에 사용할 **임베딩 모델 × LLM 조합을 정량적으로 벤치마크**하는 도구.

## 목적

챗봇을 실제로 배포하기 전에, 어떤 임베딩 모델(RAG 검색)과 어떤 LLM(답변 생성)의 조합이 가장 좋은 CS 응답 품질을 내는지 측정한다.

- 임베딩 모델 선택 → RAG 검색 품질에 영향
- LLM 선택 → 답변 생성 품질에 영향
- 두 축을 분리해서 독립적으로 측정

## 스택

| 영역 | 기술 |
|------|------|
| 백엔드 | FastAPI, Python 3.12, ChromaDB, SSE |
| 프론트엔드 | React 18, TypeScript, TailwindCSS v4, TanStack Query |
| LLM | Anthropic Claude, Google Gemini, Ollama |
| 임베딩 | Ollama (nomic-embed-text, qwen3-embedding, embeddinggemma, bge-m3), 로컬 MiniLM, OpenAI |
| 판정 | LLM-as-Judge (Claude Haiku) |

## 시작하기

### 환경 변수

```bash
cd backend
cp .env.example .env
# .env 파일에 키 입력
```

```env
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

### 백엔드 실행

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -e .
uvicorn app.main:app --port 8765 --reload
```

### 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev       # http://localhost:5175
```

### Ollama 모델 준비 (권장)

```bash
# LLM
ollama pull llama3.2:3b
ollama pull gemma3:4b
ollama pull qwen2.5:7b

# 임베딩
ollama pull nomic-embed-text
ollama pull bge-m3          # 한국어 SOTA — 강력 권장
```

## 사용 흐름

1. **문서 업로드** (`/documents`) — PDF/DOCX 내부 문서를 올리고 임베딩 모델별로 벡터화
2. **테스트 질문 관리** (`/test-suite`) — 카테고리별 CS 질문 추가/삭제 (기본 20개 포함)
3. **테스트 실행** (`/run`) — 임베딩 모델 + LLM 조합 선택 → 실시간 진행 스트리밍
4. **결과 분석** (`/results`) — 임베딩 × LLM 매트릭스 점수표, 카테고리별 분석

## 평가 지표

Claude Haiku가 각 응답을 0–10점으로 채점:

| 지표 | 설명 |
|------|------|
| relevance | 질문과 답변의 관련성 |
| accuracy | 정보 정확성 |
| helpfulness | 실질적 도움 여부 |
| korean_fluency | 한국어 자연스러움 |
| overall | 종합 CS 품질 |

## 구조

```
model-test/
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI 앱, CORS 설정
│   │   ├── config.py            환경 변수 (pydantic-settings)
│   │   ├── schemas.py           Pydantic 모델 전체
│   │   ├── state.py             인메모리 상태 (DB 없음)
│   │   ├── routers/
│   │   │   ├── documents.py     문서 업로드 / 벡터화
│   │   │   ├── test_suite.py    질문 CRUD
│   │   │   ├── runs.py          테스트 실행 / SSE 스트림 / 비교
│   │   │   └── models_router.py 사용 가능 모델 목록
│   │   └── services/
│   │       ├── vector_store.py  ChromaDB (임베딩 모델별 독립 컬렉션)
│   │       ├── llm_clients.py   Anthropic / Gemini / Ollama 통합 클라이언트
│   │       ├── test_runner.py   (임베딩 × LLM) × 질문 매트릭스 실행
│   │       ├── evaluator.py     LLM-as-Judge 채점
│   │       └── document_processor.py  PDF/DOCX 파싱 + 청킹
│   ├── data/
│   │   └── default_questions.json  기본 CS 질문 20개
│   └── pyproject.toml
└── frontend/
    └── src/
        ├── pages/
        │   ├── DashboardPage.tsx
        │   ├── DocumentsPage.tsx
        │   ├── TestSuitePage.tsx
        │   ├── RunTestPage.tsx
        │   └── ResultDetailPage.tsx
        ├── components/
        │   ├── common/ModelBadge.tsx   LLMBadge / EmbeddingBadge
        │   └── layout/
        ├── lib/api.ts               API 호출 함수
        └── types/index.ts           TypeScript 타입 정의
```

## 주요 설계 결정

- **DB 없음**: 모든 상태를 Python dict로 인메모리 관리. 서버 재시작 시 런 결과 초기화됨 (문서 청킹은 ChromaDB 파일로 유지)
- **임베딩 모델별 독립 컬렉션**: ChromaDB 컬렉션을 `docs__{emb_id}` 형식으로 분리해 차원 충돌 방지
- **asyncio.to_thread**: ChromaDB 동기 API를 비동기 이벤트 루프에서 안전하게 호출
- **SSE 스트리밍**: 테스트 진행 상황을 실시간으로 프론트엔드에 push
