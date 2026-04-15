from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import state
from app.routers import documents, test_suite, runs, models_router, debug


@asynccontextmanager
async def lifespan(app: FastAPI):
    state._load_default_questions()
    yield


app = FastAPI(
    title="Model Test API",
    description=(
        "Quantitative benchmarking of LLM + Embedding model combinations "
        "for FarmOS Shopping Mall CS chatbot"
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5175", "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(test_suite.router)
app.include_router(runs.router)
app.include_router(models_router.router)
app.include_router(debug.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "model-test", "version": "0.2.0"}
