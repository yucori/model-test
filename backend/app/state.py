"""
In-memory application state. No persistent database — data lives for the
duration of the server process.
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Optional
from .schemas import DocumentInfo, TestQuestion, TestRun, TestResult


# ── Documents ──────────────────────────────────────────────────────────────
documents: dict[str, DocumentInfo] = {}

# ── Test suite ─────────────────────────────────────────────────────────────
test_questions: dict[str, TestQuestion] = {}


def _load_default_questions() -> None:
    """Load the bundled default question set on startup."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "default_questions.json")
    try:
        with open(path, encoding="utf-8") as f:
            for q in json.load(f):
                test_questions[q["id"]] = TestQuestion(**q)
    except Exception as exc:
        print(f"[state] Failed to load default questions: {exc}")


# ── Test runs & results ─────────────────────────────────────────────────────
runs: dict[str, TestRun] = {}
results: dict[str, list[TestResult]] = {}   # run_id → [TestResult]
run_queues: dict[str, asyncio.Queue] = {}   # run_id → SSE event queue
