"""
Application Context

What it is: the single place that wires together everything the
Streamlit UI (and any future CLI) needs — settings, embedding model,
vector store, retriever, LLM client, TestbenchGenerator, DebugAnalyzer —
with the error handling required by §23 (missing API key, unavailable
embedding model, empty corpus, ChromaDB errors) turned into clear,
user-facing messages instead of raw stack traces.

Why required: app/main.py (the Streamlit UI) should describe *what to
show*, not *how to construct a TestbenchGenerator*. Keeping construction
here means a CLI or test harness can reuse the exact same wiring.

Input: Settings (from app.config.settings).
Output: an AppContext dataclass holding every component the UI needs,
or an AppError with a user-safe message if construction failed.

How it connects: app/main.py calls build_app_context(settings) once per
session and uses the returned components; on AppError it renders the
message instead of the normal UI.

What would happen if it were removed: the Streamlit UI would have to
import and wire every component itself, duplicating this logic if a
second UI (CLI, API) were ever added, and error handling would likely
end up inconsistent between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config.settings import Settings
from app.embeddings.embedder import EmbeddingModel, get_embedding_model
from app.llm.client import GeminiLLMClient, LLMClient, LLMError
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.verification.debug_analyzer import DebugAnalyzer
from app.verification.rtl_workflow import PlanTestbenchGenerator, RTLGenerator, SimulationDebugger
from app.verification.testbench_generator import TestbenchGenerator


class AppError(Exception):
    """A user-safe error message for anything that goes wrong while
    building the application context — never wraps a raw traceback or
    exposes settings.llm_api_key."""


@dataclass
class AppContext:
    settings: Settings
    embedding_model: EmbeddingModel
    vector_store: VectorStore
    retriever: Retriever
    llm_client: LLMClient
    testbench_generator: TestbenchGenerator
    debug_analyzer: DebugAnalyzer
    using_fallback_embeddings: bool
    using_fake_llm: bool
    # RTL workflow (spec -> RTL -> testbench -> log debug); defaults keep
    # any code that builds AppContext by hand working.
    rtl_generator: Optional[RTLGenerator] = None
    plan_tb_generator: Optional[PlanTestbenchGenerator] = None
    sim_debugger: Optional[SimulationDebugger] = None


def build_app_context(settings: Settings, allow_fake_llm: bool = True) -> AppContext:
    """
    Build every component the UI needs. Never raises for "expected"
    configuration gaps (missing API key, embedding model unavailable) —
    those degrade to a clearly-labeled fallback (FakeLLMClient) so the
    retrieval/knowledge-base parts of the UI stay usable even without a
    real API key, per this being a human-in-the-loop assistant, not
    something that should hard-fail just because generation isn't wired
    up yet. Only genuinely broken configuration (e.g. an unreadable
    vector_db_path) raises AppError.
    """
    try:
        embedding_model = get_embedding_model(settings.embedding_model)
    except Exception as exc:  # noqa: BLE001
        raise AppError(f"Could not initialize the embedding model: {exc}") from exc

    using_fallback_embeddings = type(embedding_model).__name__ == "DeterministicHashingEmbeddingModel"

    try:
        vector_store = VectorStore(path=Path(settings.vector_db_path))
    except Exception as exc:  # noqa: BLE001
        raise AppError(f"Could not open the local vector store at {settings.vector_db_path}: {exc}") from exc

    retriever = Retriever(vector_store, embedding_model, default_top_k=settings.top_k)

    using_fake_llm = False
    llm_client: LLMClient
    try:
        llm_client = GeminiLLMClient(
            api_key=settings.llm_api_key, model=settings.llm_model, temperature=settings.llm_temperature
        )
    except LLMError:
        if not allow_fake_llm:
            raise AppError(
                "No LLM API key configured. Set LLM_API_KEY in your .env file (see .env.example)."
            )
        using_fake_llm = True
        llm_client = _make_unconfigured_fake_llm()

    log_path = Path(settings.log_path)
    testbench_generator = TestbenchGenerator(retriever, llm_client, top_k=settings.top_k, log_path=log_path)
    debug_analyzer = DebugAnalyzer(retriever, llm_client, top_k=settings.top_k, log_path=log_path)

    return AppContext(
        settings=settings,
        embedding_model=embedding_model,
        vector_store=vector_store,
        retriever=retriever,
        llm_client=llm_client,
        testbench_generator=testbench_generator,
        debug_analyzer=debug_analyzer,
        using_fallback_embeddings=using_fallback_embeddings,
        using_fake_llm=using_fake_llm,
        rtl_generator=RTLGenerator(llm_client),
        plan_tb_generator=PlanTestbenchGenerator(llm_client),
        sim_debugger=SimulationDebugger(llm_client),
    )


def _make_unconfigured_fake_llm() -> LLMClient:
    """
    A FakeLLMClient that raises a clear, user-safe LLMError instead of
    returning a canned response — used only when no real API key is
    configured, so the UI can still be exercised (retrieval inspection,
    knowledge base) without silently pretending generation works.
    """
    from app.llm.client import FakeLLMClient

    def _raise(_prompt: str) -> str:
        raise LLMError(
            "No LLM API key configured. Set LLM_API_KEY in your .env file "
            "to enable testbench generation and debug analysis."
        )

    return FakeLLMClient(response_fn=_raise)
