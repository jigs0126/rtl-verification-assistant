"""
Application configuration.

What it is:
    A single, typed source of truth for every environment-variable-driven
    setting in the application (API keys, model names, file paths,
    RAG parameters).

Why it is required:
    Without this, configuration values would be read from os.environ in
    scattered locations throughout ingestion, embeddings, retrieval, and
    LLM code. That makes it hard to know what configuration the app
    actually depends on, and easy to typo an env var name with no error
    until runtime. Pydantic's BaseSettings validates types and required
    fields at startup instead.

Input:
    Environment variables, loaded from a `.env` file in the project root
    (see `.env.example` for the full list) or from the real process
    environment.

Output:
    A single `settings` object, importable anywhere in the app, with
    typed, validated fields.

How it connects to the next component:
    Every other module (ingestion, embeddings, vector_store, llm client)
    imports `settings` from here instead of reading environment variables
    directly.

What would happen if it were removed:
    Every module would need its own os.environ / dotenv loading logic,
    with no shared validation and no single place to see what
    configuration the whole application depends on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application configuration, loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM configuration ---------------------------------------------
    llm_api_key: str = Field(default="", description="API key for the LLM provider.")
    llm_model: str = Field(
        default="gemini-3.8-flash",
        description="Gemini model identifier (e.g. gemini-3.8-flash).",
    )
    llm_temperature: float = Field(
        default=0.2,
        description="Sampling temperature for LLM calls.",
    )

    # --- Embedding configuration -----------------------------------------
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "Embedding model name. Default is a local sentence-transformers "
            "model — no embedding API cost after the model weights are "
            "downloaded once. Can be swapped for an API-based embedding "
            "model by changing this value."
        ),
    )

    # --- Storage / paths --------------------------------------------------
    vector_db_path: Path = Field(
        default=Path("data/vector_store"),
        description="Directory where the local Chroma vector store persists.",
    )
    documents_path: Path = Field(
        default=Path("."),
        description=(
            "Root of the project being indexed. Must contain rtl/, docs/, "
            "and optionally bugs/ subdirectories. Points at the ALU project "
            "today; can be repointed at RV-CORDIX later without code changes."
        ),
    )
    log_path: Path = Field(
        default=Path("data/logs"),
        description="Directory where query logs are written.",
    )

    # --- RAG parameters -----------------------------------------------------
    chunk_size: int = Field(default=800, description="Target characters per chunk.")
    chunk_overlap: int = Field(default=120, description="Character overlap between chunks.")
    top_k: int = Field(default=4, description="Number of chunks retrieved per query.")

    def masked_api_key(self) -> str:
        """
        Return a masked representation of the LLM API key, safe to print
        or log. Never returns the real key.
        """
        key = self.llm_api_key
        if not key:
            return "<not set>"
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:5]}{'*' * (len(key) - 9)}{key[-4:]}"

    def __str__(self) -> str:  # pragma: no cover - simple display helper
        return (
            "Settings(\n"
            f"  llm_model={self.llm_model!r}\n"
            f"  llm_temperature={self.llm_temperature}\n"
            f"  llm_api_key={self.masked_api_key()!r}\n"
            f"  embedding_model={self.embedding_model!r}\n"
            f"  vector_db_path={self.vector_db_path}\n"
            f"  documents_path={self.documents_path}\n"
            f"  log_path={self.log_path}\n"
            f"  chunk_size={self.chunk_size}\n"
            f"  chunk_overlap={self.chunk_overlap}\n"
            f"  top_k={self.top_k}\n"
            ")"
        )


settings = Settings()
