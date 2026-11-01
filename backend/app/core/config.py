"""Application-wide configuration.

Every external dependency is optional by design:

* No ``DATABASE_URL``  -> local SQLite file (async via aiosqlite).
* No ``REDIS_URL``     -> in-process cache with the same interface.
* No LLM API key       -> deterministic, template-based narration.

This keeps the service runnable for development and CI while remaining a
genuine Postgres + Redis + LLM deployment when those are provided.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent

LLMProvider = Literal["openai", "gemini", "anthropic", "openrouter", "none"]


class Settings(BaseSettings):
    """Runtime settings, loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Application ------------------------------------------------------
    app_name: str = "Aarogya Clinical Intelligence API"
    version: str = "1.0.0"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # -- Security ---------------------------------------------------------
    secret_key: str = Field(
        default="dev-only-insecure-secret",
        description="Used to sign API tokens. MUST be overridden in production.",
    )
    api_keys: list[str] = Field(
        default_factory=list,
        description="When set, write endpoints require one of these in X-API-Key.",
    )

    # -- Database ---------------------------------------------------------
    database_url: str | None = None
    sqlite_filename: str = "aarogya_dev.db"
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # -- Cache ------------------------------------------------------------
    redis_url: str | None = None
    cache_ttl_seconds: int = 300

    # -- LLM --------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    llm_provider: LLMProvider = "none"
    llm_model: str | None = None
    llm_temperature: float = 0.2
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None

    # -- Model artifacts --------------------------------------------------
    model_dir: Path = PROJECT_DIR
    embedding_dimensions: int = 512

    # -- Uploads ----------------------------------------------------------
    max_upload_mb: int = 10

    # -- HTTP -------------------------------------------------------------
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def effective_database_url(self) -> str:
        """Postgres when configured, otherwise a local SQLite file."""
        if self.database_url:
            return self.database_url
        return f"sqlite+aiosqlite:///{(BACKEND_DIR / self.sqlite_filename).as_posix()}"

    @property
    def uses_postgres(self) -> bool:
        return self.effective_database_url.startswith(("postgresql", "postgres"))

    def resolve_llm(self) -> tuple[LLMProvider, str, str] | None:
        """Return (provider, model, api_key) for the first configured provider."""
        candidates: list[tuple[LLMProvider, str | None, str | None, str]] = [
            ("openai", self.openai_api_key, self.llm_model, "gpt-4o-mini"),
            ("gemini", self.gemini_api_key, self.llm_model, "gemini-1.5-flash"),
            ("anthropic", self.anthropic_api_key, self.llm_model, "claude-3-5-sonnet-latest"),
            ("openrouter", self.openrouter_api_key, self.llm_model, "openai/gpt-4o-mini"),
        ]
        if self.llm_provider != "none":
            candidates.sort(key=lambda item: item[0] != self.llm_provider)
        for provider, key, model, default_model in candidates:
            if key:
                return provider, (model or default_model), key
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
