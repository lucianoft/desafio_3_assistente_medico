"""Configuracoes centrais do projeto (lidas de variaveis de ambiente / .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

try:  # python-dotenv e opcional
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:  # pragma: no cover
    pass


def _path(env: str, default: Path) -> Path:
    value = os.getenv(env)
    return Path(value) if value else default


@dataclass(frozen=True)
class Settings:
    # Diretorios
    data_dir: Path = ROOT_DIR / "data"
    synthetic_dir: Path = ROOT_DIR / "data" / "synthetic"
    protocols_dir: Path = ROOT_DIR / "data" / "synthetic" / "protocolos"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    db_path: Path = _path("HOSPITAL_DB_PATH", ROOT_DIR / "data" / "synthetic" / "hospital.db")
    logs_dir: Path = _path("LOGS_DIR", ROOT_DIR / "logs")
    reports_dir: Path = ROOT_DIR / "reports"
    docs_dir: Path = ROOT_DIR / "docs"

    # LLM
    llm_backend: str = os.getenv("LLM_BACKEND", "fake")  # fake | hf | openai
    base_model: str = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_path: Path = _path("ADAPTER_PATH", ROOT_DIR / "models" / "qwen2.5-1.5b-hospital-lora")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "384"))
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None  # ex.: Groq (OpenAI-compatible)

    # Assistente
    retriever_k: int = int(os.getenv("RETRIEVER_K", "4"))


settings = Settings()
