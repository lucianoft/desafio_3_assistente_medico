"""Anonimizacao / pseudonimizacao de textos clinicos.

Estrategia (em camadas):
  1. Regex para identificadores estruturados: CPF, CNS, telefone, e-mail, CEP e datas.
  2. Nomes proprios: lista de nomes conhecidos (cadastro) + padroes de contexto
     ("Paciente Fulano de Tal", "Sr./Sra. Fulano").
  3. Pseudonimizacao do identificador do paciente com hash salgado (permite ligar registros
     sem expor o identificador original).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("CNS", re.compile(r"\b[1-9]\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\b")),
    ("TELEFONE", re.compile(r"\(?\b\d{2}\)?\s?9?\d{4}-?\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("CEP", re.compile(r"\b\d{5}-\d{3}\b")),
    ("DATA", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")),
]

_NAME = r"[A-Z][a-zà-ÿ]+(?:\s+(?:da|de|do|dos|das|e)?\s*[A-Z][a-zà-ÿ]+){1,4}"
CONTEXT_NAME = re.compile(rf"\b(Paciente|Sr\.?|Sra\.?|Dona|Senhor|Senhora)\s+({_NAME})")


@dataclass
class AnonymizationResult:
    text: str
    replacements: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.replacements.values())


def pseudonymize_id(identifier: str, salt: str = "fiap-fase3") -> str:
    return "PAC-" + hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()[:10].upper()


def anonymize_text(text: str, known_names: list[str] | None = None) -> AnonymizationResult:
    counts: dict[str, int] = {}

    def _sub(label: str, pattern: re.Pattern, value: str, repl: str | None = None) -> str:
        new, n = pattern.subn(repl or f"[{label}]", value)
        if n:
            counts[label] = counts.get(label, 0) + n
        return new

    # nomes conhecidos primeiro (mais especificos)
    for name in sorted(known_names or [], key=len, reverse=True):
        text = _sub("NOME", re.compile(re.escape(name), re.IGNORECASE), text)
    text = _sub("NOME", CONTEXT_NAME, text, r"\1 [NOME]")
    for label, pattern in PATTERNS:
        text = _sub(label, pattern, text)
    return AnonymizationResult(text=text, replacements=counts)


def contains_pii(text: str) -> bool:
    return any(p.search(text) for _, p in PATTERNS) or bool(CONTEXT_NAME.search(text))
