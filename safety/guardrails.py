"""Guardrails: limites de atuacao do assistente (PROT-008).

Entrada:  detecta pedido de prescricao/dose/alta, tentativa de burlar regras (prompt injection)
          e dados pessoais digitados pelo medico (mascarados antes de ir ao modelo).
Saida:    detecta linguagem prescritiva (verbo imperativo + dose), resposta sem fonte e PII;
          adiciona aviso de validacao medica e marca a resposta como pendente de validacao.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from data_pipeline.anonymize import anonymize_text, contains_pii

LIMITES_ATUACAO = [
    "Nunca prescrever medicamentos nem definir doses.",
    "Nunca decidir alta, internacao ou transferencia.",
    "Toda sugestao de conduta fica pendente de validacao medica.",
    "Responder apenas com base nos protocolos internos e no prontuario; citar a fonte.",
    "Nao enviar ao modelo nome, CPF ou contato do paciente.",
]

DISCLAIMER = ("Aviso: sugestao gerada por assistente virtual com base nos protocolos internos. "
              "Nao constitui prescricao e depende de validacao do medico responsavel.")

_PRESCRIPTION_REQUEST = re.compile(
    r"\b(prescrev\w*|receit\w*|posologia|qual (a )?dose|quantos? (mg|ml|ui|unidades)|que (rem[eé]dio|medicamento) "
    r"(devo )?(dar|usar|administrar)|indique o (rem[eé]dio|medicamento)|(liberar|dar|de) (a )?alta)\b",
    re.IGNORECASE,
)
_INJECTION = re.compile(r"\b(ignore|esque[cç]a|desconsidere|desative)\b.{0,40}\b(regras?|instru[cç][oõ]es|pol[ií]tica)",
                        re.IGNORECASE)
# dose de medicamento (exclui unidades de exame laboratorial como mg/dL e g/dL)
_DOSE = re.compile(r"\b\d+([.,]\d+)?\s?(mg|mcg|g|ml|ui|unidades|gotas)\b(?!\s*/\s*dl)", re.IGNORECASE)
_IMPERATIVE = re.compile(r"\b(prescrevo|prescreva|administre|administrar|tome|tomar|aplique|aplicar|inicie|iniciar|"
                         r"use|usar|fa[cç]a|dar alta|alta hospitalar)\b", re.IGNORECASE)
_SOURCE = re.compile(r"\[(PROT-\d{3}|PRONTUARIO|MEDQUAD)[^\]]*\]")


@dataclass
class InputCheck:
    sanitized: str
    flags: list[str] = field(default_factory=list)

    @property
    def is_prescription_request(self) -> bool:
        return "pedido_prescricao" in self.flags


@dataclass
class OutputCheck:
    text: str
    flags: list[str] = field(default_factory=list)
    requires_validation: bool = True
    blocked: bool = False


def check_input(question: str) -> InputCheck:
    flags = []
    if _PRESCRIPTION_REQUEST.search(question):
        flags.append("pedido_prescricao")
    if _INJECTION.search(question):
        flags.append("tentativa_burlar_regras")
    sanitized = question
    if contains_pii(question):
        sanitized = anonymize_text(question).text
        flags.append("pii_mascarada_na_entrada")
    return InputCheck(sanitized=sanitized, flags=flags)


def check_output(answer: str, sources: list[str]) -> OutputCheck:
    flags: list[str] = []
    text = answer.strip()
    blocked = False

    if contains_pii(text):
        text = anonymize_text(text).text
        flags.append("pii_mascarada_na_saida")

    has_dose = bool(_DOSE.search(text))
    if has_dose and _IMPERATIVE.search(text):
        flags.append("linguagem_prescritiva")
        blocked = True
        text = ("A resposta gerada continha uma indicacao de medicamento/dose e foi bloqueada pela politica "
                "de seguranca (PROT-008). Consulte o protocolo citado e defina a conduta conforme sua avaliacao.")
    elif has_dose:
        flags.append("menciona_dose")

    if not sources and not _SOURCE.search(text):
        flags.append("sem_fonte")

    return OutputCheck(text=f"{text}\n\n{DISCLAIMER}", flags=flags, requires_validation=True, blocked=blocked)
