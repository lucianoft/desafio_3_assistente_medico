"""Ferramentas LangChain (@tool) sobre a base estruturada e os protocolos.

Usadas pelos nos do LangGraph e disponiveis para um agente com tool-calling (backend openai),
caso se queira deixar a propria LLM decidir quais consultas fazer.
"""
from __future__ import annotations

import json
from functools import lru_cache

from langchain_core.tools import tool

from assistant.patient_repository import PatientRepository, format_patient_context
from assistant.retriever import ProtocolRetriever, format_docs


@lru_cache(maxsize=1)
def _repo() -> PatientRepository:
    return PatientRepository()


@lru_cache(maxsize=1)
def _retriever() -> ProtocolRetriever:
    return ProtocolRetriever.from_dir()


@tool
def consultar_prontuario(paciente_id: str) -> str:
    """Retorna o contexto clinico pseudonimizado do paciente (diagnostico, alergias, exames)."""
    return format_patient_context(_repo().get_patient_context(paciente_id))


@tool
def listar_exames_pendentes(paciente_id: str) -> str:
    """Lista os exames pendentes do paciente, indicando se estao atrasados (PROT-007)."""
    return json.dumps(_repo().get_pending_exams(paciente_id), ensure_ascii=False)


@tool
def listar_resultados_criticos(paciente_id: str) -> str:
    """Lista resultados de exames com valor critico (PROT-007)."""
    return json.dumps(_repo().get_critical_results(paciente_id), ensure_ascii=False)


@tool
def buscar_protocolo(consulta: str) -> str:
    """Busca trechos dos protocolos internos do hospital, com identificacao da fonte."""
    return format_docs(_retriever().invoke(consulta))


TOOLS = [consultar_prontuario, listar_exames_pendentes, listar_resultados_criticos, buscar_protocolo]
