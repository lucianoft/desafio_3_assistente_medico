"""API REST do assistente medico.

  uvicorn api.main:app --reload
  POST /assistente/perguntar            {"pergunta": "...", "paciente_id": "P001"}
  POST /fluxos/paciente/{paciente_id}   inicia o fluxo LangGraph (para em validacao humana)
  POST /fluxos/{trace_id}/validacao     {"medico": "...", "aprovado": true, "comentario": "..."}
  GET  /fluxos/{trace_id}               estado do fluxo
  GET  /auditoria/{trace_id}            eventos do log de auditoria
  GET  /pacientes/{paciente_id}/alertas
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from safety.audit import read_trace
from safety.guardrails import LIMITES_ATUACAO

app = FastAPI(title="Assistente Medico - Hospital Escola Fiap (Fase 3)", version="1.0.0")


class PerguntaIn(BaseModel):
    pergunta: str = Field(..., min_length=3)
    paciente_id: str | None = None
    medico: str = "medico_api"


class ValidacaoIn(BaseModel):
    medico: str
    aprovado: bool
    comentario: str = ""


@lru_cache(maxsize=1)
def assistant():
    from assistant.service import ClinicalAssistant

    return ClinicalAssistant()


@lru_cache(maxsize=1)
def flow():
    from workflows.patient_flow import PatientFlow

    return PatientFlow()


@app.get("/health")
def health():
    from assistant.llm import model_description

    return {"status": "ok", "modelo": model_description(), "limites_de_atuacao": LIMITES_ATUACAO}


@app.post("/assistente/perguntar")
def perguntar(body: PerguntaIn):
    return assistant().ask(body.pergunta, patient_id=body.paciente_id, user=body.medico).to_dict()


@app.post("/fluxos/paciente/{paciente_id}")
def iniciar_fluxo(paciente_id: str, solicitante: str = "medico_api"):
    state = flow().start(paciente_id, solicitante)
    if state.get("status") == "PACIENTE_NAO_ENCONTRADO":
        raise HTTPException(404, f"Paciente {paciente_id} nao encontrado (trace {state['trace_id']})")
    return state


@app.get("/fluxos/{trace_id}")
def estado_fluxo(trace_id: str):
    state = flow().snapshot(trace_id)
    if "patient_id" not in state:
        raise HTTPException(404, "Fluxo nao encontrado")
    return state


@app.post("/fluxos/{trace_id}/validacao")
def validar_fluxo(trace_id: str, body: ValidacaoIn):
    try:
        return flow().decide(trace_id, body.medico, body.aprovado, body.comentario)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/auditoria/{trace_id}")
def auditoria(trace_id: str):
    return read_trace(trace_id)


@app.get("/pacientes/{paciente_id}/alertas")
def alertas(paciente_id: str):
    return flow().repo.list_alerts(paciente_id)
