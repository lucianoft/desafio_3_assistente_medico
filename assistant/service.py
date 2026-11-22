"""Assistente medico: pipeline LangChain (RAG nos protocolos + prontuario + LLM ajustada + guardrails).

    pergunta -> guardrail de entrada -> [retriever protocolos || contexto do paciente (SQLite)]
             -> prompt -> LLM -> guardrail de saida -> resposta com fontes + log de auditoria
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

from assistant.llm import get_chat_model, model_description
from assistant.patient_repository import PatientNotFound, PatientRepository, format_patient_context
from assistant.prompts import QA_PROMPT
from assistant.retriever import ProtocolRetriever, doc_sources, format_docs
from safety.audit import log_event, new_trace_id
from safety.guardrails import check_input, check_output


@dataclass
class AssistantResponse:
    trace_id: str
    resposta: str
    fontes: list[dict] = field(default_factory=list)
    paciente: str | None = None
    usou_prontuario: bool = False
    guardrails: list[str] = field(default_factory=list)
    requer_validacao_medica: bool = True
    bloqueada: bool = False
    modelo: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ClinicalAssistant:
    def __init__(self, retriever: ProtocolRetriever | None = None, repository: PatientRepository | None = None,
                 llm=None):
        self.retriever = retriever or ProtocolRetriever.from_dir()
        self.repository = repository or PatientRepository()
        self.llm = llm or get_chat_model()
        self.model_name = model_description() if llm is None else type(llm).__name__
        self.answer_chain = QA_PROMPT | self.llm | StrOutputParser()

    def _context_chain(self):
        def patient(inp: dict) -> dict:
            pid = inp.get("patient_id")
            if not pid:
                return {"texto": "Nao informado.", "usado": False}
            ctx = self.repository.get_patient_context(pid)
            return {"texto": format_patient_context(ctx), "usado": True}

        return RunnableParallel(
            docs=RunnableLambda(lambda inp: self.retriever.search(inp["query"], prefer=inp.get("prefer"))),
            patient=RunnableLambda(patient),
        )

    def ask(self, question: str, patient_id: str | None = None, user: str = "medico") -> AssistantResponse:
        trace_id = new_trace_id()
        inp = check_input(question)
        log_event(trace_id, "pergunta_recebida", user, paciente=patient_id, pergunta=inp.sanitized,
                  guardrails_entrada=inp.flags)

        try:
            # paciente entra na busca para trazer o protocolo do diagnostico
            query, prefer = inp.sanitized, []
            if patient_id:
                ctx = self.repository.get_patient_context(patient_id)
                if ctx.get("protocolo_vinculado"):
                    prefer = [ctx["protocolo_vinculado"]]
                else:
                    query += f" {ctx.get('diagnostico', '')}"
            context = self._context_chain().invoke({"query": query, "patient_id": patient_id, "prefer": prefer})
        except PatientNotFound:
            log_event(trace_id, "paciente_nao_encontrado", user, paciente=patient_id)
            return AssistantResponse(trace_id, f"Paciente {patient_id} nao encontrado.", paciente=patient_id,
                                     guardrails=["paciente_nao_encontrado"], modelo=self.model_name)

        docs = context["docs"]
        if inp.is_prescription_request or "tentativa_burlar_regras" in inp.flags:
            # garante que a politica de uso esteja no contexto do modelo
            policy = self.retriever.invoke("politica limites de atuacao assistente prescreve doses")[:1]
            docs = policy + [d for d in docs if d.metadata["source_id"] != "PROT-008"]
        sources = doc_sources(docs)
        log_event(trace_id, "contexto_recuperado", user, fontes=[f"{s['fonte']}/{s['secao']}" for s in sources],
                  usou_prontuario=context["patient"]["usado"])

        raw = self.answer_chain.invoke({"protocolos": format_docs(docs) or "Nenhum protocolo encontrado.",
                                        "paciente": context["patient"]["texto"], "pergunta": inp.sanitized})
        log_event(trace_id, "resposta_modelo", user, modelo=self.model_name, resposta_bruta=raw)

        out = check_output(raw, [s["fonte"] for s in sources])
        flags = inp.flags + out.flags
        resp = AssistantResponse(trace_id=trace_id, resposta=out.text, fontes=sources, paciente=patient_id,
                                 usou_prontuario=context["patient"]["usado"], guardrails=flags,
                                 requer_validacao_medica=out.requires_validation, bloqueada=out.blocked,
                                 modelo=self.model_name)
        log_event(trace_id, "resposta_entregue", user, guardrails=flags, bloqueada=out.blocked,
                  requer_validacao=out.requires_validation)
        return resp
