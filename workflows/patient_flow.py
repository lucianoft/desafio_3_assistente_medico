"""Fluxo automatizado de triagem do paciente com LangGraph.

    START -> carregar_paciente -> verificar_exames -> buscar_protocolos -> sugerir_conduta
          -> validar_seguranca -> (ha alertas?) -> emitir_alertas -> [PAUSA: validacao humana]
                                                -> registrar_validacao -> END

O grafo e compilado com checkpointer e `interrupt_before=["registrar_validacao"]`: a execucao
para e so continua quando um medico aprova/rejeita a sugestao (human-in-the-loop).
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from assistant.llm import get_chat_model, model_description
from assistant.patient_repository import PatientNotFound, PatientRepository, format_patient_context
from assistant.prompts import FLOW_PROMPT
from assistant.retriever import ProtocolRetriever, doc_sources, format_docs
from safety.audit import log_event, new_trace_id
from safety.guardrails import check_output


class FlowState(TypedDict, total=False):
    trace_id: str
    patient_id: str
    solicitante: str
    paciente: dict
    exames_pendentes: list[dict]
    resultados_criticos: list[dict]
    fontes: list[dict]
    protocolos_texto: str
    sugestao: str
    guardrails: list[str]
    alertas: list[dict]
    decisao_medica: dict
    status: str
    etapas: Annotated[list[str], operator.add]


def _fmt_exams(exams: list[dict], pending: bool) -> str:
    if not exams:
        return "Nenhum."
    if pending:
        return "; ".join(f"{e['tipo']} ha {e['horas_desde_solicitacao']} h" + (" (ATRASADO)" if e.get("atrasado") else "")
                         for e in exams)
    return "; ".join(f"{e['tipo']} = {e['valor']} {e['unidade'] or ''}".strip() for e in exams)


class PatientFlow:
    def __init__(self, repository: PatientRepository | None = None, retriever: ProtocolRetriever | None = None,
                 llm=None, checkpointer=None):
        self.repo = repository or PatientRepository()
        self.retriever = retriever or ProtocolRetriever.from_dir()
        self.llm = llm or get_chat_model()
        self.model_name = model_description() if llm is None else type(llm).__name__
        self.graph = self._build().compile(checkpointer=checkpointer or MemorySaver(),
                                           interrupt_before=["registrar_validacao"])

    # ------------------------------------------------------------------ nos
    def carregar_paciente(self, s: FlowState) -> FlowState:
        try:
            ctx = self.repo.get_patient_context(s["patient_id"])
        except PatientNotFound:
            log_event(s["trace_id"], "fluxo_paciente_nao_encontrado", s["solicitante"], paciente=s["patient_id"])
            return {"status": "PACIENTE_NAO_ENCONTRADO", "etapas": ["carregar_paciente"]}
        log_event(s["trace_id"], "fluxo_paciente_carregado", s["solicitante"], paciente=s["patient_id"],
                  diagnostico=ctx.get("diagnostico"))
        return {"paciente": ctx, "status": "EM_ANALISE", "etapas": ["carregar_paciente"]}

    def verificar_exames(self, s: FlowState) -> FlowState:
        pend = self.repo.get_pending_exams(s["patient_id"])
        crit = self.repo.get_critical_results(s["patient_id"])
        log_event(s["trace_id"], "fluxo_exames_verificados", s["solicitante"],
                  pendentes=[e["tipo"] for e in pend], criticos=[e["tipo"] for e in crit])
        return {"exames_pendentes": pend, "resultados_criticos": crit, "etapas": ["verificar_exames"]}

    def buscar_protocolos(self, s: FlowState) -> FlowState:
        p = s["paciente"]
        query = " ".join([p.get("diagnostico", ""), *(e["tipo"] for e in s["resultados_criticos"]),
                          "conduta exames criterios de alerta"])
        prefer = [p["protocolo_vinculado"]] if p.get("protocolo_vinculado") else []
        docs = self.retriever.search(query, prefer=prefer)
        if s["resultados_criticos"] or s["exames_pendentes"]:
            topic = "valores criticos comunicacao" if s["resultados_criticos"] else "exames pendentes atrasados"
            docs += [d for d in self.retriever.search(topic, prefer=["PROT-007"])
                     if d.metadata["source_id"] == "PROT-007"][:1]
        fontes = doc_sources(docs)
        log_event(s["trace_id"], "fluxo_protocolos_recuperados", s["solicitante"],
                  fontes=[f"{f['fonte']}/{f['secao']}" for f in fontes])
        return {"fontes": fontes, "protocolos_texto": format_docs(docs), "etapas": ["buscar_protocolos"]}

    def sugerir_conduta(self, s: FlowState) -> FlowState:
        chain = FLOW_PROMPT | self.llm | StrOutputParser()
        sugestao = chain.invoke({
            "protocolos": s["protocolos_texto"], "paciente": format_patient_context(s["paciente"]),
            "pendentes": _fmt_exams(s["exames_pendentes"], True), "criticos": _fmt_exams(s["resultados_criticos"], False),
        })
        log_event(s["trace_id"], "fluxo_sugestao_gerada", s["solicitante"], modelo=self.model_name, sugestao=sugestao)
        return {"sugestao": sugestao, "etapas": ["sugerir_conduta"]}

    def validar_seguranca(self, s: FlowState) -> FlowState:
        out = check_output(s["sugestao"], [f["fonte"] for f in s["fontes"]])
        log_event(s["trace_id"], "fluxo_guardrails", s["solicitante"], flags=out.flags, bloqueada=out.blocked)
        return {"sugestao": out.text, "guardrails": out.flags, "etapas": ["validar_seguranca"]}

    def emitir_alertas(self, s: FlowState) -> FlowState:
        alerts = []
        for e in s["resultados_criticos"]:
            alerts.append({"tipo": "RESULTADO_CRITICO", "severidade": "ALTA",
                           "mensagem": f"{e['tipo']} = {e['valor']} {e['unidade'] or ''} - comunicar em ate 30 min (PROT-007)"})
        for e in s["exames_pendentes"]:
            if e.get("atrasado"):
                alerts.append({"tipo": "EXAME_ATRASADO", "severidade": "MEDIA",
                               "mensagem": f"{e['tipo']} pendente ha {e['horas_desde_solicitacao']} h "
                                           f"(limite {e['limite_horas']} h, PROT-007)"})
        if "linguagem_prescritiva" in s.get("guardrails", []):
            alerts.append({"tipo": "SUGESTAO_BLOQUEADA", "severidade": "MEDIA",
                           "mensagem": "Sugestao do modelo bloqueada por linguagem prescritiva (PROT-008)"})
        for a in alerts:
            a["id"] = self.repo.record_alert(s["patient_id"], s["trace_id"], a["tipo"], a["severidade"], a["mensagem"])
        log_event(s["trace_id"], "fluxo_alertas_emitidos", s["solicitante"], alertas=alerts)
        return {"alertas": alerts, "etapas": ["emitir_alertas"]}

    def registrar_validacao(self, s: FlowState) -> FlowState:
        d = s.get("decisao_medica") or {}
        aprovado = bool(d.get("aprovado"))
        decisao = "APROVADA" if aprovado else "REJEITADA"
        self.repo.record_validation(s["trace_id"], s["patient_id"], d.get("medico", "desconhecido"), decisao,
                                    d.get("comentario", ""))
        log_event(s["trace_id"], "fluxo_validacao_medica", d.get("medico", "desconhecido"), decisao=decisao,
                  comentario=d.get("comentario", ""))
        return {"status": f"SUGESTAO_{decisao}", "etapas": ["registrar_validacao"]}

    # ------------------------------------------------------------------ grafo
    def _build(self) -> StateGraph:
        g = StateGraph(FlowState)
        for name in ("carregar_paciente", "verificar_exames", "buscar_protocolos", "sugerir_conduta",
                     "validar_seguranca", "emitir_alertas", "registrar_validacao"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "carregar_paciente")
        g.add_conditional_edges("carregar_paciente",
                                lambda s: "fim" if s.get("status") == "PACIENTE_NAO_ENCONTRADO" else "seguir",
                                {"fim": END, "seguir": "verificar_exames"})
        g.add_edge("verificar_exames", "buscar_protocolos")
        g.add_edge("buscar_protocolos", "sugerir_conduta")
        g.add_edge("sugerir_conduta", "validar_seguranca")
        g.add_conditional_edges(
            "validar_seguranca",
            lambda s: "alertar" if (s["resultados_criticos"] or any(e.get("atrasado") for e in s["exames_pendentes"])
                                    or "linguagem_prescritiva" in s.get("guardrails", [])) else "validar",
            {"alertar": "emitir_alertas", "validar": "registrar_validacao"},
        )
        g.add_edge("emitir_alertas", "registrar_validacao")
        g.add_edge("registrar_validacao", END)
        return g

    # ------------------------------------------------------------------ API publica
    def start(self, patient_id: str, solicitante: str = "sistema") -> dict:
        trace_id = new_trace_id()
        config = {"configurable": {"thread_id": trace_id}}
        log_event(trace_id, "fluxo_iniciado", solicitante, paciente=patient_id)
        self.graph.invoke({"trace_id": trace_id, "patient_id": patient_id, "solicitante": solicitante, "etapas": []},
                          config)
        return self.snapshot(trace_id)

    def decide(self, trace_id: str, medico: str, aprovado: bool, comentario: str = "") -> dict:
        config = {"configurable": {"thread_id": trace_id}}
        snap = self.graph.get_state(config)
        if not snap.next or "registrar_validacao" not in snap.next:
            raise ValueError(f"Fluxo {trace_id} nao esta aguardando validacao medica")
        self.graph.update_state(config, {"decisao_medica": {"medico": medico, "aprovado": aprovado,
                                                            "comentario": comentario}})
        self.graph.invoke(None, config)
        return self.snapshot(trace_id)

    def snapshot(self, trace_id: str) -> dict:
        snap = self.graph.get_state({"configurable": {"thread_id": trace_id}})
        values = dict(snap.values)
        waiting = bool(snap.next) and "registrar_validacao" in snap.next
        if waiting:
            values["status"] = "AGUARDANDO_VALIDACAO_MEDICA"
        values.pop("protocolos_texto", None)
        return {"trace_id": trace_id, "aguardando_validacao": waiting, "proximo_no": list(snap.next), **values}

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
