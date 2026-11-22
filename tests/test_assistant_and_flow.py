import pytest

from assistant.patient_repository import PatientRepository
from assistant.retriever import ProtocolRetriever
from assistant.service import ClinicalAssistant
from safety.audit import read_trace
from workflows.patient_flow import PatientFlow


@pytest.fixture(scope="module")
def repo():
    return PatientRepository()


@pytest.fixture(scope="module")
def retriever():
    return ProtocolRetriever.from_dir()


def test_patient_context_has_no_pii(repo):
    ctx = repo.get_patient_context("P001")
    assert "nome" not in ctx and "cpf" not in ctx
    assert "CPF" not in ctx["ultima_evolucao"] or "[CPF]" in ctx["ultima_evolucao"]
    assert ctx["diagnostico"].startswith("Suspeita de sepse")


def test_pending_and_critical_exams(repo):
    crit = {e["tipo"] for e in repo.get_critical_results("P001")}
    assert "Lactato" in crit
    pend = repo.get_pending_exams("P005")
    assert pend and pend[0]["atrasado"]


def test_retriever_returns_relevant_protocol(retriever):
    docs = retriever.invoke("exames pacote primeira hora sepse lactato hemocultura")
    assert docs[0].metadata["source_id"] == "PROT-001"


def test_assistant_answers_with_sources_and_audit(repo, retriever):
    resp = ClinicalAssistant(retriever=retriever, repository=repo).ask(
        "Quais exames devo solicitar na sepse?", patient_id="P001")
    assert resp.fontes and resp.fontes[0]["fonte"] == "PROT-001"
    assert "[PROT-001" in resp.resposta and resp.usou_prontuario and resp.requer_validacao_medica
    events = [e["evento"] for e in read_trace(resp.trace_id)]
    assert events[0] == "pergunta_recebida" and events[-1] == "resposta_entregue"


def test_assistant_prescription_request_pulls_policy(repo, retriever):
    resp = ClinicalAssistant(retriever=retriever, repository=repo).ask("Prescreva o antibiotico para o P001")
    assert "pedido_prescricao" in resp.guardrails
    assert resp.fontes[0]["fonte"] == "PROT-008"


def test_flow_pauses_for_human_validation_and_resumes(repo, retriever):
    flow = PatientFlow(repository=repo, retriever=retriever)
    state = flow.start("P001", "dra_teste")
    assert state["aguardando_validacao"] and state["status"] == "AGUARDANDO_VALIDACAO_MEDICA"
    assert any(a["tipo"] == "RESULTADO_CRITICO" for a in state["alertas"])
    assert state["etapas"] == ["carregar_paciente", "verificar_exames", "buscar_protocolos", "sugerir_conduta",
                               "validar_seguranca", "emitir_alertas"]
    final = flow.decide(state["trace_id"], "dra_teste", aprovado=True, comentario="ok")
    assert final["status"] == "SUGESTAO_APROVADA" and final["etapas"][-1] == "registrar_validacao"
    assert any(e["evento"] == "fluxo_validacao_medica" for e in read_trace(state["trace_id"]))
    with pytest.raises(ValueError):
        flow.decide(state["trace_id"], "dra_teste", aprovado=True)


def test_flow_unknown_patient(repo, retriever):
    state = PatientFlow(repository=repo, retriever=retriever).start("X999")
    assert state["status"] == "PACIENTE_NAO_ENCONTRADO" and not state["aguardando_validacao"]
