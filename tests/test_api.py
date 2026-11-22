from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["limites_de_atuacao"]


def test_ask_and_audit():
    r = client.post("/assistente/perguntar", json={"pergunta": "Quais valores sao criticos?"})
    body = r.json()
    assert r.status_code == 200 and body["fontes"][0]["fonte"] == "PROT-007"
    assert client.get(f"/auditoria/{body['trace_id']}").json()


def test_flow_endpoints():
    r = client.post("/fluxos/paciente/P003")
    state = r.json()
    assert r.status_code == 200 and state["aguardando_validacao"]
    r = client.post(f"/fluxos/{state['trace_id']}/validacao", json={"medico": "dr_api", "aprovado": False,
                                                                    "comentario": "rever"})
    assert r.json()["status"] == "SUGESTAO_REJEITADA"
    assert client.post(f"/fluxos/{state['trace_id']}/validacao", json={"medico": "x", "aprovado": True}).status_code == 409
    assert client.post("/fluxos/paciente/X999").status_code == 404
    assert client.get("/pacientes/P003/alertas").json()
