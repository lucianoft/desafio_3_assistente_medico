"""Gera os dados sinteticos do hospital.

Saidas (em data/synthetic/):
  * protocolos/PROT-xxx.md   -> base de conhecimento do assistente (RAG)
  * faq_medicos.jsonl, modelos_documentos.jsonl
  * evolucoes_brutas.jsonl    -> textos clinicos COM dados pessoais (para testar a anonimizacao)
  * hospital.db               -> SQLite com pacientes, internacoes, exames e evolucoes

Uso:  python -m data_pipeline.generate_synthetic
"""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from core.config import settings
from data_pipeline.synthetic_content import DOCUMENT_TEMPLATES, FAQ_MEDICOS, PROTOCOLS

SEED = 42
REFERENCE_NOW = datetime(2026, 11, 20, 8, 0)  # "agora" fixo -> dados reprodutiveis

FIRST_NAMES = {"F": ["Ana", "Carla", "Elisa", "Gabriela", "Isabela", "Larissa", "Natalia", "Paula", "Sofia", "Vanessa"],
               "M": ["Bruno", "Diego", "Fabio", "Henrique", "Joao", "Marcos", "Otavio", "Rafael", "Thiago", "Wagner"]}
LAST_NAMES = ["Silva", "Souza", "Oliveira", "Pereira", "Costa", "Rodrigues", "Almeida", "Nascimento", "Lima",
              "Araujo", "Ferreira", "Carvalho", "Gomes", "Martins", "Rocha"]
SETORES = ["Pronto Atendimento", "Enfermaria Clinica", "UTI Adulto", "Ambulatorio de Mastologia"]

# Catalogo de exames: (tipo, unidade, ref_min, ref_max, crit_min, crit_max)
EXAMS = {
    "Lactato": ("mmol/L", 0.5, 2.0, None, 4.0),
    "Hemocultura": (None, None, None, None, None),
    "Hemograma - Hemoglobina": ("g/dL", 12.0, 16.0, 7.0, None),
    "Plaquetas": ("/mm3", 150000, 450000, 50000, None),
    "Creatinina": ("mg/dL", 0.6, 1.2, None, None),
    "Troponina ultrassensivel": ("ng/L", 0, 14, None, 14),
    "Glicemia": ("mg/dL", 70, 99, 50, 400),
    "Potassio": ("mEq/L", 3.5, 5.0, 3.0, 6.0),
    "Gasometria venosa - pH": (None, 7.32, 7.42, None, None),
    "PCR": ("mg/L", 0, 5, None, None),
    "Radiografia de torax": (None, None, None, None, None),
    "Eletrocardiograma": (None, None, None, None, None),
    "Biopsia mamaria": (None, None, None, None, None),
    "Ultrassonografia mamaria": (None, None, None, None, None),
}

# Cenarios fixos para a demonstracao (video) -> sempre os mesmos pacientes.
DEMO_SCENARIOS = [
    {
        "id": "P001", "idade": 67, "sexo": "M", "setor": "Pronto Atendimento", "diagnostico": "Suspeita de sepse de foco urinario",
        "cid": "A41.9", "alergias": "Dipirona", "protocolo": "PROT-001",
        "exames": [("Lactato", "LIBERADO", 4.6, None, 3), ("Hemocultura", "PENDENTE", None, None, 3),
                   ("Creatinina", "LIBERADO", 1.9, None, 3), ("Hemograma - Hemoglobina", "LIBERADO", 11.2, None, 3)],
        "evolucao": "admitido com febre de 39 C, confusao mental e disuria ha 3 dias. PA 92x58 mmHg, FR 26 irpm.",
    },
    {
        "id": "P002", "idade": 58, "sexo": "F", "setor": "Pronto Atendimento", "diagnostico": "Dor toracica a esclarecer",
        "cid": "R07.4", "alergias": "Nenhuma conhecida", "protocolo": "PROT-002",
        "exames": [("Eletrocardiograma", "LIBERADO", None, "Sem supradesnivelamento de ST", 2),
                   ("Troponina ultrassensivel", "LIBERADO", 38.0, None, 2),
                   ("Troponina ultrassensivel", "PENDENTE", None, None, 0)],
        "evolucao": "refere dor retroesternal em aperto ha 2 horas, irradiada para membro superior esquerdo. Hipertensa e tabagista.",
    },
    {
        "id": "P003", "idade": 24, "sexo": "F", "setor": "Enfermaria Clinica", "diagnostico": "Diabetes tipo 1 descompensado",
        "cid": "E10.1", "alergias": "Nenhuma conhecida", "protocolo": "PROT-003",
        "exames": [("Glicemia", "LIBERADO", 486.0, None, 5), ("Potassio", "LIBERADO", 3.1, None, 5),
                   ("Gasometria venosa - pH", "LIBERADO", 7.18, None, 5)],
        "evolucao": "nauseas, vomitos e dor abdominal, suspendeu insulina ha 2 dias. Halito cetonico.",
    },
    {
        "id": "P004", "idade": 72, "sexo": "M", "setor": "Enfermaria Clinica", "diagnostico": "Pneumonia adquirida na comunidade",
        "cid": "J18.9", "alergias": "Penicilina", "protocolo": "PROT-004",
        "exames": [("Radiografia de torax", "LIBERADO", None, "Consolidacao em base direita", 30),
                   ("PCR", "LIBERADO", 148.0, None, 30), ("Hemocultura", "PENDENTE", None, None, 60)],
        "evolucao": "tosse produtiva e febre ha 5 dias, saturacao 91% em ar ambiente. CURB-65 = 2.",
    },
    {
        "id": "P005", "idade": 49, "sexo": "F", "setor": "Ambulatorio de Mastologia", "diagnostico": "Nodulo mamario BI-RADS 4",
        "cid": "N63", "alergias": "Nenhuma conhecida", "protocolo": "PROT-005",
        "exames": [("Ultrassonografia mamaria", "LIBERADO", None, "Nodulo solido irregular 1,8 cm, BI-RADS 4", 24 * 20),
                   ("Biopsia mamaria", "PENDENTE", None, None, 24 * 18)],
        "evolucao": "nodulo palpavel em mama esquerda, quadrante superior externo. Modelo preditivo indicou 0,81 de probabilidade de malignidade.",
    },
]

RANDOM_DIAGNOSES = [
    ("Pneumonia adquirida na comunidade", "J18.9", ["Radiografia de torax", "PCR", "Hemograma - Hemoglobina"]),
    ("Crise hipertensiva", "I10", ["Eletrocardiograma", "Creatinina"]),
    ("Infeccao do trato urinario", "N39.0", ["Hemograma - Hemoglobina", "Creatinina", "PCR"]),
    ("Diabetes tipo 2 descompensado", "E11.6", ["Glicemia", "Potassio", "Creatinina"]),
    ("Insuficiencia cardiaca descompensada", "I50.9", ["Creatinina", "Potassio", "Eletrocardiograma"]),
]


def _cpf(rng: random.Random) -> str:
    d = [rng.randint(0, 9) for _ in range(11)]
    return f"{d[0]}{d[1]}{d[2]}.{d[3]}{d[4]}{d[5]}.{d[6]}{d[7]}{d[8]}-{d[9]}{d[10]}"


def _phone(rng: random.Random) -> str:
    return f"(11) 9{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"


def _random_value(rng: random.Random, tipo: str) -> tuple[float | None, str | None]:
    unidade, ref_min, ref_max, *_ = EXAMS[tipo]
    if ref_min is None:
        return None, "Sem alteracoes significativas"
    span = (ref_max - ref_min) or 1
    value = rng.uniform(ref_min - 0.2 * span, ref_max + 0.4 * span)
    return round(value, 2), None


def write_protocols(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for prot in PROTOCOLS:
        lines = [f"# {prot['id']} - {prot['titulo']}", "", f"Versao: {prot['versao']}  |  Documento interno ficticio", ""]
        for secao, texto in prot["secoes"].items():
            lines += [f"## {secao}", "", texto, ""]
        (out_dir / f"{prot['id']}.md").write_text("\n".join(lines), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


SCHEMA = """
CREATE TABLE pacientes (id TEXT PRIMARY KEY, nome TEXT, cpf TEXT, telefone TEXT, data_nascimento TEXT, sexo TEXT);
CREATE TABLE internacoes (id INTEGER PRIMARY KEY AUTOINCREMENT, paciente_id TEXT, data_admissao TEXT, setor TEXT,
    diagnostico TEXT, cid TEXT, alergias TEXT, protocolo TEXT, status TEXT);
CREATE TABLE exames (id INTEGER PRIMARY KEY AUTOINCREMENT, paciente_id TEXT, tipo TEXT, data_solicitacao TEXT,
    data_resultado TEXT, status TEXT, valor REAL, resultado_texto TEXT, unidade TEXT,
    ref_min REAL, ref_max REAL, crit_min REAL, crit_max REAL, urgente INTEGER);
CREATE TABLE evolucoes (id INTEGER PRIMARY KEY AUTOINCREMENT, paciente_id TEXT, data TEXT, profissional TEXT, texto TEXT);
CREATE TABLE alertas (id INTEGER PRIMARY KEY AUTOINCREMENT, paciente_id TEXT, trace_id TEXT, tipo TEXT,
    severidade TEXT, mensagem TEXT, criado_em TEXT, status TEXT);
CREATE TABLE validacoes (id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, paciente_id TEXT, medico TEXT,
    decisao TEXT, comentario TEXT, criado_em TEXT);
"""


def build_database(db_path: Path, n_random: int = 25, seed: int = SEED) -> list[dict]:
    """Cria o SQLite do hospital e devolve as evolucoes brutas (com PII)."""
    rng = random.Random(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    scenarios = list(DEMO_SCENARIOS)
    for i in range(n_random):
        diag, cid, exams = rng.choice(RANDOM_DIAGNOSES)
        scenarios.append({
            "id": f"P{100 + i:03d}", "idade": rng.randint(18, 90), "sexo": rng.choice("MF"),
            "setor": rng.choice(SETORES[:3]), "diagnostico": diag, "cid": cid,
            "alergias": rng.choice(["Nenhuma conhecida", "Dipirona", "Penicilina", "Sulfa"]),
            "protocolo": None,
            "exames": [(t, rng.choice(["LIBERADO", "LIBERADO", "PENDENTE"]), None, None, rng.randint(1, 96)) for t in exams],
            "evolucao": "evolui estavel, sem queixas novas nas ultimas 24 horas.",
        })

    raw_notes = []
    for sc in scenarios:
        nome = f"{rng.choice(FIRST_NAMES[sc['sexo']])} {rng.choice(LAST_NAMES)} {rng.choice(LAST_NAMES)}"
        cpf, tel = _cpf(rng), _phone(rng)
        nasc = (REFERENCE_NOW - timedelta(days=365 * sc["idade"] + rng.randint(0, 364))).date().isoformat()
        con.execute("INSERT INTO pacientes VALUES (?,?,?,?,?,?)", (sc["id"], nome, cpf, tel, nasc, sc["sexo"]))
        admissao = REFERENCE_NOW - timedelta(hours=max(h for *_, h in sc["exames"]) + 2)
        con.execute(
            "INSERT INTO internacoes (paciente_id, data_admissao, setor, diagnostico, cid, alergias, protocolo, status)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (sc["id"], admissao.isoformat(timespec="minutes"), sc["setor"], sc["diagnostico"], sc["cid"], sc["alergias"],
             sc["protocolo"], "ATIVA"),
        )
        for tipo, status, valor, texto, horas_atras in sc["exames"]:
            unidade, ref_min, ref_max, crit_min, crit_max = EXAMS[tipo]
            if status == "LIBERADO" and valor is None and texto is None:
                valor, texto = _random_value(rng, tipo)
            solicitado = REFERENCE_NOW - timedelta(hours=horas_atras)
            resultado = (solicitado + timedelta(hours=1)).isoformat(timespec="minutes") if status == "LIBERADO" else None
            urgente = 1 if sc["setor"] in ("Pronto Atendimento", "UTI Adulto") else 0
            con.execute(
                "INSERT INTO exames (paciente_id, tipo, data_solicitacao, data_resultado, status, valor, resultado_texto,"
                " unidade, ref_min, ref_max, crit_min, crit_max, urgente) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sc["id"], tipo, solicitado.isoformat(timespec="minutes"), resultado, status, valor, texto, unidade,
                 ref_min, ref_max, crit_min, crit_max, urgente),
            )
        # Texto livre "como o medico escreve" -> contem PII de proposito
        texto = (f"Paciente {nome}, {sc['idade']} anos, CPF {cpf}, telefone {tel}, internado em "
                 f"{admissao.strftime('%d/%m/%Y')} no setor {sc['setor']}. {sc['evolucao'][0].upper()}{sc['evolucao'][1:]} "
                 f"Hipotese: {sc['diagnostico']} (CID {sc['cid']}). Alergias: {sc['alergias']}.")
        con.execute("INSERT INTO evolucoes (paciente_id, data, profissional, texto) VALUES (?,?,?,?)",
                    (sc["id"], admissao.isoformat(timespec="minutes"), "Dr(a). Plantonista", texto))
        raw_notes.append({"paciente_id": sc["id"], "texto": texto, "diagnostico": sc["diagnostico"], "cid": sc["cid"],
                          "idade": sc["idade"], "sexo": sc["sexo"], "setor": sc["setor"], "alergias": sc["alergias"],
                          "evolucao": sc["evolucao"], "nome": nome, "cpf": cpf, "telefone": tel})
    con.commit()
    con.close()
    return raw_notes


def main() -> None:
    out = settings.synthetic_dir
    out.mkdir(parents=True, exist_ok=True)
    write_protocols(settings.protocols_dir)
    _write_jsonl(out / "faq_medicos.jsonl", FAQ_MEDICOS)
    _write_jsonl(out / "modelos_documentos.jsonl", DOCUMENT_TEMPLATES)
    notes = build_database(settings.db_path)
    _write_jsonl(out / "evolucoes_brutas.jsonl", notes)
    print(f"Protocolos: {len(PROTOCOLS)} | FAQs: {len(FAQ_MEDICOS)} | Modelos: {len(DOCUMENT_TEMPLATES)} | "
          f"Pacientes: {len(notes)} -> {settings.db_path}")


if __name__ == "__main__":
    main()
