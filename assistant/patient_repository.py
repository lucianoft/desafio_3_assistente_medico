"""Acesso a base estruturada do hospital (prontuario, exames, alertas).

Principio de minimizacao: o que vai para a LLM e apenas o contexto PSEUDONIMIZADO
(codigo, idade, sexo, setor, diagnostico, alergias, exames). Nome/CPF/telefone ficam no banco.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from core.config import settings
from data_pipeline.generate_synthetic import REFERENCE_NOW

PENDING_URGENT_HOURS = 6     # PROT-007
PENDING_ROUTINE_HOURS = 48   # PROT-007


class PatientNotFound(LookupError):
    pass


class PatientRepository:
    def __init__(self, db_path: Path | None = None, now: datetime | None = None):
        self.db_path = Path(db_path or settings.db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Base {self.db_path} nao encontrada. Rode: python app.py setup")
        self.now = now or REFERENCE_NOW  # dados sinteticos usam um "agora" fixo

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _hours_since(self, iso: str) -> float:
        return round((self.now - datetime.fromisoformat(iso)).total_seconds() / 3600, 1)

    # ------------------------------------------------------------------ leitura
    def list_patient_ids(self) -> list[str]:
        with closing(self._conn()) as con:
            return [r["id"] for r in con.execute("SELECT id FROM pacientes ORDER BY id")]

    def get_patient_context(self, patient_id: str) -> dict:
        with closing(self._conn()) as con:
            p = con.execute("SELECT id, nome, data_nascimento, sexo FROM pacientes WHERE id = ?", (patient_id,)).fetchone()
            if p is None:
                raise PatientNotFound(patient_id)
            i = con.execute("SELECT * FROM internacoes WHERE paciente_id = ? AND status = 'ATIVA' "
                            "ORDER BY data_admissao DESC LIMIT 1", (patient_id,)).fetchone()
            ev = con.execute("SELECT texto FROM evolucoes WHERE paciente_id = ? ORDER BY data DESC LIMIT 1",
                             (patient_id,)).fetchone()
        idade = int((self.now.date() - datetime.fromisoformat(p["data_nascimento"]).date()).days // 365.25)
        ctx = {"codigo": p["id"], "idade": idade, "sexo": p["sexo"], "exames": self.get_exams(patient_id)}
        if i:
            ctx.update({"setor": i["setor"], "diagnostico": i["diagnostico"], "cid": i["cid"], "alergias": i["alergias"],
                        "protocolo_vinculado": i["protocolo"],
                        "horas_internado": self._hours_since(i["data_admissao"])})
        if ev:
            from data_pipeline.anonymize import anonymize_text

            # nome usado so para mascarar; nunca sai deste metodo
            ctx["ultima_evolucao"] = anonymize_text(ev["texto"], known_names=[p["nome"]]).text
        return ctx

    def get_exams(self, patient_id: str) -> list[dict]:
        with closing(self._conn()) as con:
            rows = con.execute("SELECT * FROM exames WHERE paciente_id = ? ORDER BY data_solicitacao",
                               (patient_id,)).fetchall()
        exams = []
        for r in rows:
            e = {"tipo": r["tipo"], "status": r["status"], "valor": r["valor"], "unidade": r["unidade"],
                 "resultado": r["resultado_texto"], "urgente": bool(r["urgente"]),
                 "horas_desde_solicitacao": self._hours_since(r["data_solicitacao"]),
                 "referencia": [r["ref_min"], r["ref_max"]]}
            e["critico"] = self._is_critical(r)
            e["alterado"] = r["valor"] is not None and r["ref_min"] is not None and not (r["ref_min"] <= r["valor"] <= r["ref_max"])
            exams.append(e)
        return exams

    @staticmethod
    def _is_critical(r) -> bool:
        v = r["valor"]
        if v is None:
            return False
        if r["tipo"].startswith("Troponina"):
            return r["crit_max"] is not None and v > r["crit_max"]
        return (r["crit_min"] is not None and v < r["crit_min"]) or (r["crit_max"] is not None and v >= r["crit_max"])

    def get_pending_exams(self, patient_id: str) -> list[dict]:
        out = []
        for e in self.get_exams(patient_id):
            if e["status"] != "PENDENTE":
                continue
            limit = PENDING_URGENT_HOURS if e["urgente"] else PENDING_ROUTINE_HOURS
            e["atrasado"] = e["horas_desde_solicitacao"] > limit
            e["limite_horas"] = limit
            out.append(e)
        return out

    def get_critical_results(self, patient_id: str) -> list[dict]:
        return [e for e in self.get_exams(patient_id) if e["critico"]]

    # ------------------------------------------------------------------ escrita
    def record_alert(self, patient_id: str, trace_id: str, tipo: str, severidade: str, mensagem: str) -> int:
        with closing(self._conn()) as con, con:
            cur = con.execute("INSERT INTO alertas (paciente_id, trace_id, tipo, severidade, mensagem, criado_em, status)"
                              " VALUES (?,?,?,?,?,?,?)",
                              (patient_id, trace_id, tipo, severidade, mensagem, datetime.now().isoformat(timespec="seconds"),
                               "ABERTO"))
            return cur.lastrowid

    def record_validation(self, trace_id: str, patient_id: str, medico: str, decisao: str, comentario: str = "") -> int:
        with closing(self._conn()) as con, con:
            cur = con.execute("INSERT INTO validacoes (trace_id, paciente_id, medico, decisao, comentario, criado_em)"
                              " VALUES (?,?,?,?,?,?)",
                              (trace_id, patient_id, medico, decisao, comentario, datetime.now().isoformat(timespec="seconds")))
            con.execute("UPDATE alertas SET status = ? WHERE trace_id = ?", ("RECONHECIDO", trace_id))
            return cur.lastrowid

    def list_alerts(self, patient_id: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM alertas", ()
        if patient_id:
            sql, args = sql + " WHERE paciente_id = ?", (patient_id,)
        with closing(self._conn()) as con:
            return [dict(r) for r in con.execute(sql + " ORDER BY id", args)]


def format_patient_context(ctx: dict) -> str:
    """Texto compacto para o prompt (sem PII)."""
    lines = [f"Paciente {ctx['codigo']} | {ctx['idade']} anos | sexo {ctx['sexo']} | setor {ctx.get('setor', '-')}",
             f"Diagnostico: {ctx.get('diagnostico', '-')} (CID {ctx.get('cid', '-')}) | Alergias: {ctx.get('alergias', '-')}"]
    for e in ctx.get("exames", []):
        if e["status"] == "PENDENTE":
            lines.append(f"- {e['tipo']}: PENDENTE ha {e['horas_desde_solicitacao']} h")
        else:
            val = f"{e['valor']} {e['unidade'] or ''}".strip() if e["valor"] is not None else e["resultado"]
            tag = " (CRITICO)" if e["critico"] else " (alterado)" if e["alterado"] else ""
            lines.append(f"- {e['tipo']}: {val}{tag}")
    if ctx.get("ultima_evolucao"):
        lines.append(f"Ultima evolucao: {ctx['ultima_evolucao']}")
    return "\n".join(lines)
