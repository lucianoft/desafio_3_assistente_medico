"""CLI do projeto.

  python app.py setup                                   # gera dados sinteticos + SQLite + dataset (sem MedQuAD)
  python app.py perguntar "Quais exames da sepse?" --paciente P001
  python app.py fluxo P001                              # roda o LangGraph e pede validacao do medico
  python app.py fluxo P001 --decisao aprovar --medico "Dra. Ana"
  python app.py auditoria <trace_id>
  python app.py diagrama                                # exporta docs/fluxo_langgraph.mmd
"""
from __future__ import annotations

import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def cmd_setup(args) -> None:
    from data_pipeline import build_dataset, generate_synthetic

    generate_synthetic.main()
    build_dataset.main([] if args.medquad else ["--skip-medquad"])


def cmd_perguntar(args) -> None:
    from assistant.service import ClinicalAssistant

    resp = ClinicalAssistant().ask(args.pergunta, patient_id=args.paciente, user=args.medico)
    print(f"\n[trace_id {resp.trace_id}] modelo: {resp.modelo}\n")
    print(resp.resposta)
    print("\nFontes:")
    for f in resp.fontes:
        print(f"  - {f['fonte']} / {f['secao']} (v{f['versao']}, score {f['score']})")
    if resp.usou_prontuario:
        print(f"  - PRONTUARIO do paciente {resp.paciente} (dados pseudonimizados)")
    print(f"\nGuardrails: {resp.guardrails or 'nenhum'} | requer validacao medica: {resp.requer_validacao_medica}")


def cmd_fluxo(args) -> None:
    from workflows.patient_flow import PatientFlow

    flow = PatientFlow()
    state = flow.start(args.paciente, solicitante=args.medico)
    print(f"\n[trace_id {state['trace_id']}] etapas: {' -> '.join(state.get('etapas', []))}")
    if state.get("status") == "PACIENTE_NAO_ENCONTRADO":
        print("Paciente nao encontrado.")
        return
    print(f"\nExames pendentes: {[e['tipo'] + (' (ATRASADO)' if e.get('atrasado') else '') for e in state['exames_pendentes']]}")
    criticos = [f"{e['tipo']}={e['valor']}" for e in state["resultados_criticos"]]
    print(f"Resultados criticos: {criticos}")
    print("\nAlertas emitidos:")
    for a in state.get("alertas", []) or [{"severidade": "-", "mensagem": "nenhum"}]:
        print(f"  [{a['severidade']}] {a['mensagem']}")
    print(f"\nSugestao do assistente:\n{state['sugestao']}")
    print(f"\nFontes: {sorted({f['fonte'] + '/' + f['secao'] for f in state['fontes']})}")
    print(f"\nStatus: {state['status']}")

    decisao = args.decisao
    if decisao is None:
        decisao = input("\nMedico, aprovar a sugestao? [a]provar / [r]ejeitar: ").strip().lower()
    aprovado = decisao.startswith("a")
    comentario = args.comentario or ("" if args.decisao else input("Comentario (opcional): ").strip())
    final = flow.decide(state["trace_id"], args.medico, aprovado, comentario)
    print(f"\nStatus final: {final['status']} | etapas: {' -> '.join(final['etapas'])}")
    print(f"Auditoria: python app.py auditoria {final['trace_id']}")


def cmd_auditoria(args) -> None:
    from safety.audit import read_trace

    events = read_trace(args.trace_id)
    if not events:
        print("Nenhum evento para esse trace_id.")
    _print_json(events)


def cmd_diagrama(args) -> None:
    from core.config import settings
    from workflows.patient_flow import PatientFlow

    mmd = PatientFlow().mermaid()
    settings.docs_dir.mkdir(exist_ok=True)
    path = settings.docs_dir / "fluxo_langgraph.mmd"
    path.write_text(mmd, encoding="utf-8")
    print(mmd)
    print(f"\nSalvo em {path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Assistente medico - Tech Challenge Fase 3")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="gera dados sinteticos, SQLite e dataset")
    p.add_argument("--medquad", action="store_true", help="inclui o MedQuAD (baixa do Hugging Face)")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("perguntar", help="pergunta ao assistente")
    p.add_argument("pergunta")
    p.add_argument("--paciente")
    p.add_argument("--medico", default="medico_demo")
    p.set_defaults(func=cmd_perguntar)

    p = sub.add_parser("fluxo", help="executa o fluxo LangGraph para um paciente")
    p.add_argument("paciente")
    p.add_argument("--medico", default="medico_demo")
    p.add_argument("--decisao", choices=["aprovar", "rejeitar"])
    p.add_argument("--comentario")
    p.set_defaults(func=cmd_fluxo)

    p = sub.add_parser("auditoria", help="mostra o log de auditoria de um trace_id")
    p.add_argument("trace_id")
    p.set_defaults(func=cmd_auditoria)

    p = sub.add_parser("diagrama", help="exporta o diagrama Mermaid do LangGraph")
    p.set_defaults(func=cmd_diagrama)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
