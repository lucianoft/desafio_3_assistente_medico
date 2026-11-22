"""Monta o dataset de instrucoes para o fine-tuning (formato chat / "messages").

Fontes:
  1. MedQuAD (Hugging Face: keivalya/MedQuad-MedicalQnADataset) - Q&A medico geral (ingles), com curadoria.
  2. Protocolos internos  -> perguntas por secao, resposta com citacao da fonte.
  3. FAQ de medicos       -> perguntas frequentes com fonte.
  4. Modelos de documentos (laudos, receitas, procedimentos).
  5. Evolucoes clinicas   -> ANONIMIZADAS e convertidas em resumo estruturado.
  6. Exemplos de seguranca -> recusa de prescricao/dose/alta (limites de atuacao).

Saidas (data/processed/): train.jsonl, val.jsonl, test.jsonl, dataset_stats.json

Uso:
  python -m data_pipeline.build_dataset                    # com MedQuAD (precisa de internet)
  python -m data_pipeline.build_dataset --skip-medquad     # so dados internos
  python -m data_pipeline.build_dataset --medquad-csv arquivo.csv
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from core.config import settings
from data_pipeline.anonymize import anonymize_text, contains_pii, pseudonymize_id
from data_pipeline.synthetic_content import (DOCUMENT_TEMPLATES, FAQ_MEDICOS, HOSPITAL, PROTOCOLS,
                                             SAFETY_EXAMPLES)

SYSTEM_PROMPT = (
    f"Voce e o assistente virtual medico do {HOSPITAL}. Apoia medicos com base nos protocolos internos "
    "e no contexto do paciente. Regras: nunca prescreva medicamentos, nunca defina doses e nunca de alta; "
    "toda sugestao de conduta depende de validacao medica; sempre cite a fonte (ex.: [PROT-001]); "
    "se nao souber, diga que nao encontrou a informacao."
)
SYSTEM_PROMPT_EN = (
    "You are a medical assistant answering clinicians' questions with evidence-based health information. "
    "Do not prescribe drugs or doses. Source: MedQuAD (NIH)."
)

HF_MEDQUAD = "keivalya/MedQuad-MedicalQnADataset"


def _msg(system: str, user: str, assistant: str, source: str, kind: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
        "kind": kind,
    }


# ----------------------------------------------------------------------------- MedQuAD
def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind(".\n"))
    return (cut[: end + 1] if end > max_chars * 0.5 else cut).strip()


def curate_medquad(rows: list[dict], max_samples: int, max_answer_chars: int, seed: int) -> tuple[list[dict], dict]:
    """Limpeza + deduplicacao + balanceamento por qtype."""
    stats = Counter(total_bruto=len(rows))
    seen: set[str] = set()
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        q = re.sub(r"\s+", " ", str(r.get("Question") or "")).strip()
        a = re.sub(r"\s+", " ", str(r.get("Answer") or "")).strip()
        qtype = str(r.get("qtype") or "unknown").strip()
        if not q or not a or a.lower() in ("nan", "none"):
            stats["sem_resposta"] += 1
            continue
        if len(a) < 40:
            stats["resposta_curta"] += 1
            continue
        key = q.lower()
        if key in seen:
            stats["duplicada"] += 1
            continue
        seen.add(key)
        by_type.setdefault(qtype, []).append({"q": q, "a": _truncate(a, max_answer_chars), "qtype": qtype})

    rng = random.Random(seed)
    for items in by_type.values():
        rng.shuffle(items)
    # round-robin entre os tipos -> balanceamento
    selected: list[dict] = []
    pools = {k: list(v) for k, v in by_type.items()}
    while len(selected) < max_samples and any(pools.values()):
        for k in sorted(pools):
            if pools[k] and len(selected) < max_samples:
                selected.append(pools[k].pop())
    stats["selecionadas"] = len(selected)
    stats["qtypes"] = len(by_type)
    examples = [
        {**_msg(SYSTEM_PROMPT_EN, s["q"], s["a"], "medquad", "medquad"), "qtype": s["qtype"]} for s in selected
    ]
    return examples, dict(stats)


def load_medquad(csv_path: str | None) -> list[dict]:
    if csv_path:
        import pandas as pd

        return pd.read_csv(csv_path).to_dict("records")
    from datasets import load_dataset

    return list(load_dataset(HF_MEDQUAD, split="train"))


# ----------------------------------------------------------------------------- dados internos
SECTION_QUESTIONS = {
    "Objetivo": "Qual o objetivo do protocolo de {titulo}?",
    "Conduta": "Qual a conduta recomendada no protocolo de {titulo}?",
    "Conduta inicial": "Qual a conduta inicial no protocolo de {titulo}?",
    "Exames": "Quais exames o protocolo de {titulo} recomenda?",
    "Criterios de alerta": "Quais situacoes geram alerta no protocolo de {titulo}?",
}


def protocol_examples() -> list[dict]:
    out = []
    for p in PROTOCOLS:
        for secao, texto in p["secoes"].items():
            pergunta = SECTION_QUESTIONS.get(secao, f"O que diz a secao '{secao}' do protocolo de {{titulo}}?")
            out.append(_msg(SYSTEM_PROMPT, pergunta.format(titulo=p["titulo"].lower()),
                            f"{texto} [{p['id']} - {secao}]", p["id"], "protocolo"))
    return out


def faq_examples() -> list[dict]:
    return [_msg(SYSTEM_PROMPT, f["pergunta"], f"{f['resposta']} [{f['fonte']}]", f["fonte"], "faq") for f in FAQ_MEDICOS]


def template_examples() -> list[dict]:
    return [
        _msg(SYSTEM_PROMPT, f"Gere o modelo institucional de {t['nome'].lower()}.", t["conteudo"], t["tipo"], "modelo")
        for t in DOCUMENT_TEMPLATES
    ]


def safety_examples() -> list[dict]:
    return [_msg(SYSTEM_PROMPT, s["pergunta"], s["resposta"], "PROT-008", "seguranca") for s in SAFETY_EXAMPLES]


def note_examples(notes_path: Path) -> tuple[list[dict], dict]:
    """Evolucoes brutas -> anonimizadas -> par (texto, resumo estruturado)."""
    out, stats = [], Counter()
    if not notes_path.exists():
        return out, {}
    for line in notes_path.read_text(encoding="utf-8").splitlines():
        n = json.loads(line)
        res = anonymize_text(n["texto"], known_names=[n["nome"]])
        stats.update(res.replacements)
        if contains_pii(res.text):
            stats["descartadas_pii_residual"] += 1
            continue
        resumo = (
            f"Paciente {pseudonymize_id(n['paciente_id'])}, {n['idade']} anos, sexo {n['sexo']}, setor {n['setor']}.\n"
            f"S: {n['evolucao']}\nA: {n['diagnostico']} (CID {n['cid']}). Alergias: {n['alergias']}.\n"
            "P: seguir protocolo institucional correspondente; conduta a definir pelo medico assistente."
        )
        out.append(_msg(SYSTEM_PROMPT, f"Resuma a evolucao clinica em formato SOAP:\n{res.text}", resumo,
                        "evolucao", "resumo"))
    return out, dict(stats)


# ----------------------------------------------------------------------------- split
def split(examples: list[dict], seed: int, val: float = 0.1, test: float = 0.1) -> dict[str, list[dict]]:
    """Split estratificado por 'kind'. Exemplos de seguranca ficam todos no treino
    (a avaliacao de seguranca usa prompts adversariais proprios, ver finetuning/evaluate.py)."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for ex in examples:
        groups.setdefault(ex["kind"], []).append(ex)
    parts = {"train": [], "val": [], "test": []}
    for kind, items in groups.items():
        rng.shuffle(items)
        if kind == "seguranca" or len(items) < 10:
            parts["train"] += items
            continue
        n_test, n_val = max(1, int(len(items) * test)), max(1, int(len(items) * val))
        parts["test"] += items[:n_test]
        parts["val"] += items[n_test:n_test + n_val]
        parts["train"] += items[n_test + n_val:]
    for v in parts.values():
        rng.shuffle(v)
    return parts


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-medquad", action="store_true")
    ap.add_argument("--medquad-csv", default=None)
    ap.add_argument("--medquad-samples", type=int, default=3000)  # tamanho pensado para o Colab free
    ap.add_argument("--max-answer-chars", type=int, default=1500)
    ap.add_argument("--internal-repeat", type=int, default=3,
                    help="repete os exemplos internos (poucos) para equilibrar com o MedQuAD")
    ap.add_argument("--out-dir", default=str(settings.processed_dir))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    stats: dict = {}
    internal = protocol_examples() + faq_examples() + template_examples() + safety_examples()
    notes, stats["anonimizacao"] = note_examples(settings.synthetic_dir / "evolucoes_brutas.jsonl")
    internal += notes
    stats["internos_unicos"] = len(internal)

    examples: list[dict] = []
    if not args.skip_medquad:
        medquad, stats["medquad"] = curate_medquad(load_medquad(args.medquad_csv), args.medquad_samples,
                                                   args.max_answer_chars, args.seed)
        examples += medquad

    parts = split(examples + internal, args.seed)
    # oversampling dos dados internos apenas no treino
    internal_train = [e for e in parts["train"] if e["kind"] != "medquad"]
    parts["train"] += internal_train * (args.internal_repeat - 1)
    random.Random(args.seed).shuffle(parts["train"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"{name}_total"] = len(rows)
        stats[f"{name}_por_tipo"] = dict(Counter(r["kind"] for r in rows))
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


if __name__ == "__main__":
    main()
