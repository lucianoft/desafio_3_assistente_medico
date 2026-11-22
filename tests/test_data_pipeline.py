from data_pipeline.anonymize import anonymize_text, contains_pii, pseudonymize_id
from data_pipeline.build_dataset import curate_medquad, note_examples, protocol_examples, safety_examples, split
from core.config import settings


def test_anonymize_removes_identifiers():
    text = ("Paciente Maria Clara Souza, CPF 123.456.789-09, tel (11) 98765-4321, email maria@ex.com, "
            "internada em 05/11/2026.")
    res = anonymize_text(text)
    assert "Maria" not in res.text and "123.456" not in res.text and "98765" not in res.text
    assert "maria@ex.com" not in res.text and "05/11/2026" not in res.text
    assert {"NOME", "CPF", "TELEFONE", "EMAIL", "DATA"} <= set(res.replacements)
    assert not contains_pii(res.text)


def test_anonymize_keeps_clinical_values():
    res = anonymize_text("Glicemia 486 mg/dL, lactato 4,6 mmol/L, PA 92x58 mmHg.")
    assert res.text == "Glicemia 486 mg/dL, lactato 4,6 mmol/L, PA 92x58 mmHg."


def test_pseudonymize_is_stable():
    assert pseudonymize_id("P001") == pseudonymize_id("P001") != pseudonymize_id("P002")


def test_synthetic_notes_are_anonymized(synthetic_data):
    import json

    path = settings.synthetic_dir / "evolucoes_brutas.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(n, ensure_ascii=False) for n in synthetic_data), encoding="utf-8")
    examples, stats = note_examples(path)
    assert len(examples) == len(synthetic_data)
    for ex in examples:
        user = ex["messages"][1]["content"]
        assert not contains_pii(user)
    assert stats["CPF"] == len(synthetic_data)


def test_curate_medquad_filters_and_balances():
    rows = [{"qtype": "treatment", "Question": "How to treat X?", "Answer": "A" * 100},
            {"qtype": "treatment", "Question": "How to treat X?", "Answer": "B" * 100},  # duplicada
            {"qtype": "symptoms", "Question": "Symptoms of Y?", "Answer": None},       # sem resposta
            {"qtype": "symptoms", "Question": "What are symptoms of Z?", "Answer": "Fever and cough. " * 10}]
    ex, stats = curate_medquad(rows, max_samples=10, max_answer_chars=1500, seed=1)
    assert len(ex) == 2 and stats["duplicada"] == 1 and stats["sem_resposta"] == 1


def test_internal_examples_cite_source_and_split():
    prot = protocol_examples()
    assert all("[PROT-" in e["messages"][-1]["content"] for e in prot)
    parts = split(prot + safety_examples(), seed=1)
    assert all(e["kind"] != "seguranca" for e in parts["test"])
    assert parts["test"] and parts["val"]
