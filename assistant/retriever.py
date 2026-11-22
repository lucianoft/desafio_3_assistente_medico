"""Recuperacao dos protocolos internos (RAG) para contextualizar e EXPLICAR as respostas.

Cada protocolo e quebrado por secao (## ...) e vira um Document com metadados de fonte
(id, titulo, secao, versao). Busca lexical BM25 (leve, roda em CPU, sem download de embeddings);
pode ser trocada por um vector store (FAISS/Chroma) mantendo a interface BaseRetriever.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

from core.config import settings

_STOP = set("a o e de da do das dos em no na nos nas para por com sem um uma os as ao qual quais que como quando "
            "deve devo se ou ha mais the of and to in is what".split())


# siglas comuns -> termo usado nos protocolos
_SYNONYMS = {"ecg": "eletrocardiograma", "pa": "pressao arterial", "cad": "cetoacidose diabetica",
             "pac": "pneumonia comunidade", "sca": "sindrome coronariana", "iam": "infarto miocardio",
             "hb": "hemoglobina", "k": "potassio", "uti": "terapia intensiva uti"}


def tokenize(text: str) -> list[str]:
    """Minusculas, sem acentos, sem stopwords, siglas expandidas e stemming leve por prefixo
    (5 letras: 'estratificar' e 'estratificacao' -> 'estra'), suficiente para o portugues clinico."""
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    words = []
    for w in re.findall(r"[a-z0-9]+", text):
        words += _SYNONYMS.get(w, w).split()
    return [t[:5] for t in words if t not in _STOP and (len(t) > 1 or t.isdigit())]


def load_protocol_documents(protocols_dir: Path | None = None) -> list[Document]:
    docs = []
    for path in sorted(Path(protocols_dir or settings.protocols_dir).glob("PROT-*.md")):
        text = path.read_text(encoding="utf-8")
        header = text.splitlines()[0].lstrip("# ").strip()
        prot_id, _, titulo = header.partition(" - ")
        versao = re.search(r"Versao:\s*([\d.]+)", text)
        for block in re.split(r"\n## ", text)[1:]:
            secao, _, body = block.partition("\n")
            docs.append(Document(page_content=body.strip(), metadata={
                "source_id": prot_id, "titulo": titulo, "secao": secao.strip(),
                "versao": versao.group(1) if versao else None, "arquivo": path.name}))
    return docs


class ProtocolRetriever(BaseRetriever):
    """Busca hibrida em dois niveis, ambos BM25:
      * tema   -> qual protocolo trata do assunto (indice com o protocolo inteiro);
      * secao  -> qual trecho responde a pergunta (indice por secao).
    score = secao_normalizado + tema_normalizado (+ bonus para protocolos preferidos, ex.: o
    protocolo vinculado a internacao do paciente).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    documents: list[Document]
    k: int = 4
    bm25_sections: BM25Okapi | None = None
    bm25_topics: BM25Okapi | None = None
    topic_ids: list[str] = []
    prefer_bonus: float = 1.0

    @classmethod
    def from_dir(cls, protocols_dir: Path | None = None, k: int | None = None) -> "ProtocolRetriever":
        docs = load_protocol_documents(protocols_dir)
        if not docs:
            raise FileNotFoundError("Nenhum protocolo encontrado. Rode: python app.py setup")
        topics: dict[str, list[str]] = {}
        for d in docs:
            topics.setdefault(d.metadata["source_id"], [d.metadata["titulo"]]).append(d.page_content)
        return cls(documents=docs, k=k or settings.retriever_k,
                   bm25_sections=BM25Okapi([tokenize(f"{d.metadata['secao']} {d.page_content}") for d in docs]),
                   bm25_topics=BM25Okapi([tokenize(" ".join(v)) for v in topics.values()]),
                   topic_ids=list(topics))

    @staticmethod
    def _norm(scores) -> list[float]:
        top = float(max(scores)) if len(scores) else 0.0  # float nativo (checkpoint do LangGraph serializa)
        return [max(float(s), 0.0) / top if top > 0 else 0.0 for s in scores]

    def search(self, query: str, prefer: list[str] | None = None, k: int | None = None) -> list[Document]:
        tokens = tokenize(query)
        sec = self._norm(self.bm25_sections.get_scores(tokens))
        topic = dict(zip(self.topic_ids, self._norm(self.bm25_topics.get_scores(tokens))))
        prefer = set(prefer or [])
        final = []
        for i, d in enumerate(self.documents):
            sid = d.metadata["source_id"]
            score = sec[i] + topic[sid] + (self.prefer_bonus if sid in prefer else 0.0)
            if d.metadata["secao"] == "Objetivo":  # secao generica: menos util para responder
                score *= 0.7
            final.append(score if sec[i] > 0 or sid in prefer else 0.0)
        ranked = sorted(range(len(final)), key=lambda i: final[i], reverse=True)
        out = []
        for i in ranked[: k or self.k]:
            if final[i] <= 0:
                break
            doc = self.documents[i]
            out.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "score": round(final[i], 3)}))
        return out

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.search(query)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['source_id']} - {d.metadata['secao']}] ({d.metadata['titulo']})\n{d.page_content}"
                       for d in docs)


def doc_sources(docs: list[Document]) -> list[dict]:
    return [{"fonte": d.metadata["source_id"], "titulo": d.metadata["titulo"], "secao": d.metadata["secao"],
             "versao": d.metadata.get("versao"), "score": d.metadata.get("score"), "trecho": d.page_content[:200]}
            for d in docs]
