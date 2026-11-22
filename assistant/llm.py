"""Fabrica da LLM usada pelo LangChain.

Backends (variavel LLM_BACKEND):
  * hf     -> modelo base + adaptador LoRA treinado (transformers + peft), via langchain-huggingface;
  * openai -> API OpenAI ou compativel (OPENAI_BASE_URL, ex.: Groq) - util para comparacao;
  * fake   -> modelo deterministico local que monta a resposta a partir do contexto recuperado.
              Permite rodar fluxos, testes e a API sem GPU (o pipeline LangChain/LangGraph e o mesmo).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.config import settings


class FakeClinicalChatModel(BaseChatModel):
    """Simula a LLM ajustada: extrai trechos dos protocolos do prompt e responde citando a fonte."""

    @property
    def _llm_type(self) -> str:
        return "fake-clinical"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        prompt = str(messages[-1].content)
        blocks = re.findall(r"\[(PROT-\d{3} - [^\]]+)\] \([^)]*\)\n(.+?)(?=\n\n\[PROT-|\n\nCONTEXTO|\Z)", prompt, re.S)
        paciente = re.search(r"CONTEXTO DO PACIENTE:\n(.+?)\n\n", prompt, re.S)
        parts = []
        # prioriza secoes praticas (a "Objetivo" so entra se nao houver outra)
        blocks = sorted(blocks, key=lambda b: b[0].endswith("Objetivo"))
        if "Tarefa: elabore uma SUGESTAO" in prompt:
            pend = re.search(r"EXAMES PENDENTES:\n(.+?)\n\n", prompt, re.S)
            crit = re.search(r"RESULTADOS CRITICOS:\n(.+?)\n\n", prompt, re.S)
            parts.append("Sugestao de proximos passos (pendente de validacao medica):")
            if crit and "Nenhum" not in crit.group(1):
                parts.append(f"- Comunicar resultados criticos ao medico responsavel em ate 30 minutos: "
                             f"{crit.group(1).strip()} [PROT-007 - Comunicacao]")
            if pend and "Nenhum" not in pend.group(1):
                parts.append(f"- Cobrar exames pendentes junto ao laboratorio/imagem: {pend.group(1).strip()} "
                             f"[PROT-007 - Exames pendentes]")
            for ref, text in blocks[:2]:
                parts.append(f"- {text.strip()} [{ref}]")
        else:
            if paciente and "Nao informado" not in paciente.group(1):
                parts.append(f"Considerando o contexto do paciente [PRONTUARIO]: {paciente.group(1).splitlines()[1]}.")
            for ref, text in blocks[:2]:
                parts.append(f"{text.strip()} [{ref}]")
            if not blocks:
                parts.append("Nao encontrei essa informacao nos protocolos internos.")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="\n".join(parts)))])


def _build_hf() -> BaseChatModel:
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    base = settings.base_model
    adapter = settings.adapter_path
    tok = AutoTokenizer.from_pretrained(adapter if (adapter / "tokenizer_config.json").exists() else base)
    cuda = torch.cuda.is_available()
    from finetuning.compat import dtype_kwarg

    model = AutoModelForCausalLM.from_pretrained(base, **dtype_kwarg(torch.float16 if cuda else torch.float32),
                                                 device_map="auto" if cuda else None)
    if (adapter / "adapter_config.json").exists():
        from peft import PeftModel

        from finetuning.compat import patch_peft_torchao

        patch_peft_torchao()  # torchao antigo do Colab

        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()  # inferencia mais rapida
    gen = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=settings.max_new_tokens,
                   do_sample=False, return_full_text=False)
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen, model_id=base), tokenizer=tok)


def _build_openai() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.openai_model, temperature=0, base_url=settings.openai_base_url)


@lru_cache(maxsize=1)
def get_chat_model(backend: str | None = None) -> BaseChatModel:
    backend = (backend or settings.llm_backend).lower()
    if backend == "hf":
        return _build_hf()
    if backend == "openai":
        return _build_openai()
    if backend == "fake":
        return FakeClinicalChatModel()
    raise ValueError(f"LLM_BACKEND invalido: {backend}")


def model_description(backend: str | None = None) -> str:
    backend = (backend or settings.llm_backend).lower()
    if backend == "hf":
        return f"{settings.base_model} + LoRA ({settings.adapter_path.name})"
    if backend == "openai":
        return settings.openai_model
    return "fake-clinical (deterministico)"
