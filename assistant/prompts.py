"""Prompts do assistente (mesmo system prompt usado no fine-tuning)."""
from langchain_core.prompts import ChatPromptTemplate

from data_pipeline.build_dataset import SYSTEM_PROMPT

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "PROTOCOLOS RELEVANTES:\n{protocolos}\n\n"
     "CONTEXTO DO PACIENTE:\n{paciente}\n\n"
     "PERGUNTA DO MEDICO: {pergunta}\n\n"
     "Responda em portugues, de forma objetiva, usando apenas as informacoes acima. "
     "Cite as fontes entre colchetes, ex.: [PROT-001 - Conduta] ou [PRONTUARIO]."),
])

FLOW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "PROTOCOLOS RELEVANTES:\n{protocolos}\n\n"
     "CONTEXTO DO PACIENTE:\n{paciente}\n\n"
     "EXAMES PENDENTES:\n{pendentes}\n\n"
     "RESULTADOS CRITICOS:\n{criticos}\n\n"
     "Tarefa: elabore uma SUGESTAO de proximos passos para a equipe medica (exames a cobrar, pontos do protocolo "
     "a seguir, riscos a monitorar). Nao prescreva medicamentos nem doses. Cite as fontes entre colchetes."),
])
