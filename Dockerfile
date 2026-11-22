# API do assistente medico (Fase 3) - imagem leve, roda em CPU.
# Backends suportados na imagem: fake (padrao, deterministico) e openai (OpenAI/Groq).
# O backend hf (Qwen + LoRA) precisa de torch/transformers e GPU: use o notebook do Colab ou a instalacao local.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY core/ core/
COPY data_pipeline/ data_pipeline/
COPY assistant/ assistant/
COPY safety/ safety/
COPY workflows/ workflows/
COPY api/ api/
COPY data/ data/
COPY app.py .

# Banco e logs em /runtime (volume): o banco versionado da imagem e copiado na primeira subida,
# assim alertas/validacoes gravados pelos fluxos persistem entre reinicios sem alterar a imagem.
ENV LLM_BACKEND=fake \
    HOSPITAL_DB_PATH=/runtime/hospital.db \
    LOGS_DIR=/runtime/logs

RUN useradd --create-home --uid 1000 app && mkdir -p /runtime/logs && chown -R app:app /runtime
USER app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["sh", "-c", "[ -f \"$HOSPITAL_DB_PATH\" ] || cp data/synthetic/hospital.db \"$HOSPITAL_DB_PATH\"; exec uvicorn api.main:app --host 0.0.0.0 --port 8000"]
