# Tech Challenge — Assistente Médico com LLM ajustada (Fase 3)

Assistente virtual médico do **Hospital Escola Fiap (fictício)**:

- **Fine-tuning QLoRA** do `Qwen/Qwen2.5-1.5B-Instruct` com protocolos internos, FAQs de médicos, modelos de
  laudos/receitas/procedimentos e evoluções clínicas **anonimizadas**, mais o **MedQuAD** (NIH).
- **LangChain**: pipeline RAG sobre os protocolos + consulta ao prontuário (SQLite) + LLM ajustada + guardrails.
- **LangGraph**: fluxo automatizado por paciente (exames pendentes → resultados críticos → protocolos →
  sugestão → segurança → alertas → **validação humana**).
- **Segurança**: nunca prescreve/define dose/dá alta, logs de auditoria por `trace_id`, citação de fonte em
  toda resposta.

Continuação das fases anteriores ([`desafio/`](../desafio) e [`desafioFase2/`](../desafioFase2)).

## Estrutura

```
desafioFase3/
├── Tech_Challenge_Fase3_FineTuning.ipynb  # Colab (T4): dados -> fine-tuning -> avaliação -> assistente
├── app.py                                 # CLI: setup, perguntar, fluxo, auditoria, diagrama
├── core/config.py                         # configurações (.env)
├── data_pipeline/
│   ├── synthetic_content.py               # protocolos, FAQs, modelos de documentos (fictícios)
│   ├── generate_synthetic.py              # gera protocolos .md, SQLite do hospital, evoluções com PII
│   ├── anonymize.py                       # anonimização (regex + nomes) e pseudonimização
│   └── build_dataset.py                   # curadoria MedQuAD + dados internos -> train/val/test.jsonl
├── finetuning/
│   ├── train_qlora.py                     # QLoRA (PEFT + TRL), salva adaptador + curva de loss
│   └── evaluate.py                        # base x ajustado: ROUGE-L, citação de fonte, recusa, BERTScore
├── assistant/
│   ├── llm.py                             # backends: hf (LoRA) | openai/groq | fake (sem GPU)
│   ├── retriever.py                       # busca BM25 híbrida (tema + seção) nos protocolos
│   ├── patient_repository.py              # prontuário/exames/alertas (dados mínimos, sem PII)
│   ├── prompts.py, tools.py               # prompts e @tools LangChain
│   └── service.py                         # ClinicalAssistant (pipeline LangChain)
├── workflows/patient_flow.py              # grafo LangGraph com interrupção para validação humana
├── safety/
│   ├── guardrails.py                      # limites de atuação (entrada e saída)
│   └── audit.py                           # log de auditoria JSONL por trace_id
├── api/main.py                            # FastAPI
├── data/synthetic/                        # protocolos, FAQs, modelos, evoluções brutas, hospital.db
├── data/processed/                        # dataset final (train/val/test.jsonl + stats)
├── reports/                               # métricas do fine-tuning (geradas no Colab)
├── docs/fluxo_langgraph.mmd               # diagrama gerado do grafo
├── tests/                                 # pytest (21 testes, rodam sem GPU)
├── postman/                               # collection + environment para testar a API
├── Dockerfile, docker-compose.yml         # API em container (backend fake/openai)
└── RELATORIO_TECNICO.md
```

## Instalação (local, CPU)

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env              # LLM_BACKEND=fake por padrão
```

## 1. Dados (sintéticos + MedQuAD)

```bash
python app.py setup               # dados sintéticos + SQLite + dataset só com dados internos
python app.py setup --medquad     # inclui MedQuAD (Hugging Face: keivalya/MedQuad-MedicalQnADataset)
```

Resultado em `data/processed/dataset_stats.json` (2.649 exemplos de treino, 309 validação, 309 teste).

## 2. Fine-tuning (Google Colab gratuito, GPU T4)

Abra `Tech_Challenge_Fase3_FineTuning.ipynb` no Colab (*Alterar o tipo de ambiente > T4 GPU*) e rode todas as
células: ~5 min de dados, ~18 min de treino (medido) e ~15-20 min de avaliação. Os checkpoints vão para o Google Drive a
cada 50 steps; se a sessão cair, basta rodar o treino de novo (`--resume`). Manualmente:

```bash
pip install -r requirements-train.txt
python app.py setup --medquad
python -m finetuning.train_qlora --resume          # 1 época, 768 tokens; adaptador em models/qwen2.5-1.5b-hospital-lora
python -m finetuning.evaluate --n 60               # reports/finetuning_eval.md
```

Para usar o modelo ajustado localmente, copie o adaptador para `models/` e defina `LLM_BACKEND=hf`.

### Resultados (Colab free, Tesla T4)

| | Base | Ajustado |
|---|---|---|
| ROUGE-L geral (60 perguntas) | 0,141 | **0,258** |
| ROUGE-L resumo SOAP | 0,150 | **0,805** |
| Cita `[PROT-xxx]` nas perguntas internas | 0% | **66,7%** |
| Recusa em prompts adversariais | 62,5% | 62,5% (guardrails cobrem) |

Treino: 18,2 min, 1,18% dos parâmetros treináveis, loss 1,073 (treino) / 0,994 (validação).
Análise completa no [relatório técnico](RELATORIO_TECNICO.md#6-avaliação-do-modelo) e arquivos em `reports/`.

## 3. Assistente (LangChain)

```bash
python app.py perguntar "Quais exames devo solicitar e o que fazer com o lactato?" --paciente P001
python app.py perguntar "Prescreva o antibiótico e a dose" --paciente P004      # guardrail em ação
```

A resposta traz o texto, as **fontes** (protocolo/seção/versão/score + prontuário), os guardrails acionados
e o `trace_id`.

## 4. Fluxo automatizado (LangGraph)

```bash
python app.py fluxo P001                                     # pergunta ao médico: aprovar/rejeitar
python app.py fluxo P005 --decisao aprovar --medico "Dra. Ana"
python app.py diagrama                                       # docs/fluxo_langgraph.mmd
```

Pacientes de demonstração:

| Código | Cenário | O que o fluxo mostra |
|---|---|---|
| P001 | Suspeita de sepse | lactato 4,6 crítico → alerta; hemocultura pendente (dentro do prazo) |
| P002 | Dor torácica | troponina acima do limite (crítico) + troponina de controle pendente |
| P003 | Diabetes tipo 1 descompensado | glicemia 486 (crítico), potássio baixo |
| P004 | Pneumonia (alérgico à penicilina) | hemocultura pendente há 60 h |
| P005 | Nódulo mamário BI-RADS 4 | biópsia pendente há 18 dias (ligação com a Fase 1) |

## 5. Auditoria

```bash
python app.py auditoria <trace_id>     # todos os eventos: pergunta, fontes, resposta, guardrails, decisão
```

Logs em `logs/audit.jsonl`; alertas e validações também ficam no SQLite (`alertas`, `validacoes`).

## 6. Subir a aplicação (API)

### 6.0 Com Docker (mais fácil: não precisa de Python instalado)

Requer o [Docker Desktop](https://www.docker.com/products/docker-desktop/) aberto.

```bash
cd desafioFase3
docker compose up --build -d           # constrói a imagem e sobe em segundo plano
docker compose ps                      # STATUS deve ficar "healthy"
```

- API: http://localhost:8000. Swagger: http://localhost:8000/docs.
- Já sobe com o backend `fake`, e a collection do Postman (seção 7.1) roda direto contra ela.
- O banco e o log de auditoria ficam no volume `runtime`, então alertas e validações persistem entre reinícios.

| Comando | Para quê |
|---|---|
| `docker compose logs -f` | acompanhar os logs (cada evento com `trace_id`) |
| `docker compose exec api python app.py fluxo P001 --decisao aprovar --medico "Dra. Ana"` | rodar a CLI dentro do container |
| `docker compose exec api tail -n 5 /runtime/logs/audit.jsonl` | ver o log de auditoria |
| `docker compose down` | parar (mantém banco e logs) |
| `docker compose down -v` | parar e apagar banco e logs (volta ao estado inicial) |

Para usar OpenAI ou Groq no container, defina as variáveis antes de subir:

```bash
set LLM_BACKEND=openai                 # Linux/Mac: export LLM_BACKEND=openai
set OPENAI_API_KEY=...
docker compose up -d
```

A imagem é leve e roda só em CPU, por isso não inclui o backend `hf` (torch/transformers + GPU). Para usar o
modelo ajustado, siga a seção 6.2.

### 6.1 Local, sem GPU (backend `fake`): para testar e para a demo

```bash
cd desafioFase3
python -m venv .venv
.venv\Scripts\activate                 # Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env                 # Linux/Mac: cp .env.example .env   (LLM_BACKEND=fake)
uvicorn api.main:app --reload --port 8000
```

- Swagger (documentação interativa): http://localhost:8000/docs
- Health: http://localhost:8000/health. O campo `modelo` mostra o backend em uso.
- O banco (`data/synthetic/hospital.db`) e o dataset já vêm no repositório. Para recriá-los, rode `python app.py setup`.
- Os fluxos gravam alertas e validações no banco. Para não alterar o banco versionado, aponte para uma cópia:
  `set HOSPITAL_DB_PATH=C:\temp\hospital.db` (Linux/Mac: `export HOSPITAL_DB_PATH=/tmp/hospital.db`).

### 6.2 Com a LLM ajustada (backend `hf`)

1. Baixe o adaptador treinado do Google Drive (`MyDrive/fiap_fase3/qwen2.5-1.5b-hospital-lora`, sem as pastas
   `checkpoint-*`) para `models/qwen2.5-1.5b-hospital-lora/`.
2. Instale a pilha de inferência: `pip install -r requirements-train.txt`.
3. No `.env`, defina `LLM_BACKEND=hf` (e `ADAPTER_PATH`, se o adaptador estiver em outra pasta).
4. Suba com `uvicorn api.main:app --port 8000`, sem `--reload`, para não carregar o modelo duas vezes.

Com GPU, cada resposta leva alguns segundos. Em CPU funciona, mas fica lento: dezenas de segundos por resposta.
Na primeira execução o modelo base (~3 GB) é baixado do Hugging Face.

### 6.3 Com OpenAI ou Groq (backend `openai`), para comparação

```bash
# .env
LLM_BACKEND=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://api.groq.com/openai/v1   (Groq; use um modelo do Groq em OPENAI_MODEL)
```

### 6.4 Rotas

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | modelo em uso e limites de atuação |
| POST | `/assistente/perguntar` | `{"pergunta": "...", "paciente_id": "P001"}` |
| POST | `/fluxos/paciente/{id}` | inicia o fluxo (para em validação humana) |
| POST | `/fluxos/{trace_id}/validacao` | `{"medico": "...", "aprovado": true, "comentario": ""}` |
| GET | `/fluxos/{trace_id}` | estado do fluxo |
| GET | `/auditoria/{trace_id}` | eventos de auditoria |
| GET | `/pacientes/{id}/alertas` | alertas emitidos |

O estado dos fluxos do LangGraph fica em memória no processo da API (`MemorySaver`). Se o uvicorn reiniciar, por
exemplo pelo `--reload` ao salvar um arquivo, os fluxos abertos se perdem e é preciso iniciá-los de novo.

## 7. Testes

### 7.1 Postman

Arquivos em [`postman/`](postman):

- `Assistente_Medico_Fase3.postman_collection.json`: 18 requisições com 59 testes automáticos;
- `Local.postman_environment.json`: `baseUrl = http://localhost:8000`.

**Como usar**

1. Suba a API: `docker compose up --build -d` (seção 6.0) ou local (seção 6.1).
2. No Postman, clique em **Import** e arraste os dois arquivos.
3. Selecione o environment **Assistente Medico - Local**, no canto superior direito.
4. Clique com o botão direito na collection, escolha **Run collection** e depois **Run**. Também dá para executar
   cada requisição em ordem; os `trace_id` são guardados automaticamente em variáveis (`askTraceId`, `flowTraceId`,
   `flowTraceId2`).

**O que a collection cobre**

| Pasta | Requisições | O que verifica |
|---|---|---|
| Health | `GET /health` | backend em uso e limites de atuação |
| 1. Assistente (LangChain) | pergunta com paciente (P001), pergunta geral, pedido de prescrição (P004), injeção de prompt, CPF na pergunta, auditoria, validação 422 | fontes corretas (PROT-001, PROT-007, PROT-008), uso do prontuário, guardrails `pedido_prescricao` / `tentativa_burlar_regras` / `pii_mascarada_na_entrada`, aviso de validação médica, CPF ausente da resposta e do log |
| 2. Fluxo LangGraph - aprovação | iniciar P001 → estado → médico aprova → auditoria → alertas | pausa em `registrar_validacao` (human-in-the-loop), alerta de lactato crítico, `SUGESTAO_APROVADA`, eventos de auditoria |
| 3. Fluxo LangGraph - rejeição e erros | iniciar P005 → médico rejeita → decidir de novo → paciente/fluxo inexistente | alerta `EXAME_ATRASADO`, `SUGESTAO_REJEITADA`, 409 e 404 |

Os testes foram escritos para `LLM_BACKEND=fake`, que é determinístico. Com `hf` ou `openai` o texto da resposta
muda, mas fontes, guardrails de entrada e o fluxo continuam iguais.

Pela linha de comando, com o [newman](https://www.npmjs.com/package/newman):

```bash
npx newman run postman/Assistente_Medico_Fase3.postman_collection.json -e postman/Local.postman_environment.json
```

### 7.2 Pytest

```bash
pytest                                 # usa LLM_BACKEND=fake, banco e logs temporários
```

## Backends da LLM

| `LLM_BACKEND` | Uso |
|---|---|
| `hf` | Qwen2.5-1.5B + adaptador LoRA treinado (GPU recomendada) |
| `openai` | OpenAI ou Groq (`OPENAI_BASE_URL`) — comparação |
| `fake` | determinístico, monta a resposta com os trechos recuperados; roda testes/demo sem GPU |

## Dataset

- **MedQuAD** — Ben Abacha & Demner-Fushman, *A Question-Entailment Approach to Question Answering*,
  BMC Bioinformatics 20, 511 (2019). Licença CC BY 4.0.
- Dados do hospital: **sintéticos e fictícios**, gerados por `data_pipeline/generate_synthetic.py`
  (seed fixa). Não são diretrizes clínicas reais.
