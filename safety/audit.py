"""Log de auditoria (JSON Lines) para rastreamento de cada interacao/fluxo.

Cada evento tem: timestamp, trace_id, evento, usuario e dados. O mesmo trace_id liga a pergunta,
as fontes consultadas, o prompt enviado ao modelo, a resposta, os guardrails acionados, os alertas
e a decisao do medico.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.config import settings

_lock = threading.Lock()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("assistente")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _audit_file() -> Path:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    return settings.logs_dir / "audit.jsonl"


def log_event(trace_id: str, event: str, user: str = "sistema", **data) -> dict:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "trace_id": trace_id,
              "evento": event, "usuario": user, **data}
    with _lock, _audit_file().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    logger.info("trace=%s evento=%s %s", trace_id, event,
                {k: (v if len(str(v)) < 120 else str(v)[:117] + "...") for k, v in data.items()})
    return record


def read_trace(trace_id: str) -> list[dict]:
    path = _audit_file()
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("trace_id") == trace_id:
                events.append(rec)
    return events
