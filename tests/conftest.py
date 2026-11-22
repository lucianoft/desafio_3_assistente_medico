import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ambiente isolado: banco e logs temporarios, LLM fake (nao precisa de GPU)
_TMP = Path(tempfile.mkdtemp(prefix="fase3_tests_"))
os.environ["HOSPITAL_DB_PATH"] = str(_TMP / "hospital.db")
os.environ["LOGS_DIR"] = str(_TMP / "logs")
os.environ["LLM_BACKEND"] = "fake"

import pytest  # noqa: E402

from core.config import settings  # noqa: E402
from data_pipeline.generate_synthetic import build_database, write_protocols  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def synthetic_data():
    write_protocols(settings.protocols_dir)
    notes = build_database(settings.db_path)
    return notes
