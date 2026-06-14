"""
Bootstrap ML model and RAG knowledge-base artifacts when missing.

Streamlit Community Cloud uses ephemeral disk, so artifacts are rebuilt on cold
starts unless you commit pre-built files or mount external storage.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_PATH = PROJECT_ROOT / "models" / "loan_xgb.pkl"
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "faiss_index.bin"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.jsonl"
LOAN_CSV_PATH = PROJECT_ROOT / "data" / "Loan_default.csv"


def get_artifact_status() -> Dict[str, Any]:
    """Return which runtime artifacts exist on disk."""
    return {
        "model_ready": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "kb_ready": FAISS_INDEX_PATH.exists() and METADATA_PATH.exists(),
        "faiss_index_path": str(FAISS_INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
        "loan_csv_ready": LOAN_CSV_PATH.exists(),
        "loan_csv_path": str(LOAN_CSV_PATH),
    }


def bootstrap_if_needed(quick: bool = True, force: bool = False) -> Dict[str, Any]:
    """
    Ensure ML model and demo RAG index exist.

    Args:
        quick: Use faster training / smaller demo KB (recommended for cloud).
        force: Rebuild even when artifacts already exist.

    Returns:
        Summary dict with booleans for each step.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    status = get_artifact_status()
    results: Dict[str, Any] = {
        "loan_csv_created": False,
        "model_trained": False,
        "kb_built": False,
        **status,
    }

    if force or not status["loan_csv_ready"]:
        from scripts.generate_synthetic_loan_data import generate_synthetic_loan_csv

        generate_synthetic_loan_csv(
            output_path=str(LOAN_CSV_PATH),
            n_rows=3000 if quick else 8000,
        )
        results["loan_csv_created"] = True
        logger.info("Synthetic loan CSV created at %s", LOAN_CSV_PATH)

    if force or not status["model_ready"]:
        from scripts.export_model_from_notebook import train_and_export_model

        trained = train_and_export_model(
            csv_path=str(LOAN_CSV_PATH),
            quick=quick,
        )
        results["model_trained"] = bool(trained)
        if not trained:
            raise RuntimeError("Failed to train ML model during bootstrap.")

    if force or not status["kb_ready"]:
        from scripts.create_demo_kb import build_demo_knowledge_base

        build_demo_knowledge_base(
            index_path=str(FAISS_INDEX_PATH),
            metadata_path=str(METADATA_PATH),
            quick=quick,
        )
        results["kb_built"] = True

    results.update(get_artifact_status())
    return results
