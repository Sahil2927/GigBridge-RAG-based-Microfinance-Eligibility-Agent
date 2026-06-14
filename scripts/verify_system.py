#!/usr/bin/env python3
"""
Pre-deployment connectivity audit for ClariFI / GigBridge.

Traces each pipeline (ML, RAG, storage, bootstrap) and reports pass/fail
with evidence. Run: python scripts/verify_system.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
WARN = "WARN"

results: List[Tuple[str, str, str]] = []


def record(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))
    icon = {"PASS": "[OK]", "FAIL": "[!!]", "SKIP": "[--]", "WARN": "[??]"}.get(status, "[??]")
    print(f"{icon} {status:4} | {name}")
    print(f"         {detail}\n")


def check_artifacts() -> None:
    from src.setup.artifacts import get_artifact_status

    status = get_artifact_status()
    ok = status["model_ready"] and status["kb_ready"]
    record(
        "Runtime artifacts",
        PASS if ok else FAIL,
        f"model={status['model_ready']} ({status['model_path']}), "
        f"kb={status['kb_ready']} (faiss={Path(status['faiss_index_path']).exists()})",
    )


def check_profile_pipeline() -> Dict[str, Any] | None:
    try:
        from src.profile.profile_builder import build_profile_from_json
        from src.utils.validators import validate_profile_schema, sanitize_profile

        sample = PROJECT_ROOT / "example_inputs" / "likely_eligible_gig_worker.json"
        with open(sample, encoding="utf-8") as f:
            raw = json.load(f)

        profile = build_profile_from_json(raw)
        valid, errors = validate_profile_schema(profile)
        if not valid:
            record("Profile pipeline", FAIL, f"validation errors: {errors}")
            return None

        profile = sanitize_profile(profile)
        record(
            "Profile pipeline",
            PASS,
            f"user_id={profile['user_id']}, sections="
            f"{list(k for k in profile if not k.startswith('_'))}",
        )
        return profile
    except Exception as e:
        record("Profile pipeline", FAIL, f"{type(e).__name__}: {e}")
        return None


def check_faiss_retrieval() -> None:
    try:
        from src.kb.retriever import retrieve_chunks

        chunks = retrieve_chunks("psychometric predictors loan repayment microfinance", k=3)
        if not chunks:
            record("FAISS retrieval", FAIL, "retrieve_chunks returned 0 results")
            return
        record(
            "FAISS retrieval",
            PASS,
            f"retrieved {len(chunks)} chunks; top score={chunks[0].get('score', 'n/a'):.4f}, "
            f"source={chunks[0].get('paper_title', chunks[0].get('source_file'))}",
        )
    except Exception as e:
        record("FAISS retrieval", FAIL, f"{type(e).__name__}: {e}")


def check_ml_pipeline() -> None:
    try:
        from src.ml.prediction_service import PredictionService

        sample = PROJECT_ROOT / "example_inputs" / "to_test_ml_model.json"
        with open(sample, encoding="utf-8") as f:
            ml_inputs = json.load(f)

        service = PredictionService()
        result = service.predict_from_raw_inputs(ml_inputs, consent=False, user_id="verify_user")

        required = {"prediction", "probability", "explanation", "next_step", "meta"}
        missing = required - set(result.keys())
        if missing:
            record("ML pipeline", FAIL, f"missing response keys: {missing}")
            return

        record(
            "ML pipeline",
            PASS,
            f"prediction={result['prediction']}, probability={result['probability']:.4f}, "
            f"shap_features={len(result.get('explanation') or [])}, "
            f"next_step={result['next_step'].get('action')}",
        )
    except Exception as e:
        record("ML pipeline", FAIL, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def check_groq_config() -> Tuple[bool, str | None, str]:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL_NAME", "mixtral-8x7b-32768")

    if not api_key or api_key == "your-groq-api-key-here":
        record(
            "Groq API config",
            SKIP,
            "GROQ_API_KEY not set (.env missing or placeholder). RAG live API test skipped.",
        )
        return False, None, model

    if not api_key.startswith("gsk_"):
        record("Groq API config", WARN, f"key set (len={len(api_key)}) but does not start with gsk_")
    else:
        record("Groq API config", PASS, f"key present, model={model}")

    return True, api_key, model


def check_groq_api_live(api_key: str, model: str) -> None:
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
            temperature=0,
        )
        text = (response.choices[0].message.content or "").strip()
        record("Groq API live call", PASS, f"model={model}, response={text!r}")
    except Exception as e:
        record("Groq API live call", FAIL, f"{type(e).__name__}: {e}")


def check_rag_pipeline(profile: Dict[str, Any] | None, api_key: str | None, model: str) -> None:
    if profile is None:
        record("RAG pipeline", SKIP, "no profile from profile pipeline")
        return
    if not api_key:
        record("RAG pipeline", SKIP, "requires GROQ_API_KEY for full assess_eligibility()")
        return

    try:
        from src.agent.rag_agent import RAGAgent

        agent = RAGAgent(groq_api_key=api_key, model_name=model)
        decision = agent.assess_eligibility(profile, k=4)

        required = {
            "eligibility",
            "risk_score",
            "verdict_text",
            "strong_points",
            "weak_points",
            "actionable_recommendations",
            "confidence",
        }
        missing = required - set(decision.keys())
        if missing:
            record("RAG pipeline", FAIL, f"missing decision keys: {missing}")
            return

        record(
            "RAG pipeline",
            PASS,
            f"eligibility={decision['eligibility']}, risk={decision['risk_score']:.2f}, "
            f"confidence={decision['confidence']}, strong_points={len(decision['strong_points'])}",
        )
    except Exception as e:
        record("RAG pipeline", FAIL, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def check_memory_consent() -> None:
    try:
        from src.memory.store import (
            save_user_profile,
            get_user_profile,
            save_consented_submission,
            get_bank_submissions,
            delete_user_data,
        )

        uid = "verify_connectivity_user"
        profile = {
            "user_id": uid,
            "timestamp": "2026-01-01T00:00:00",
            "demographics": {"age": 30, "gender": "female", "occupation": "gig_worker", "monthly_income": 20000},
        }
        decision = {"overall": {"eligibility": "yes", "risk_score": 0.2, "assessment_type": "verify"}}

        save_user_profile(uid, profile)
        loaded = get_user_profile(uid)
        if not loaded or loaded["user_id"] != uid:
            record("Memory / consent storage", FAIL, "save_user_profile / get_user_profile round-trip failed")
            return

        save_consented_submission(uid, profile, decision)
        subs = get_bank_submissions(limit=100)
        found = any(s.get("user_id") == uid for s in subs)
        delete_user_data(uid)

        sub_file = PROJECT_ROOT / "data" / "consented_submissions" / f"{uid}.json"
        if sub_file.exists():
            sub_file.unlink(missing_ok=True)

        record(
            "Memory / consent storage",
            PASS if found else FAIL,
            f"profile round-trip OK, consented submission saved={found}, cleanup done",
        )
    except Exception as e:
        record("Memory / consent storage", FAIL, f"{type(e).__name__}: {e}")


def check_app_imports() -> None:
    try:
        import app  # noqa: F401

        record("Streamlit app import", PASS, "app.py imports without error (UI not started)")
    except Exception as e:
        record("Streamlit app import", FAIL, f"{type(e).__name__}: {e}")


def main() -> int:
    print("=" * 72)
    print("ClariFI pre-deployment connectivity audit")
    print("=" * 72 + "\n")

    check_artifacts()
    profile = check_profile_pipeline()
    check_faiss_retrieval()
    check_ml_pipeline()
    has_key, api_key, model = check_groq_config()
    if has_key and api_key:
        check_groq_api_live(api_key, model)
    check_rag_pipeline(profile, api_key, model)
    check_memory_consent()
    check_app_imports()

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    counts = {PASS: 0, FAIL: 0, SKIP: 0, WARN: 0}
    for _, status, _ in results:
        counts[status] = counts.get(status, 0) + 1

    for name, status, detail in results:
        print(f"  {status:4}  {name}")

    print()
    print(
        f"Total: {len(results)} checks | "
        f"PASS={counts[PASS]} FAIL={counts[FAIL]} SKIP={counts[SKIP]} WARN={counts[WARN]}"
    )

    if counts[FAIL]:
        print("\nFix FAIL items before deployment.")
        return 1
    if counts[SKIP]:
        print("\nSKIP items need .env / GROQ_API_KEY for full RAG on deploy.")
    else:
        print("\nAll connected checks passed.")
    return 0 if not counts[FAIL] else 1


if __name__ == "__main__":
    raise SystemExit(main())
