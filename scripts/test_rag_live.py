#!/usr/bin/env python3
"""Live RAG end-to-end test with step-by-step trace."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


def step(n: int, title: str) -> None:
    print(f"\n--- Step {n}: {title} ---")


def main() -> int:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

    step(1, "Check Groq API key")
    if not api_key or api_key == "your-groq-api-key-here":
        print("FAIL: Set GROQ_API_KEY in .env (get one at https://console.groq.com/)")
        return 1
    print(f"OK: key length={len(api_key)}, model={model}")

    step(2, "Build profile from example_inputs/likely_eligible_gig_worker.json")
    from src.profile.profile_builder import build_profile_from_json
    from src.utils.validators import validate_profile_schema

    sample = PROJECT_ROOT / "example_inputs" / "likely_eligible_gig_worker.json"
    with open(sample, encoding="utf-8") as f:
        profile = build_profile_from_json(json.load(f))
    valid, errors = validate_profile_schema(profile)
    if not valid:
        print(f"FAIL: profile validation: {errors}")
        return 1
    print(f"OK: user_id={profile['user_id']}")

    step(3, "FAISS retrieval (local, no API)")
    from src.kb.retriever import retrieve_chunks

    query = "microfinance eligibility psychometric mobile metadata repayment"
    chunks = retrieve_chunks(query, k=4)
    print(f"OK: {len(chunks)} chunks retrieved")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] score={c.get('score', 0):.4f} source={c.get('paper_title', c.get('source_file'))}")

    step(4, "Groq API ping")
    from groq import Groq

    client = Groq(api_key=api_key)
    ping = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=5,
        temperature=0,
    )
    print(f"OK: Groq responded: {(ping.choices[0].message.content or '').strip()!r}")

    step(5, "Full RAG assessment (FAISS + Groq LLM)")
    from src.agent.rag_agent import RAGAgent

    agent = RAGAgent(groq_api_key=api_key, model_name=model)
    decision = agent.assess_eligibility(profile, k=4)

    print("OK: RAG decision received")
    print(json.dumps(
        {
            "eligibility": decision.get("eligibility"),
            "risk_score": decision.get("risk_score"),
            "confidence": decision.get("confidence"),
            "verdict_text": decision.get("verdict_text"),
            "strong_points": decision.get("strong_points", [])[:3],
            "weak_points": decision.get("weak_points", [])[:3],
        },
        indent=2,
    ))

    print("\n=== ALL RAG STEPS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
