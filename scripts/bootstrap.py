#!/usr/bin/env python3
"""
One-command local setup for ClariFI.

Creates:
  - data/Loan_default.csv (synthetic, if missing)
  - models/loan_xgb.pkl
  - data/faiss_index.bin + data/metadata.jsonl (demo KB)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.setup.artifacts import bootstrap_if_needed, get_artifact_status  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap ClariFI runtime artifacts")
    parser.add_argument("--force", action="store_true", help="Rebuild all artifacts")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use larger dataset / slower training (not recommended on Streamlit Cloud)",
    )
    args = parser.parse_args()

    print("ClariFI bootstrap")
    print("-" * 40)
    before = get_artifact_status()
    print(f"Before: model={before['model_ready']}, kb={before['kb_ready']}, csv={before['loan_csv_ready']}")

    results = bootstrap_if_needed(quick=not args.full, force=args.force)

    print("-" * 40)
    print(f"Loan CSV created: {results.get('loan_csv_created', False)}")
    print(f"Model trained:    {results.get('model_trained', False)}")
    print(f"KB built:         {results.get('kb_built', False)}")
    print(f"After:  model={results['model_ready']}, kb={results['kb_ready']}")
    print("\nNext: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
