"""
Build a demo FAISS knowledge base without PDF files.

Uses embedded microfinance research snippets so RAG works out of the box on
Streamlit Cloud (no PDF upload step required).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEMO_RESEARCH_SNIPPETS: List[str] = [
    """
    Mobile phone metadata can proxy creditworthiness for unbanked populations.
    Higher unique contact counts and stable daily call patterns correlate with
    lower default rates in microfinance portfolios, especially among gig workers
    with irregular formal income documentation.
    """,
    """
    Psychometric assessments measuring conscientiousness (self-discipline,
    reliability, planning) show predictive power for loan repayment among
    borrowers without traditional credit histories. Scores above the median are
    associated with 15-25% lower default probability in field studies.
    """,
    """
    Airtime top-up frequency and average top-up amounts reflect liquidity
    management. Regular small top-ups suggest stable cash inflows. Extended
    periods of inactivity or highly volatile top-up amounts may indicate
    income instability and elevated credit risk.
    """,
    """
    Savings behavior metrics—including savings frequency, amount variance, and
    wallet balance lows—provide behavioral signals beyond income statements.
    Borrowers who save at least twice per month and maintain fewer balance lows
    demonstrate stronger repayment capacity in alternative-data scoring models.
    """,
    """
    Self-Help Group (SHG) membership and peer monitoring strength capture social
    collateral effects. SHG members with active peer monitoring show improved
    repayment rates due to reputational incentives and informal enforcement
    within community lending circles.
    """,
    """
    Bill payment timeliness derived from digital wallet or utility payment logs
    is a strong proxy for credit discipline. Timeliness scores above 0.8
    typically support eligibility for small-ticket microloans when combined with
    stable mobile usage patterns.
    """,
    """
    Prior loan history remains informative even when using unconventional data.
    Previous defaults are high-risk indicators. Occasional late payments with
    zero defaults may be acceptable for borderline cases when alternative signals
    (psychometrics, savings, SHG membership) are strong.
    """,
    """
    Ethical alternative-data lending requires explicit consent, transparent
    feature use, and the ability to delete raw behavioral logs. Only aggregated
    features should be shared with lenders; psychometric item-level responses
    should remain confidential.
    """,
]


def build_demo_knowledge_base(
    index_path: str = "./data/faiss_index.bin",
    metadata_path: str = "./data/metadata.jsonl",
    quick: bool = True,
) -> None:
    """Create FAISS index + metadata from embedded demo research text."""
    from src.kb.ingest import (
        chunk_text,
        create_embeddings,
        build_faiss_index,
        save_index_and_metadata,
    )

    snippets = DEMO_RESEARCH_SNIPPETS if quick else DEMO_RESEARCH_SNIPPETS * 2
    chunks: List[Dict[str, Any]] = []

    for doc_idx, snippet in enumerate(snippets):
        for chunk in chunk_text(snippet.strip(), chunk_size=120, overlap=20):
            chunks.append(
                {
                    "text": chunk["text"],
                    "chunk_index": chunk["chunk_index"],
                    "source_file": f"demo_research_{doc_idx + 1}.txt",
                    "page_number": 1,
                }
            )

    if not chunks:
        raise RuntimeError("No demo chunks generated.")

    logger.info("Creating embeddings for %s demo chunks...", len(chunks))
    embeddings = create_embeddings(chunks, model_name="all-MiniLM-L6-v2")
    index = build_faiss_index(embeddings, index_type="flat")
    save_index_and_metadata(index, chunks, index_path, metadata_path)
    logger.info("Demo knowledge base saved to %s", index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build demo FAISS knowledge base")
    parser.add_argument("--index-path", default="./data/faiss_index.bin")
    parser.add_argument("--metadata-path", default="./data/metadata.jsonl")
    args = parser.parse_args()

    build_demo_knowledge_base(args.index_path, args.metadata_path)
