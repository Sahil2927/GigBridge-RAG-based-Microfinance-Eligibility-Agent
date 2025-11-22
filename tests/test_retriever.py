"""
Unit tests for retriever.
"""

import pytest
import os
from pathlib import Path
from src.kb.retriever import ChunkRetriever, retrieve_chunks


@pytest.mark.skipif(
    not Path("./data/faiss_index.bin").exists() or not Path("./data/metadata.jsonl").exists(),
    reason="FAISS index not found. Run ingestion first."
)
def test_retriever_initialization():
    """Test retriever initialization."""
    retriever = ChunkRetriever()
    assert retriever.index is not None
    assert len(retriever.chunks) > 0


@pytest.mark.skipif(
    not Path("./data/faiss_index.bin").exists() or not Path("./data/metadata.jsonl").exists(),
    reason="FAISS index not found. Run ingestion first."
)
def test_retrieve_chunks():
    """Test chunk retrieval."""
    query = "psychometric predictors of loan repayment"
    results = retrieve_chunks(query, k=5)
    
    assert len(results) > 0
    assert len(results) <= 5
    assert all("text" in chunk for chunk in results)
    assert all("score" in chunk for chunk in results)
    assert all("paper_title" in chunk for chunk in results)


@pytest.mark.skipif(
    not Path("./data/faiss_index.bin").exists() or not Path("./data/metadata.jsonl").exists(),
    reason="FAISS index not found. Run ingestion first."
)
def test_retrieve_chunks_with_threshold():
    """Test retrieval with score threshold."""
    query = "microfinance eligibility"
    results = retrieve_chunks(query, k=10)
    
    # Should have some results
    assert len(results) > 0
    
    # Test with threshold
    from src.kb.retriever import get_retriever
    retriever = get_retriever()
    filtered = retriever.retrieve_chunks_with_threshold(query, k=10, min_score=0.1)
    
    assert len(filtered) <= len(results)
    assert all(chunk["score"] >= 0.1 for chunk in filtered)


