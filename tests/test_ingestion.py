"""
Unit tests for knowledge base ingestion.
"""

import pytest
import os
from pathlib import Path
from src.kb.ingest import (
    extract_text_from_pdf, chunk_text, process_pdfs,
    create_embeddings, build_faiss_index
)


def test_chunk_text():
    """Test text chunking."""
    text = " ".join(["word"] * 1000)  # 1000 words
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    
    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)
    assert all("chunk_index" in chunk for chunk in chunks)


def test_chunk_text_empty():
    """Test chunking empty text."""
    chunks = chunk_text("", chunk_size=100, overlap=20)
    assert len(chunks) == 0


def test_process_pdfs_no_folder():
    """Test processing PDFs when folder doesn't exist."""
    chunks = process_pdfs("./nonexistent_folder")
    # Should return empty list or create folder
    assert isinstance(chunks, list)


@pytest.mark.skipif(not Path("./research_papers").exists(), reason="No research_papers folder")
def test_process_pdfs():
    """Test PDF processing (requires PDFs in research_papers folder)."""
    chunks = process_pdfs("./research_papers")
    assert isinstance(chunks, list)
    # If PDFs exist, should have chunks
    if len(list(Path("./research_papers").glob("*.pdf"))) > 0:
        assert len(chunks) > 0
        assert all("chunk_id" in chunk for chunk in chunks)
        assert all("paper_id" in chunk for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)


def test_create_embeddings():
    """Test embedding creation."""
    chunks = [
        {"text": "This is a test chunk."},
        {"text": "Another test chunk with more words."}
    ]
    embeddings = create_embeddings(chunks, model_name="all-MiniLM-L6-v2")
    
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0  # Should have embedding dimension


def test_build_faiss_index():
    """Test FAISS index building."""
    import numpy as np
    
    # Create dummy embeddings
    embeddings = np.random.rand(10, 384).astype('float32')
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    index = build_faiss_index(embeddings, index_type="flat")
    
    assert index.ntotal == 10
    assert index.d == 384


