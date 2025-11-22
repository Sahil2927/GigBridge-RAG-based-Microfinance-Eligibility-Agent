"""
Knowledge Base Ingestion Module

This module handles:
- PDF text extraction from research papers
- Text chunking with overlap
- Embedding generation using sentence-transformers
- FAISS index creation and persistence
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber

# Fix TensorFlow DLL issue - set before importing sentence_transformers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using pdfplumber.
    Falls back to basic extraction if pdfplumber fails.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logger.info(f"Successfully extracted text from {pdf_path}")
    except Exception as e:
        logger.warning(f"pdfplumber failed for {pdf_path}: {e}. Trying fallback...")
        try:
            # Fallback: try pdfminer.six if available
            from pdfminer.high_level import extract_text as pdfminer_extract
            text = pdfminer_extract(pdf_path)
            logger.info(f"Fallback extraction successful for {pdf_path}")
        except Exception as e2:
            logger.error(f"All extraction methods failed for {pdf_path}: {e2}")
            text = ""
    return text


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[Dict[str, Any]]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in words
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of chunk dictionaries with 'text' and 'chunk_index' keys
    """
    words = text.split()
    chunks = []
    chunk_index = 0
    
    i = 0
    while i < len(words):
        # Take chunk_size words
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_index
            })
            chunk_index += 1
        
        # Move forward by (chunk_size - overlap) words
        i += chunk_size - overlap
        
        # Prevent infinite loop
        if i >= len(words):
            break
    
    logger.info(f"Created {len(chunks)} chunks from text")
    return chunks


def process_pdfs(pdf_folder: str = "./research_papers") -> List[Dict[str, Any]]:
    """
    Process all PDFs in the specified folder.
    
    Args:
        pdf_folder: Path to folder containing PDF files
        
    Returns:
        List of dictionaries, each containing:
        - chunk_id: Unique identifier
        - paper_id: PDF filename
        - paper_title: PDF filename (without extension)
        - chunk_index: Index within the paper
        - text: Chunk text
    """
    pdf_folder = Path(pdf_folder)
    if not pdf_folder.exists():
        logger.warning(f"PDF folder {pdf_folder} does not exist. Creating it.")
        pdf_folder.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_folder}")
        return []
    
    all_chunks = []
    chunk_id_counter = 0
    
    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path.name}...")
        text = extract_text_from_pdf(str(pdf_path))
        
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path.name}")
            continue
        
        chunks = chunk_text(text, chunk_size=400, overlap=80)
        paper_title = pdf_path.stem  # filename without extension
        
        for chunk in chunks:
            all_chunks.append({
                "chunk_id": f"chunk_{chunk_id_counter}",
                "paper_id": pdf_path.name,
                "paper_title": paper_title,
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"]
            })
            chunk_id_counter += 1
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def create_embeddings(chunks: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate embeddings for all chunks using sentence-transformers.
    
    Args:
        chunks: List of chunk dictionaries
        model_name: Name of the sentence-transformer model
        
    Returns:
        Numpy array of embeddings (n_chunks, embedding_dim)
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    texts = [chunk["text"] for chunk in chunks]
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """
    Build a FAISS index from embeddings.
    
    Args:
        embeddings: Numpy array of normalized embeddings
        index_type: Type of index ("flat" for small datasets, "hnsw" for larger)
        
    Returns:
        FAISS index object
    """
    dimension = embeddings.shape[1]
    
    if index_type == "flat":
        # IndexFlatIP for inner product (since vectors are normalized)
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "hnsw":
        # HNSW for larger datasets
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 200
    else:
        raise ValueError(f"Unknown index_type: {index_type}")
    
    # Add embeddings to index
    logger.info(f"Adding {len(embeddings)} embeddings to FAISS index...")
    index.add(embeddings.astype('float32'))
    
    logger.info(f"FAISS index built with {index.ntotal} vectors")
    return index


def save_index_and_metadata(index: faiss.Index, chunks: List[Dict[str, Any]], 
                           index_path: str = "./data/faiss_index.bin",
                           metadata_path: str = "./data/metadata.jsonl"):
    """
    Save FAISS index and chunk metadata to disk.
    
    Args:
        index: FAISS index object
        chunks: List of chunk dictionaries
        index_path: Path to save FAISS index
        metadata_path: Path to save metadata (JSONL format)
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    # Save FAISS index
    logger.info(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    # Save metadata as JSONL (one JSON object per line)
    logger.info(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    logger.info("Index and metadata saved successfully")


def load_index_and_metadata(index_path: str = "./data/faiss_index.bin",
                           metadata_path: str = "./data/metadata.jsonl") -> tuple:
    """
    Load FAISS index and metadata from disk.
    
    Args:
        index_path: Path to FAISS index file
        metadata_path: Path to metadata JSONL file
        
    Returns:
        Tuple of (FAISS index, list of chunk dictionaries)
    """
    logger.info(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    
    logger.info(f"Loading metadata from {metadata_path}...")
    chunks = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    logger.info(f"Loaded {len(chunks)} chunks from metadata")
    return index, chunks


def ingest_knowledge_base(pdf_folder: str = "./research_papers",
                         index_path: str = "./data/faiss_index.bin",
                         metadata_path: str = "./data/metadata.jsonl",
                         model_name: str = "all-MiniLM-L6-v2",
                         index_type: str = "flat"):
    """
    Complete knowledge base ingestion pipeline.
    
    Args:
        pdf_folder: Folder containing PDF files
        index_path: Path to save FAISS index
        metadata_path: Path to save metadata
        model_name: Sentence-transformer model name
        index_type: FAISS index type ("flat" or "hnsw")
    """
    logger.info("Starting knowledge base ingestion...")
    
    # Step 1: Process PDFs
    chunks = process_pdfs(pdf_folder)
    if not chunks:
        logger.error("No chunks created. Cannot proceed.")
        return
    
    # Step 2: Create embeddings
    embeddings = create_embeddings(chunks, model_name=model_name)
    
    # Step 3: Build FAISS index
    index = build_faiss_index(embeddings, index_type=index_type)
    
    # Step 4: Save index and metadata
    save_index_and_metadata(index, chunks, index_path, metadata_path)
    
    logger.info("Knowledge base ingestion completed successfully!")


if __name__ == "__main__":
    # Run ingestion if called directly
    ingest_knowledge_base()

