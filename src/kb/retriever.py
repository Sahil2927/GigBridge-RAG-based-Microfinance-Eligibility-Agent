"""
Knowledge Base Retriever Module

This module handles retrieval of relevant chunks from the FAISS index
based on user queries.
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple

# Fix TensorFlow DLL issue - set before importing sentence_transformers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkRetriever:
    """
    Retrieves relevant chunks from the knowledge base using FAISS.
    """
    
    def __init__(self, index_path: str = "./data/faiss_index.bin",
                 metadata_path: str = "./data/metadata.jsonl",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retriever.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSONL file
            model_name: Sentence-transformer model name
        """
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Load index and metadata
        self._load_index_and_metadata()
    
    def _load_index_and_metadata(self):
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}. Run ingestion first.")
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}. Run ingestion first.")
        
        logger.info(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        
        logger.info(f"Loading metadata from {self.metadata_path}...")
        self.chunks = []
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.chunks)} chunks and index with {self.index.ntotal} vectors")
    
    def retrieve_chunks(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: User query string
            k: Number of chunks to retrieve
            
        Returns:
            List of chunk dictionaries with added 'score' field
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype('float32')
        
        # Search in FAISS index
        # For IndexFlatIP, higher scores mean more similar
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Retrieve chunks
        retrieved_chunks = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(score)
                chunk['rank'] = i + 1
                retrieved_chunks.append(chunk)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: '{query[:50]}...'")
        return retrieved_chunks
    
    def retrieve_chunks_with_threshold(self, query: str, k: int = 6, 
                                       min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with a minimum similarity score threshold.
        
        Args:
            query: User query string
            k: Maximum number of chunks to retrieve
            min_score: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of chunk dictionaries with scores above threshold
        """
        chunks = self.retrieve_chunks(query, k=k)
        filtered_chunks = [chunk for chunk in chunks if chunk['score'] >= min_score]
        
        logger.info(f"Filtered to {len(filtered_chunks)} chunks above threshold {min_score}")
        return filtered_chunks


# Global retriever instance (lazy loading)
_retriever_instance = None


def get_retriever(index_path: str = "./data/faiss_index.bin",
                 metadata_path: str = "./data/metadata.jsonl",
                 model_name: str = "all-MiniLM-L6-v2") -> ChunkRetriever:
    """
    Get or create a global retriever instance.
    
    Args:
        index_path: Path to FAISS index
        metadata_path: Path to metadata file
        model_name: Embedding model name
        
    Returns:
        ChunkRetriever instance
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = ChunkRetriever(index_path, metadata_path, model_name)
    return _retriever_instance


def retrieve_chunks(query: str, k: int = 6, 
                   index_path: str = "./data/faiss_index.bin",
                   metadata_path: str = "./data/metadata.jsonl",
                   model_name: str = "all-MiniLM-L6-v2") -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve chunks.
    
    Args:
        query: User query string
        k: Number of chunks to retrieve
        index_path: Path to FAISS index
        metadata_path: Path to metadata file
        model_name: Embedding model name
        
    Returns:
        List of retrieved chunk dictionaries
    """
    retriever = get_retriever(index_path, metadata_path, model_name)
    return retriever.retrieve_chunks(query, k=k)

