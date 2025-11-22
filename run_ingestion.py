#!/usr/bin/env python3
"""
Simple script to run knowledge base ingestion.

Usage:
    python run_ingestion.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.kb.ingest import ingest_knowledge_base
import os

if __name__ == "__main__":
    print("Starting knowledge base ingestion...")
    print(f"Looking for PDFs in: {Path('research_papers').absolute()}")
    
    # Check if research_papers folder exists
    if not Path("research_papers").exists():
        print("Creating research_papers folder...")
        os.makedirs("research_papers", exist_ok=True)
        print("⚠️  Please add PDF files to the research_papers/ folder and run again.")
        sys.exit(1)
    
    # Count PDFs
    pdf_files = list(Path("research_papers").glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    if len(pdf_files) == 0:
        print("⚠️  No PDF files found in research_papers/ folder!")
        print("Please add PDF files and run again.")
        sys.exit(1)
    
    # Run ingestion
    try:
        ingest_knowledge_base(
            pdf_folder="./research_papers",
            index_path="./data/faiss_index.bin",
            metadata_path="./data/metadata.jsonl",
            model_name="all-MiniLM-L6-v2",
            index_type="flat"
        )
        print("\n✅ Ingestion completed successfully!")
        print(f"Index saved to: data/faiss_index.bin")
        print(f"Metadata saved to: data/metadata.jsonl")
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

