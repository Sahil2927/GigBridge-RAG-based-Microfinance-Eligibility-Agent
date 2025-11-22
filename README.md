# RAG-based Microfinance Eligibility Agent

A complete RAG (Retrieval-Augmented Generation) system for assessing microfinance eligibility using unconventional data sources (mobile metadata, psychometrics, financial behavior, social networks) and research evidence.

## 🎯 Overview

This system:
- Ingests research PDFs to build a knowledge base
- Accepts user profiles with unconventional data (mobile metadata, psychometrics, etc.)
- Uses RAG to assess eligibility based on research evidence
- Implements ethical-by-design consent mechanisms
- Provides a Streamlit UI for user interaction
- Stores consented submissions in a bank database (sandbox)

## 📋 Features

- **Knowledge Base Ingestion**: Extract, chunk, and embed research PDFs using sentence-transformers and FAISS
- **RAG-based Assessment**: Use Groq LLM (Mixtral/Llama3/Gemma) to make eligibility decisions based on retrieved research
- **Profile Building**: Convert raw unconventional data into structured profiles
- **Consent Management**: Explicit consent mechanism with data deletion on denial
- **Admin View**: View consented submissions (password-protected)
- **Privacy-First**: Only aggregated features shared, raw data can be deleted

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API key (get from https://console.groq.com/)
- Research PDFs in `research_papers/` folder

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   
   **Option A: Create a `.env` file (Recommended for local development):**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Then edit .env and add your API key
   GROQ_API_KEY=your-groq-api-key-here
   GROQ_MODEL_NAME=mixtral-8x7b-32768
   ```
   
   **Option B: Set environment variables directly:**
   ```bash
   # Windows (PowerShell)
   $env:GROQ_API_KEY='your-groq-api-key-here'
   $env:GROQ_MODEL_NAME='mixtral-8x7b-32768'
   
   # Linux/Mac
   export GROQ_API_KEY='your-groq-api-key-here'
   export GROQ_MODEL_NAME='mixtral-8x7b-32768'
   ```

4. **Upload research PDFs:**
   - Place your research PDFs in the `research_papers/` folder
   - The system will process all `.pdf` files in this folder
   - If you have PDFs in the root directory, run:
     ```bash
     python move_pdfs_to_research_folder.py
     ```

## 📚 Usage

### Option 1: Google Colab (Recommended)

#### Step 1: Upload Project to Colab

1. Upload the entire project folder to Google Colab
2. Or clone from GitHub if hosted

#### Step 2: Run Ingestion Notebook

1. Open `notebooks/ingest_kb.ipynb`
2. Run all cells in order:
   - Install packages
   - Upload PDFs to `research_papers/` folder
   - Run ingestion
   - Verify ingestion
   - Test retrieval

#### Step 3: Run Demo Notebook

1. Open `notebooks/demo_end2end.ipynb`
2. Set your `GROQ_API_KEY` in the setup cell
3. Run all cells to see the end-to-end flow

#### Step 4: Run Streamlit App (Optional)

To run Streamlit in Colab, use ngrok:

```python
# In a Colab cell
!pip install pyngrok
from pyngrok import ngrok

# Start Streamlit
!streamlit run app.py --server.port 8501 &

# Create tunnel
public_url = ngrok.connect(8501)
print(f"Streamlit app available at: {public_url}")
```

### Option 2: Local Installation

#### Step 1: Ingest Knowledge Base

```bash
# Run ingestion script
python -m src.kb.ingest

# Or use the notebook
jupyter notebook notebooks/ingest_kb.ipynb
```

This will:
- Extract text from all PDFs in `research_papers/`
- Chunk text (400 words, 80-word overlap)
- Generate embeddings using `all-MiniLM-L6-v2`
- Build FAISS index
- Save to `data/faiss_index.bin` and `data/metadata.jsonl`

#### Step 2: Run Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

#### Step 3: Use the App

1. **Assessment Page:**
   - Enter user ID (or generate automatically)
   - Fill in demographics
   - Upload or enter unconventional data (mobile, psychometrics, financial, social, loan history)
   - Click "Run Assessment"
   - Review results
   - Give or decline consent

2. **Admin View:**
   - Password: `admin123` (change in production!)
   - View all consented submissions

## 📁 Project Structure

```
GigBridge/
├── src/
│   ├── kb/
│   │   ├── ingest.py          # PDF ingestion, chunking, embedding, FAISS
│   │   └── retriever.py        # Chunk retrieval from FAISS
│   ├── agent/
│   │   └── rag_agent.py        # RAG agent with Groq LLM
│   ├── profile/
│   │   └── profile_builder.py # Profile building from raw data
│   ├── memory/
│   │   └── store.py            # Memory and consent management
│   └── utils/
│       └── validators.py       # Schema validation
├── notebooks/
│   ├── ingest_kb.ipynb         # Knowledge base ingestion notebook
│   └── demo_end2end.ipynb      # End-to-end demo notebook
├── tests/
│   ├── test_ingestion.py
│   ├── test_retriever.py
│   ├── test_profile_builder.py
│   ├── test_memory.py
│   └── test_validators.py
├── example_inputs/
│   ├── likely_eligible_gig_worker.json
│   ├── borderline_user.json
│   └── not_eligible_user.json
├── research_papers/            # Place PDFs here
├── data/
│   ├── faiss_index.bin         # Generated FAISS index
│   ├── metadata.jsonl          # Generated chunk metadata
│   ├── memory.json             # User memory (created at runtime)
│   ├── consented_submissions/ # Consented submissions (created at runtime)
│   └── bank_unstructured_db.jsonl  # Bank database (created at runtime)
├── logs/
│   └── actions.log             # Audit log (created at runtime)
├── app.py                      # Streamlit app
├── requirements.txt
└── README.md
```

## 📊 Profile Schema

User profiles follow this exact schema:

```json
{
  "user_id": "string",
  "timestamp": "ISO8601",
  "demographics": {
    "age": int,
    "gender": "string",
    "occupation": "string",
    "monthly_income": float
  },
  "mobile_metadata": {
    "avg_daily_calls": float | "NA",
    "unique_contacts_30d": int | "NA",
    "days_inactive_last_30": int | "NA",
    "airtime_topup_frequency": float | "NA",
    "avg_topup_amount": float | "NA",
    "data_usage_variance": float | "NA"
  },
  "psychometrics": {
    "C_q1": int, "C_q2": int, ... or "conscientiousness_score": float | "NA"
  },
  "financial_behavior": {
    "savings_frequency": float | "NA",
    "savings_amount_variance": float | "NA",
    "wallet_balance_lows_last_90d": int | "NA",
    "bill_payment_timeliness": float | "NA"
  },
  "social_network": {
    "shg_membership": bool | "NA",
    "peer_monitoring_strength": float | "NA"
  },
  "loan_history": {
    "previous_loans": int | "NA",
    "previous_defaults": int | "NA",
    "previous_late_payments": int | "NA",
    "avg_repayment_delay_days": float | "NA"
  },
  "_raw_unconventional": { ... }  // Raw uploaded logs (for audit)
}
```

## 🔍 RAG Output Schema

The RAG agent returns:

```json
{
  "eligibility": "yes" | "no" | "maybe",
  "risk_score": 0.0-1.0,
  "verdict_text": "Brief explanation",
  "strong_points": ["point 1", "point 2", "point 3"],
  "weak_points": ["point 1", "point 2", "point 3"],
  "required_unconventional_data": ["item1", "item2"],
  "actionable_recommendations": ["rec 1", "rec 2", "rec 3", "rec 4"],
  "confidence": "high" | "medium" | "low",
  "raw_internal_reasoning": "Internal reasoning with CHUNK references"
}
```

## 🧪 Testing

Run unit tests:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_ingestion.py

# Run with verbose output
pytest tests/ -v
```

**Note:** Some tests require the FAISS index to be built first (run ingestion).

## 🔐 Privacy & Ethics

### Ethical Scoring Principles

1. **Transparency**: Users see exactly what data is shared
2. **Consent**: Explicit consent required before data sharing
3. **Deletion**: Users can decline consent and all data is deleted
4. **Aggregation**: Only aggregated features shared, not raw logs
5. **Audit Trail**: All actions logged in `logs/actions.log`

### Redress Mechanism

If a user disputes an eligibility decision:

1. **Manual Review**: Admin can review the decision and underlying data
2. **Appeal Process**: User can request re-assessment with additional data
3. **Data Correction**: Users can update their profile and re-run assessment
4. **Human Override**: Admins can manually adjust decisions (implement in production)

### Privacy Statement

- **Data Storage**: Profiles stored locally until consent given
- **Data Sharing**: Only aggregated features shared with bank (not raw logs)
- **Data Deletion**: All data deleted if consent denied
- **Audit Logs**: All actions logged for compliance
- **No External Sharing**: Data never shared outside the system without consent

## 🛠️ Configuration

### Groq Models

Supported models (set in `RAGAgent` or `app.py`):
- `mixtral-8x7b-32768` (default, recommended)
- `llama3-70b-8192`
- `gemma-7b-it`

### FAISS Index Type

For small datasets (<10k chunks): `"flat"` (default)
For larger datasets: `"hnsw"` (set in `ingest_knowledge_base()`)

### Embedding Model

Default: `all-MiniLM-L6-v2` (384 dimensions, fast, good quality)

Alternatives:
- `all-mpnet-base-v2` (768 dimensions, better quality, slower)
- `paraphrase-multilingual-MiniLM-L12-v2` (for multilingual)

## 📝 Example Usage

### Example 1: Using Python API

```python
from src.profile.profile_builder import build_profile
from src.agent.rag_agent import RAGAgent
from src.memory.store import save_consented_submission
import os

# Set API key
os.environ["GROQ_API_KEY"] = "your-key"

# Build profile
profile = build_profile(
    user_id="user_001",
    demographics={"age": 30, "gender": "female", "occupation": "gig_worker", "monthly_income": 20000},
    raw_unconventional_data={
        "call_logs_30d": [...],
        "psychometric_responses": {...},
        # ... other data
    }
)

# Run assessment
agent = RAGAgent(groq_api_key=os.getenv("GROQ_API_KEY"))
decision = agent.assess_eligibility(profile)

# Handle consent
if user_consents:
    save_consented_submission(profile["user_id"], profile, decision)
```

### Example 2: Using Example Profiles

```python
import json
from src.profile.profile_builder import build_profile_from_json
from src.agent.rag_agent import RAGAgent

# Load example profile
with open("example_inputs/likely_eligible_gig_worker.json") as f:
    data = json.load(f)

# Build profile
profile = build_profile_from_json(data)

# Assess
agent = RAGAgent()
decision = agent.assess_eligibility(profile)

print(f"Eligibility: {decision['eligibility']}")
print(f"Risk Score: {decision['risk_score']}")
```

## 🐛 Troubleshooting

### Issue: "FAISS index not found"

**Solution:** Run ingestion first:
```bash
python -m src.kb.ingest
# Or use the notebook: notebooks/ingest_kb.ipynb
```

### Issue: "GROQ_API_KEY not found"

**Solution:** Set the environment variable:
```bash
export GROQ_API_KEY='your-key'
# Or in Python:
import os
os.environ["GROQ_API_KEY"] = "your-key"
```

### Issue: "No PDFs found"

**Solution:** Ensure PDFs are in `research_papers/` folder:
```bash
ls research_papers/*.pdf
```

### Issue: "LLM response parsing failed"

**Solution:** The LLM may have returned non-JSON text. The system has fallback handling, but you can:
1. Try a different Groq model
2. Adjust temperature in `rag_agent.py` (lower = more deterministic)
3. Check `logs/actions.log` for details

## 📄 License

This project is provided as-is for research and educational purposes.

## 🙏 Acknowledgments

- Research papers in `research_papers/` folder
- Groq for LLM API
- Sentence-transformers for embeddings
- FAISS for vector search
- Streamlit for UI

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the test files for usage examples
3. Check `logs/actions.log` for error details

## 🔄 Future Enhancements

- [ ] Multi-language support
- [ ] Calibration utility for risk scores
- [ ] Advanced admin dashboard
- [ ] API endpoints for integration
- [ ] Batch processing support
- [ ] Model fine-tuning on labeled data

---

**Built with ❤️ for ethical AI in microfinance**

