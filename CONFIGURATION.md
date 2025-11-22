# Configuration Guide: Groq API Key and LLM Model

## Quick Setup

### Option 1: Environment Variables (Recommended)

Set these environment variables before running:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your-groq-api-key-here"
$env:GROQ_MODEL_NAME="mixtral-8x7b-32768"

# Windows (CMD)
set GROQ_API_KEY=your-groq-api-key-here
set GROQ_MODEL_NAME=mixtral-8x7b-32768

# Linux/Mac
export GROQ_API_KEY='your-groq-api-key-here'
export GROQ_MODEL_NAME='mixtral-8x7b-32768'
```

### Option 2: Create `.env` file (Recommended for local development)

1. Copy the example file:
   ```bash
   # Windows
   copy .env.example .env
   
   # Linux/Mac
   cp .env.example .env
   ```

2. Edit `.env` and add your actual values:
   ```
   GROQ_API_KEY=your-groq-api-key-here
   GROQ_MODEL_NAME=mixtral-8x7b-32768
   ```

**Note:** The `.env` file is gitignored and won't be committed to version control.

## Configuration Locations

### 1. **Streamlit App (`app.py`)**

**API Key:**
- Reads from environment variable: `GROQ_API_KEY`
- Or from Streamlit secrets: `st.secrets.get("GROQ_API_KEY")`
- **Location:** Line 276

**Model Name:**
- Reads from environment variable: `GROQ_MODEL_NAME` (defaults to "mixtral-8x7b-32768")
- **Location:** Line 278
- **Available models:**
  - `"mixtral-8x7b-32768"` (default, recommended)
  - `"llama3-70b-8192"`
  - `"gemma-7b-it"`
  - `"llama-3.1-8b-instant"`

### 2. **Notebooks**

#### `notebooks/demo_end2end.ipynb`

**API Key:**
- **Location:** Cell 2 (Setup cell), Line 43
- Change: `os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"`

**Model Name:**
- **Location:** Cell 4 (RAG Assessment cell), Line 147
- Change: `model_name="mixtral-8x7b-32768"` to your preferred model

#### `notebooks/ingest_kb.ipynb`

- **No Groq API key needed** (only uses embeddings, not LLM)

### 3. **Programmatic Usage**

When using the Python API directly:

```python
from src.agent.rag_agent import RAGAgent
import os

# Set API key
os.environ["GROQ_API_KEY"] = "your-key-here"

# Create agent with specific model
agent = RAGAgent(
    groq_api_key=os.getenv("GROQ_API_KEY"),  # or pass directly
    model_name="mixtral-8x7b-32768"  # Change this
)
```

## Available Groq Models

| Model Name | Description | Context Window |
|------------|-------------|----------------|
| `mixtral-8x7b-32768` | Mixtral 8x7B (default) | 32,768 tokens |
| `llama3-70b-8192` | Llama 3 70B | 8,192 tokens |
| `llama-3.1-8b-instant` | Llama 3.1 8B Instant | Fast, smaller |
| `gemma-7b-it` | Gemma 7B Instruct | 8,192 tokens |

## Getting Your Groq API Key

1. Go to https://console.groq.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and set it as `GROQ_API_KEY`

## Verification

To verify your configuration is working:

```python
import os
from src.agent.rag_agent import RAGAgent

# Check API key
api_key = os.getenv("GROQ_API_KEY")
if api_key:
    print("✅ API key found")
    # Test agent creation
    try:
        agent = RAGAgent(groq_api_key=api_key, model_name="mixtral-8x7b-32768")
        print("✅ Agent created successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ API key not found")
```

## Summary

**For Streamlit App:**
- Set `GROQ_API_KEY` environment variable
- Set `GROQ_MODEL_NAME` environment variable (optional, defaults to mixtral-8x7b-32768)

**For Notebooks:**
- Edit the setup cell in `notebooks/demo_end2end.ipynb` (line 43 for API key, line 147 for model)

**For Scripts:**
- Set environment variables or pass directly to `RAGAgent` constructor

