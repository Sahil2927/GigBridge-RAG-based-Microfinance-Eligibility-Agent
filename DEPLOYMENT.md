# Deploying ClariFI to Streamlit Community Cloud

This guide takes the project from zero to a live URL on [Streamlit Community Cloud](https://streamlit.io/cloud).

## What gets deployed

| Component | Source at runtime |
|-----------|-----------------|
| **ML model** | Auto-built on first load via `ensure_runtime_artifacts()` (~30–90s) |
| **RAG index** | Demo KB built from embedded research snippets (no PDF upload needed) |
| **Groq LLM** | Your `GROQ_API_KEY` secret |
| **User data** | Ephemeral JSON under `data/` (resets on redeploy) |

> **Note:** Streamlit Cloud disk is ephemeral. The app bootstraps artifacts on cold start unless you commit pre-built files (optional).

---

## Prerequisites

1. GitHub account with this repo pushed to GitHub
2. [Streamlit Community Cloud](https://share.streamlit.io/) account
3. [Groq API key](https://console.groq.com/) for RAG assessments

---

## Step 1 — Push code to GitHub

```bash
git add .
git commit -m "Prepare ClariFI for Streamlit Cloud deployment"
git push origin master
```

Do **not** commit `.env` or `.streamlit/secrets.toml` (they are gitignored).

---

## Step 2 — Local smoke test (recommended)

```powershell
cd E:\CLariFi\ClariFI
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set GROQ_API_KEY

python scripts/bootstrap.py
streamlit run app.py
```

Verify:

- **Assessment → ML Model Only** → Run ML Prediction → SHAP chart appears
- **Assessment → RAG Agent Only** → Run RAG Assessment → eligibility result (needs Groq key)
- **Admin View** → login with `admin123` (or your `ADMIN_PASSWORD`)

---

## Step 3 — Create the Streamlit Cloud app

1. Go to https://share.streamlit.io/
2. Click **Create app**
3. Select your GitHub repo (`ClariFI`)
4. Set **Main file path**: `app.py`
5. Branch: `master` (or your default branch)
6. Click **Advanced settings** if you need Python 3.10+ (default is usually fine)

---

## Step 4 — Configure secrets

In the Streamlit Cloud app: **Settings → Secrets**, paste:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxx"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
ADMIN_PASSWORD = "choose-a-strong-password"
```

Save and **Reboot app**.

Secrets are read by `src/utils/config.py` via `st.secrets`, with env fallback for local dev.

---

## Step 5 — First deploy behaviour

On the **first cold start**, the app shows:

> Setting up ML model and knowledge base (first run ~30–90s)...

This runs:

```text
python scripts/bootstrap.py   # equivalent logic in ensure_runtime_artifacts()
```

- Synthetic `Loan_default.csv` → quick XGBoost train → `models/loan_xgb.pkl`
- Demo FAISS index → `data/faiss_index.bin`

Subsequent requests in the same container reuse cached artifacts via `@st.cache_resource`.

---

## Environment variables (optional)

| Variable | Default | Purpose |
|----------|---------|---------|
| `SKIP_BOOTSTRAP` | `false` | Skip auto-setup (only if you ship pre-built artifacts) |
| `BOOTSTRAP_QUICK` | `true` | Faster training for cloud |
| `GROQ_API_KEY` | — | Override secret locally via `.env` |
| `ADMIN_PASSWORD` | `admin123` | Admin dashboard password |

---

## Using real data instead of demo bootstrap

### ML — real loan CSV

1. Place Kaggle `Loan_default.csv` in `data/`
2. Run locally:

   ```bash
   python scripts/export_model_from_notebook.py --csv-path data/Loan_default.csv
   ```

3. Optionally commit `models/loan_xgb.pkl` for faster cloud starts (update `.gitignore` if needed)

### RAG — research PDFs

1. Add PDFs to `research_papers/`
2. Run locally:

   ```bash
   python run_ingestion.py
   ```

3. Optionally commit `data/faiss_index.bin` + `data/metadata.jsonl`

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| App stuck on bootstrap | Check Cloud logs; try `BOOTSTRAP_QUICK=true` |
| RAG fails | Verify `GROQ_API_KEY` in Secrets |
| ML not loaded | Run `python scripts/bootstrap.py` locally and inspect logs |
| Admin login fails | Match password to `ADMIN_PASSWORD` secret |
| Module not found | Ensure `requirements.txt` is complete and redeploy |

---

## Production checklist

- [ ] Change `ADMIN_PASSWORD` secret
- [ ] Restrict admin page (consider OAuth / Streamlit auth later)
- [ ] Replace synthetic training data with real validated dataset
- [ ] Add real research PDFs for RAG
- [ ] Review ethics copy vs actual data handling
- [ ] Set up monitoring for Groq API usage and errors

---

## Alternative deployments

The same app runs in Docker for Azure/Railway/Render:

```dockerfile
# Example (create Dockerfile if needed)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python scripts/bootstrap.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

For Streamlit Cloud you do **not** need Docker.
