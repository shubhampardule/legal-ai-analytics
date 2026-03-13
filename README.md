# Legal AI Analytics System

Explainable legal case analytics web app for:

- case outcome prediction
- local explanation of the prediction
- similar case retrieval
- interactive dashboard (React)
- “Chat with this Case” (beta, retrieval-grounded)

> ⚠️ **Research prototype only** — not legal advice.

---

## Table of Contents

1. [What this project does](#what-this-project-does)
2. [Important note about GitHub clone](#important-note-about-github-clone)
3. [Tech stack](#tech-stack)
4. [Tested environment](#tested-environment)
5. [Project structure](#project-structure)
6. [Option A: Run with prebuilt data/artifacts](#option-a-run-with-prebuilt-dataartifacts)
7. [Option B: Full rebuild from raw dataset (no DB/models)](#option-b-full-rebuild-from-raw-dataset-no-dbmodels)
8. [Run backend + frontend](#run-backend--frontend)
9. [Verify everything works](#verify-everything-works)
10. [Environment variables](#environment-variables)
11. [Common issues and fixes](#common-issues-and-fixes)
12. [Main API routes](#main-api-routes)
13. [What not to commit](#what-not-to-commit)

---

## What this project does

- Predicts case outcome (`accepted` / `rejected`)
- Shows prediction confidence and summary
- Explains output with top terms + evidence sentences
- Retrieves similar cases using MiniLM + FAISS
- Supports filtering similar cases by outcome/year
- Extracts legal entities (statutes/judges)
- Provides “Chat with this Case” (beta)

---

## Important note about GitHub clone

Because dataset and model artifacts are large and/or gated, this repository typically **does not include**:

- `data/raw/ildc/*`
- `data/processed/ildc/*`
- `artifacts/*`

That means after cloning, you must do one of these:

### Option A (fastest)
Get a prepared data+artifacts bundle from project owner and extract it in repo root.

### Option B (fully reproducible)
Download gated raw ILDC files and run all build scripts locally.

This README gives complete commands for both options.

---

## Tech stack

### Backend
- FastAPI
- Uvicorn
- Pydantic

### ML / NLP
- TF-IDF + Logistic Regression (prediction)
- Sentence Transformers (MiniLM)
- FAISS (similarity index)
- NumPy, Pandas, PyArrow, scikit-learn

### Frontend
- React + Vite
- Tailwind CSS
- Chart.js

---

## Tested environment

- Python `3.13`
- Node.js `22`
- npm `10`
- Windows PowerShell (primary tested shell)

---

## Project structure

```text
backend/                 FastAPI app
frontend/                React dashboard
scripts/                 data prep, model training, utility scripts
data/                    raw + processed dataset files
artifacts/               trained models, embeddings, FAISS index, reports
requirements.txt         Python dependencies
improvement.md           roadmap + feature status
PROJECT_REPORT.md        project report
```

---

## Option A: Run with prebuilt data/artifacts

Use this path if someone shares a zip containing already generated `data/processed` and `artifacts`.

## 1) Clone

```bash
git clone <your-repo-url>
cd Legal
```

## 2) Create Python venv

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3) Install backend dependencies

```bash
pip install -r requirements.txt
```

## 4) Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## 5) Extract prepared bundle

Extract shared bundle so these paths exist:

- `data/processed/ildc/train.parquet`
- `data/processed/ildc/dev.parquet`
- `data/processed/ildc/test.parquet`
- `data/splits/ildc/*.txt`
- `artifacts/baseline/tfidf_logreg/model.joblib`
- `artifacts/baseline/tfidf_logreg/vectorizer.joblib`
- `artifacts/retrieval_case_embeddings/case_embeddings.npy`
- `artifacts/retrieval_case_embeddings/case_metadata.parquet`
- `artifacts/similarity_index/ildc_cases_ip.index`

Then go to [Run backend + frontend](#run-backend--frontend).

---

## Option B: Full rebuild from raw dataset (no DB/models)

Use this when cloned repo has code only.

## 1) Get ILDC dataset access

Request access here:

- https://huggingface.co/datasets/Exploration-Lab/IL-TUR

After access, place raw files inside `data/raw/ildc/`:

- `multi_train-00000-of-00002.parquet`
- `multi_train-00001-of-00002.parquet`
- `multi_dev-00000-of-00001.parquet`
- `test-00000-of-00001.parquet`

Your raw directory should look like:

```text
data/raw/ildc/
  multi_train-00000-of-00002.parquet
  multi_train-00001-of-00002.parquet
  multi_dev-00000-of-00001.parquet
  test-00000-of-00001.parquet
```

## 2) Environment and dependency install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

## 3) Run full data/model pipeline (exact order)

Run from repo root:

```powershell
python scripts/preprocess_ildc.py
python scripts/save_ildc_splits.py
python scripts/train_baseline_tfidf_logreg.py
python scripts/train_advanced_minilm_embedding.py
python scripts/build_case_embedding_corpus.py
python scripts/build_similarity_index.py
```

### Notes

- `train_advanced_minilm_embedding.py` is the slowest step (GPU highly recommended).
- First MiniLM run may download model files if not cached.
- Prediction endpoint uses baseline TF-IDF model.
- Similarity search uses MiniLM embeddings + FAISS index.

## 4) Quick artifact sanity check (manual)

Confirm these files were created:

```text
artifacts/baseline/tfidf_logreg/model.joblib
artifacts/baseline/tfidf_logreg/vectorizer.joblib
artifacts/retrieval_case_embeddings/case_embeddings.npy
artifacts/retrieval_case_embeddings/case_metadata.parquet
artifacts/similarity_index/ildc_cases_ip.index
```

---

## Run backend + frontend

Open two terminals.

## Terminal 1 — Backend

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## Terminal 2 — Frontend

```powershell
cd frontend
npm run dev
```

Open the URL shown by Vite (usually `http://127.0.0.1:5173` or next free port).

---

## Verify everything works

## A) API health check

Visit:

- `http://127.0.0.1:8000/health`

Should return `status: ok`.

## B) In-app test

Use sample case ID in UI:

- `1980_211`

Click **Run Analysis** and verify:

- prediction card appears,
- explanation appears,
- similar cases list appears.

## C) Full stack smoke test script

```powershell
python scripts/full_stack_smoke_test.py
```

Expected report output:

- `artifacts/integration/full_stack_smoke_test.json`

---

## Environment variables

## Backend

Optional variables:

- `ALLOWED_ORIGINS` (comma-separated)
- `DEBUG` (`true/false`)
- `ANALYSIS_CACHE_MAX_SIZE`
- `ANALYSIS_CACHE_TTL_SECONDS`

## Frontend

If you want direct API URL instead of proxy, create `frontend/.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## Common issues and fixes

## 1) Case ID gives no result / not found

- Invalid case IDs should show user-friendly error in UI.
- Try known IDs like `1980_211`, `1973_261`, `1983_326`.

## 2) `ModuleNotFoundError` when running scripts

- Ensure venv is activated.
- Run commands from project root unless script says otherwise.

## 3) Backend starts but analyze fails

Usually missing artifacts. Recheck:

- baseline model/vectorizer files
- retrieval embeddings + metadata
- FAISS index file

## 4) Frontend can’t call backend

- Check backend running on `127.0.0.1:8000`
- Check Vite port and proxy
- If using `.env`, verify `VITE_API_BASE_URL`

## 5) MiniLM download or model load issues

- Ensure internet for first model download
- Retry once model cache is created

## 6) Very slow build/training

- Advanced embedding training is heavy on CPU
- Prefer GPU for `train_advanced_minilm_embedding.py`

---

## Main API routes

- `GET /health`
- `GET /api/v1/meta`
- `POST /api/v1/predict`
- `POST /api/v1/explain`
- `POST /api/v1/similar-cases`
- `POST /api/v1/analyze`
- `POST /api/v1/chat-case` *(beta)*
- `GET /api/v1/cases`
- `GET /api/v1/cases/{case_id}`

---

## What not to commit

Keep these out of Git:

- `frontend/node_modules/`
- `frontend/dist/`
- `data/`
- `artifacts/`
- `.env`
- `hftoken`
- `__pycache__/`

Reason:
- large size,
- gated dataset constraints,
- generated files make repo heavy.

---

## Collaboration recommendation (important)

Since DB/models are not uploaded to GitHub, for teammates:

1. Share **code via GitHub**.
2. Share **prepared `data/processed` + `artifacts`** as release asset / drive zip.
3. Mention extraction path in README (this file).
4. Keep rebuild path documented (Option B) for full reproducibility.

---

## License and dataset terms

- This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE).
- Follow ILDC/CJPE dataset usage terms from source provider.

---

## Author

- GitHub: [@shubhampardule](https://github.com/shubhampardule)
