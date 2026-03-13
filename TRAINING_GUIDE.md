# Team Training & Rebuild Playbook

This is the teammate-facing quick guide for model build/rebuild in this repo.

Current setup:

- **Prediction + Explanation:** `MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33`
- **Similarity Retrieval:** `sentence-transformers/all-MiniLM-L6-v2` + FAISS

---

## What we train vs what we don’t

### DeBERTa (prediction)
No training required in our current flow.

- We use it as a pretrained zero-shot model.
- First run downloads/caches weights.

### MiniLM + FAISS (similar cases)
Needs artifact build/rebuild when starting fresh.

---

## Minimal setup we usually run

Run from repo root.

### 1) Python env

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Build similarity artifacts

```powershell
python scripts/preprocess_ildc.py
python scripts/save_ildc_splits.py
python scripts/train_advanced_minilm_embedding.py
python scripts/build_case_embedding_corpus.py
python scripts/build_similarity_index.py
```

This produces the data used by `/similar-cases` and similarity sections in `/analyze`.

---

## Full reproducible pipeline (from raw ILDC)

Use this for clean machine setup / full rebuild.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/preprocess_ildc.py
python scripts/save_ildc_splits.py
python scripts/train_advanced_minilm_embedding.py
python scripts/build_case_embedding_corpus.py
python scripts/build_similarity_index.py
```

---

## Optional legacy baseline (for experiments only)

Not part of default runtime path.

```powershell
python scripts/train_baseline_tfidf_logreg.py
```

---

## Run app after build

### Backend

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

---

## Quick verification for teammates

- Health: `http://127.0.0.1:8000/health`
- Analyze test: use case ID `1980_211`
- Expected: prediction, explanation, and similar cases all render

---

## First-run delays (expected)

- DeBERTa first run: download + warmup (one-time)
- MiniLM embedding generation: slowest offline step
- FAISS build: relatively quick once embeddings are ready

---

## Common issues and quick fixes

### `ModuleNotFoundError`

- Ensure `.venv` is active
- Re-run `pip install -r requirements.txt`

### Similar-case results missing/failing

Rebuild in this exact order:

1. `python scripts/train_advanced_minilm_embedding.py`
2. `python scripts/build_case_embedding_corpus.py`
3. `python scripts/build_similarity_index.py`

### First prediction call is slow

Normal behavior during DeBERTa model load. Subsequent calls should be faster.
