# Redrob Ranker — Sandbox Demo

A hosted, runnable demo of the ranker on a small candidate sample (≤100), as required by the
challenge (Section 10.5). It runs the **same** scoring/reasoning logic as `rank.py`; embeddings are
computed live here (fine for ≤100 candidates, ~1s) instead of from the precomputed full-pool matrix.

## Run locally
```bash
pip install -r sandbox/requirements.txt
streamlit run sandbox/app.py
```
Then upload a `.jsonl`/`.json` of ≤100 candidate records (same schema as `candidates.jsonl`), or use
the bundled 50-candidate sample.

## Deploy (any one)
- **HuggingFace Spaces** (Streamlit SDK): create a Space, add `sandbox/app.py` as `app.py`,
  `sandbox/requirements.txt`, `sandbox/sample_candidates.jsonl`, and the `redrob_ranker/` package +
  `precompute.py` + `rank.py` at the repo root. Free CPU tier is sufficient.
- **Streamlit Community Cloud**: point it at this repo, main file `sandbox/app.py`.
- **Docker**: `pip install -r sandbox/requirements.txt && streamlit run sandbox/app.py --server.port 8501`.

## What it shows
Ranked table (rank, score, title, company, country, YoE, PES/EMB/RoleGate/Availability, honeypot flag,
and the plain-language reasoning) plus a breakdown of how the score is composed. The full 100K pool is
not needed — small-sample reproducibility is the point.
