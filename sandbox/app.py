"""
Redrob ranker — sandbox demo (Streamlit).

Accepts a small candidate sample (<=100, uploaded .jsonl/.json or the bundled sample), runs the
SAME ranking logic as rank.py, and shows the ranked table + per-candidate score breakdown and
reasoning. Embeddings are computed live here (fine for <=100 candidates, ~1s); the full-pool path
uses precomputed vectors instead.

Run locally:   streamlit run sandbox/app.py
Deploy:        HuggingFace Spaces / Streamlit Cloud (see sandbox/README.md)
"""
import json
import os
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precompute import candidate_text, JD_QUERY_TEXT, BGE_QUERY_INSTRUCTION
from rank import EMB_FLOOR, EMB_CEIL
from redrob_ranker import features as F, scoring as S, reasoning as R

st.set_page_config(page_title="Redrob Ranker", layout="wide")


@st.cache_resource
def get_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    m.max_seq_length = 96
    return m


def load_candidates(uploaded):
    if uploaded is None:
        path = os.path.join(os.path.dirname(__file__), "sample_candidates.jsonl")
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    raw = uploaded.read().decode("utf-8")
    raw = raw.strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


def rank_sample(cands, model):
    texts = [candidate_text(c) for c in cands]
    mat = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    jd = model.encode([BGE_QUERY_INSTRUCTION + JD_QUERY_TEXT], normalize_embeddings=True,
                      convert_to_numpy=True).astype(np.float32)[0]
    cos = mat @ jd
    emb = np.clip((cos - EMB_FLOOR) / (EMB_CEIL - EMB_FLOOR), 0.0, 1.0)
    feats = [F.compute_features(c) for c in cands]
    feats, _ = S.combine(feats, emb)
    ordered = S.order_topk(feats, k=min(100, len(feats)))
    emitted = S.emit_scores(ordered)
    reasons = R.generate_all(emitted)
    return emitted, reasons


st.title("🔎 Redrob — Intelligent Candidate Ranker")
st.caption("Ranks candidates for the *Senior AI Engineer (Founding Team)* JD — by fit, not keywords. "
           "Same scoring as the full pipeline; embeddings computed live for this small sample.")

uploaded = st.sidebar.file_uploader("Upload candidates (.jsonl or .json, ≤100)", type=["jsonl", "json"])
st.sidebar.markdown("Leave empty to use the bundled 50-candidate sample.")

if st.sidebar.button("Rank", type="primary") or "ranked" not in st.session_state:
    with st.spinner("Loading model + ranking ..."):
        cands = load_candidates(uploaded)[:100]
        model = get_model()
        emitted, reasons = rank_sample(cands, model)
        st.session_state["ranked"] = (emitted, reasons, len(cands))

emitted, reasons, n = st.session_state["ranked"]
st.success(f"Ranked {n} candidates · top {len(emitted)} shown")

rows = []
for cid, rank, score, f in emitted:
    fa = f["facts"]
    rows.append({
        "rank": rank, "score": round(score, 4),
        "title": fa["current_title"], "company": fa["current_company"],
        "country": fa["country"], "yoe": fa["yoe"],
        "PES": round(f["PES"], 2), "EMB": round(f["emb"], 2), "RoleGate": round(f["T_mult"], 2),
        "Avail(B)": round(f["B"], 2), "honeypot": f["is_honeypot"],
        "reasoning": reasons[cid],
    })
st.dataframe(rows, use_container_width=True, hide_index=True)

with st.expander("How the score is built"):
    st.markdown(
        "`content = 0.46·PES + 0.10·EMB + 0.14·Title + 0.10·Skills + 0.08·Exp + 0.04·Loc + 0.08·Base`  \n"
        "`final = (content + synergy)·RoleGate · TrapPenalty · Availability · HoneypotGate`  \n"
        "PES reads career *descriptions* (least gameable); the role-class gate crushes keyword-stuffers; "
        "behavioral availability separates reachable from perfect-on-paper; 5 checks exclude impossible profiles.")
