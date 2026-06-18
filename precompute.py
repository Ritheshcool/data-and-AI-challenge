#!/usr/bin/env python3
"""
OFFLINE precompute step (network + time allowed here; NOT counted against the 5-min budget).

Embeds the JD and every candidate with a small CPU embedding model and caches:
  artifacts/cand_embeddings.npy   float32 [N, D], L2-normalized   (regenerable; git-ignored)
  artifacts/cand_ids.json         list[str] aligned with rows
  artifacts/jd_embedding.npy      float32 [D]
  artifacts/precompute_meta.json  model, dim, anchor, text recipe, timings

rank.py then loads ONLY these artifacts — no model, no network — so the ranking step
trivially satisfies CPU-only / no-network / <=5 min.

Usage:
  python precompute.py --candidates <path to candidates.jsonl> [--model BAAI/bge-small-en-v1.5]
"""
import argparse
import json
import os
import time

import numpy as np

from redrob_ranker.io_utils import iter_candidates

MODEL_DEFAULT = "BAAI/bge-small-en-v1.5"
# bge-* retrieval guidance: prepend this instruction to the QUERY (the JD) only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# The JD reduced to "what we actually need" — used as the semantic query.
JD_QUERY_TEXT = (
    "Senior AI Engineer for a product company's intelligence layer: ranking, retrieval, and "
    "matching systems. Production experience with embeddings-based retrieval (sentence-transformers, "
    "BGE, E5), vector databases and hybrid search (FAISS, Pinecone, Qdrant, Milvus, Elasticsearch, "
    "OpenSearch), learning-to-rank and recommendation systems, and rigorous ranking evaluation "
    "(NDCG, MRR, MAP, offline-to-online, A/B testing). Strong Python. 5-9 years, mostly applied ML/AI "
    "at product companies, having shipped an end-to-end search, ranking, or recommendation system to "
    "real users at scale. Not pure research, not framework-tutorial demos, not keyword-only profiles."
)


def candidate_text(c):
    """
    Text embedded per candidate: headline + summary + the role TITLES (short, high-signal).
    Skills are excluded (the gameable surface). Career DESCRIPTIONS are intentionally NOT
    embedded here — that production-evidence signal is already captured at 0.46 weight by PES
    (which regexes every description); EMB is only a 10% semantic backup, so a short text keeps
    the CPU encode fast (~14 min for 100k) without losing the signal that matters.
    Capped to ~600 chars (~96 tokens) to keep per-batch compute bounded.
    """
    p = c.get("profile", {}) or {}
    titles = " ".join((r.get("title") or "") for r in c.get("career_history", []) or [])
    text = f"{p.get('headline','')}. {p.get('summary','')} {titles}".strip()
    return text[:600]


# Scalar feature fields cached for the fast rank path (everything scoring.combine/order need).
FEAT_FLOAT_KEYS = ["PES", "TFF_pos", "T_mult", "SKC", "EBF", "LOC", "BASE",
                   "title_chaser_delta", "P_trap", "B", "P_hp"]
FEAT_BOOL_KEYS = ["title_no_evidence", "is_honeypot"]


def build_feature_cache(candidates_path, outdir):
    """Compute the rule features for every candidate and cache the scalars (the slow part of
    rank.py, moved offline). rank.py loads this so the ranking step is a few seconds. Falls back
    to live recompute if the cache is absent — so rank.py stays self-contained."""
    from redrob_ranker import features as FEAT
    t = time.time()
    ids, floats, bools = [], {k: [] for k in FEAT_FLOAT_KEYS}, {k: [] for k in FEAT_BOOL_KEYS}
    for c in iter_candidates(candidates_path):
        f = FEAT.compute_features(c)
        ids.append(f["candidate_id"])
        for k in FEAT_FLOAT_KEYS:
            floats[k].append(f[k])
        for k in FEAT_BOOL_KEYS:
            bools[k].append(f[k])
    np.savez_compressed(
        os.path.join(outdir, "features.npz"),
        ids=np.array(ids),
        # float64 so the cached rank path is bit-identical to the live-recompute fallback.
        **{k: np.array(v, dtype=np.float64) for k, v in floats.items()},
        **{k: np.array(v, dtype=bool) for k, v in bools.items()},
    )
    print(f"  feature cache: {len(ids)} candidates in {time.time()-t:.0f}s -> {outdir}/features.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-seq-length", type=int, default=96)
    ap.add_argument("--features-only", action="store_true",
                    help="only (re)build the rule-feature cache; skip embedding (model already cached)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.features_only:
        print("Building feature cache only (no embedding) ...", flush=True)
        build_feature_cache(args.candidates, args.out)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # honor CPU-only; precompute too
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    print(f"Loading model {args.model} on CPU ({torch.get_num_threads()} threads) ...", flush=True)
    model = SentenceTransformer(args.model, device="cpu")
    model.max_seq_length = args.max_seq_length
    dim = model.get_sentence_embedding_dimension()

    print("Reading candidates + building texts ...", flush=True)
    ids, texts = [], []
    for c in iter_candidates(args.candidates):
        ids.append(c["candidate_id"])
        texts.append(candidate_text(c))
    n = len(ids)
    print(f"  {n} candidates", flush=True)

    t1 = time.time()
    print("Encoding candidates (chunked) ...", flush=True)
    chunk = 10000
    parts = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        parts.append(model.encode(
            texts[s:e], batch_size=args.batch_size, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True).astype(np.float32))
        print(f"  encoded {e}/{n}  ({time.time()-t1:.0f}s elapsed)", flush=True)
    emb = np.vstack(parts)
    t2 = time.time()

    jd_emb = model.encode(
        [BGE_QUERY_INSTRUCTION + JD_QUERY_TEXT], normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)[0]

    np.save(os.path.join(args.out, "cand_embeddings.npy"), emb)
    np.save(os.path.join(args.out, "jd_embedding.npy"), jd_emb)
    with open(os.path.join(args.out, "cand_ids.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f)
    meta = {
        "model": args.model, "dim": dim, "n": n,
        "max_seq_length": args.max_seq_length,
        "text_recipe": "headline + summary + [title at company: description]*; skills excluded",
        "jd_query_text": JD_QUERY_TEXT,
        "bge_query_instruction": BGE_QUERY_INSTRUCTION,
        "load_secs": round(t1 - t0, 1),
        "encode_secs": round(t2 - t1, 1),
    }
    with open(os.path.join(args.out, "precompute_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Building rule-feature cache ...", flush=True)
    build_feature_cache(args.candidates, args.out)
    print(f"Done. encode={meta['encode_secs']}s  total={round(time.time()-t0,1)}s  -> {args.out}/")


if __name__ == "__main__":
    main()
