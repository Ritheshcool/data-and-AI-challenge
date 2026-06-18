#!/usr/bin/env python3
"""
Redrob candidate ranker — ONLINE ranking step (<=5 min, <=16GB, CPU-only, NO network).

Reads candidates.jsonl + precomputed embedding artifacts (from precompute.py) and writes
the top-100 submission CSV. No model load and no network here — the only precompute
dependency is the cached embedding matrix, which the challenge spec explicitly permits.

Single reproduce command:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
import os
# Determinism: pin BLAS threads + hash seed BEFORE numpy import.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import csv
import json
import re
import sys
import time

import numpy as np

from redrob_ranker import features as F
from redrob_ranker import scoring as S
from redrob_ranker import reasoning as R
from redrob_ranker.io_utils import iter_candidates

# Must match precompute.build_feature_cache.
FEAT_FLOAT_KEYS = ["PES", "TFF_pos", "T_mult", "SKC", "EBF", "LOC", "BASE",
                   "title_chaser_delta", "P_trap", "B", "P_hp"]
FEAT_BOOL_KEYS = ["title_no_evidence", "is_honeypot"]
_ID_RX = re.compile(r'"candidate_id"\s*:\s*"(CAND_\d{7})"')


def load_feature_cache(artdir):
    """Load precomputed scalar features -> list of slim feature dicts, or None if absent."""
    p = os.path.join(artdir, "features.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    ids = d["ids"]
    floats = {k: d[k] for k in FEAT_FLOAT_KEYS}
    bools = {k: d[k] for k in FEAT_BOOL_KEYS}
    feats = []
    for i, cid in enumerate(ids):
        f = {"candidate_id": str(cid)}
        for k in FEAT_FLOAT_KEYS:
            f[k] = float(floats[k][i])
        for k in FEAT_BOOL_KEYS:
            f[k] = bool(bools[k][i])
        feats.append(f)
    return feats


def fetch_records(path, idset):
    """Stream the jsonl and return {candidate_id: record} for ids in idset (cheap id pre-match)."""
    recs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _ID_RX.search(line)
            if not m or m.group(1) not in idset:
                continue
            c = json.loads(line)
            recs[c["candidate_id"]] = c
            if len(recs) == len(idset):
                break
    return recs

# §EXACT WEIGHTS B calibration — frozen from the actual bge-small cosine distribution on
# this pool: pool p50/p90/p99 = 0.643/0.669/0.689, while the known gold fits sit at 0.717-0.836
# (all above pool p99). So the discriminative band is ~[0.66, 0.80]: bulk of pool -> 0, genuine
# fits -> 0.4-1.0. (The earlier 0.20/0.55 saturated everyone to 1.0 and wasted the signal.)
EMB_FLOOR, EMB_CEIL = 0.66, 0.80


def load_embeddings(artdir):
    emb_path = os.path.join(artdir, "cand_embeddings.npy")
    ids_path = os.path.join(artdir, "cand_ids.json")
    jd_path = os.path.join(artdir, "jd_embedding.npy")
    if not (os.path.exists(emb_path) and os.path.exists(ids_path) and os.path.exists(jd_path)):
        return None
    mat = np.load(emb_path, mmap_mode="r").astype(np.float32)
    with open(ids_path, encoding="utf-8") as f:
        ids = json.load(f)
    jd = np.load(jd_path).astype(np.float32)
    cos = mat @ jd                                  # both L2-normalized -> cosine
    emb_score = np.clip((cos - EMB_FLOOR) / (EMB_CEIL - EMB_FLOOR), 0.0, 1.0)
    return {cid: float(emb_score[i]) for i, cid in enumerate(ids)}, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--debug-out", default=None, help="optional CSV dump of top-100 feature detail")
    args = ap.parse_args()

    t0 = time.time()
    emb_loaded = load_embeddings(args.artifacts)
    if emb_loaded is None:
        print(f"[warn] no embedding artifacts in {args.artifacts}/ — EMB component set to 0 "
              f"(run precompute.py for full quality).", file=sys.stderr)
        emb_by_id, cos_arr = {}, None
    else:
        emb_by_id, cos_arr = emb_loaded
        if cos_arr is not None:
            print(f"[info] cosine(JD, candidate): min={cos_arr.min():.3f} "
                  f"median={np.median(cos_arr):.3f} p99={np.percentile(cos_arr,99):.3f} "
                  f"max={cos_arr.max():.3f}")

    feats = load_feature_cache(args.artifacts)
    used_cache = feats is not None
    if used_cache:
        print(f"[info] loaded feature cache ({len(feats)} candidates)")
    else:
        print("[info] no feature cache — computing rule features live (slower, still in budget) ...")
        feats = [F.compute_features(c) for c in iter_candidates(args.candidates)]
    n = len(feats)
    emb = np.array([emb_by_id.get(f["candidate_id"], 0.0) for f in feats], dtype=np.float64)
    missing = sum(1 for f in feats if f["candidate_id"] not in emb_by_id) if emb_by_id else n
    print(f"[info] {n} candidates; EMB missing for {missing}")

    n_hp = sum(1 for f in feats if f["is_honeypot"])
    print(f"[info] honeypots flagged: {n_hp}")
    if n_hp > 300:
        print(f"[warn] honeypot flag count {n_hp} > 300 — detection may be too loose; inspect.",
              file=sys.stderr)

    feats, p60 = S.combine(feats, emb)
    n_elite = sum(1 for f in feats if f["head_eligible"] and not f["is_honeypot"])
    print(f"[info] head-eligible (PES>=.55, P_trap=1, B>=.92): {n_elite}; content p60={p60:.3f}")

    # When using the cache we take a BUFFERED prelim (140), re-derive those records' features LIVE
    # (negligible), and (a) attach `facts` for reasoning, (b) re-apply the honeypot gate on fresh
    # features — a defensive refilter that guarantees 0 honeypots in the top-100 even if the cache
    # is stale relative to the code (e.g. signals were added after the cache was built). Cache-based
    # scores are kept so ordering is unchanged when cache and code agree.
    if used_cache:
        prelim = S.order_topk(feats, k=140)
        records = fetch_records(args.candidates, {f["candidate_id"] for f in prelim})
        enriched = []
        for f in prelim:
            full = F.compute_features(records[f["candidate_id"]])
            for k in ("content", "emb", "final", "head_eligible"):
                full[k] = f[k]
            enriched.append(full)
        dropped = [f["candidate_id"] for f in enriched if f["is_honeypot"]]
        ordered = [f for f in enriched if not f["is_honeypot"]][:100]
        if dropped:
            print(f"[warn] refilter dropped {len(dropped)} fresh-flagged honeypot(s) — cache is "
                  f"stale vs code; rebuild features.npz. ({dropped[:6]})", file=sys.stderr)
    else:
        ordered = S.order_topk(feats, k=100)

    emitted = S.emit_scores(ordered)
    reasons = R.generate_all(emitted)

    with open(args.out, "w", encoding="utf-8", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank, score, f in emitted:
            w.writerow([cid, rank, f"{score:.6f}", reasons[cid]])

    # diagnostics
    hp_in_top = sum(1 for cid, rank, score, f in emitted if f["is_honeypot"])
    print(f"[info] wrote {args.out}  (honeypots in top-100: {hp_in_top})")
    print(f"[info] total rank time: {time.time()-t0:.1f}s")

    if args.debug_out:
        with open(args.debug_out, "w", encoding="utf-8", newline="") as fdbg:
            w = csv.writer(fdbg)
            w.writerow(["rank", "candidate_id", "final", "content", "PES", "EMB", "T_mult",
                        "SKC", "EBF", "LOC", "BASE", "B", "P_trap", "P_hp", "title_class",
                        "trap_label", "current_title", "current_company", "country",
                        "yoe", "days_inactive", "resp_rate", "notice", "reasoning"])
            for cid, rank, score, f in emitted:
                fa = f["facts"]
                w.writerow([rank, cid, f"{f['final']:.4f}", f"{f['content']:.4f}",
                            f"{f['PES']:.3f}", f"{f['emb']:.3f}", f"{f['T_mult']:.2f}",
                            f"{f['SKC']:.3f}", f"{f['EBF']:.3f}", f"{f['LOC']:.2f}",
                            f"{f['BASE']:.3f}", f"{f['B']:.3f}", f"{f['P_trap']:.2f}",
                            f"{f['P_hp']:.2f}", f["title_class"], f["trap_label"] or "",
                            fa["current_title"], fa["current_company"], fa["country"],
                            fa["yoe"], fa["days_inactive"], fa["recruiter_response_rate"],
                            fa["notice_period_days"], reasons[cid]])
        print(f"[info] wrote debug detail to {args.debug_out}")


if __name__ == "__main__":
    main()
