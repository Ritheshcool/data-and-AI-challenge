"""
Score combination, top-10 safety lock, ordering, and validator-safe score emission.
Implements analysis/design_spec.md §1.2 composition, §3.4 title caps, §7 ordering.
"""
import numpy as np

# §EXACT WEIGHTS — content block (sums to 1.0), all components in [0,1].
W = {"PES": 0.46, "EMB": 0.10, "TFF_pos": 0.14, "SKC": 0.10, "EBF": 0.08, "LOC": 0.04, "BASE": 0.08}
SYNERGY_W = 0.08          # 0.08 · TFF_pos · PES
GATE_FLOOR, GATE_SLOPE = 0.35, 0.65   # gate = 0.35 + 0.65·T_mult

# Head (ranks 1-10) eligibility — §7 top-10 safety lock.
HEAD_PES_MIN, HEAD_B_MIN = 0.55, 0.92


def combine(feats, emb):
    """
    feats: list of feature dicts (features.compute_features output).
    emb:   np.array aligned to feats, the EMB component already calibrated to [0,1].
    Returns the same dicts annotated with 'content', 'final', 'head_eligible'.
    """
    n = len(feats)
    content = np.empty(n, dtype=np.float64)
    for i, f in enumerate(feats):
        c = (W["PES"] * f["PES"] + W["EMB"] * emb[i] + W["TFF_pos"] * f["TFF_pos"]
             + W["SKC"] * f["SKC"] + W["EBF"] * f["EBF"] + W["LOC"] * f["LOC"]
             + W["BASE"] * f["BASE"] + f["title_chaser_delta"])
        content[i] = c

    # TITLE_NO_EVIDENCE: cap content at the pool's 60th percentile (not a multiplier).
    p60 = float(np.percentile(content, 60))
    for i, f in enumerate(feats):
        if f["title_no_evidence"]:
            content[i] = min(content[i], p60)

    for i, f in enumerate(feats):
        synergy = SYNERGY_W * f["TFF_pos"] * f["PES"]
        gated = (content[i] + synergy) * (GATE_FLOOR + GATE_SLOPE * f["T_mult"])
        final = gated * f["P_trap"] * f["B"] * f["P_hp"]
        f["content"] = float(content[i])
        f["emb"] = float(emb[i])
        f["final"] = float(final)
        f["head_eligible"] = (f["PES"] >= HEAD_PES_MIN and f["P_trap"] == 1.0
                              and f["P_hp"] == 1.0 and f["B"] >= HEAD_B_MIN)
    return feats, p60


def order_topk(feats, k=100):
    """
    Exclude honeypots; sort by (-final, candidate_id); apply the top-10 safety lock
    (ranks 1-10 must be head-eligible if >=10 exist); return the top-k feature dicts in rank order.
    """
    pool = [f for f in feats if not f["is_honeypot"]]
    pool.sort(key=lambda f: (-f["final"], f["candidate_id"]))

    elite = [f for f in pool if f["head_eligible"]]
    if len(elite) >= 10:
        head = elite[:10]
        head_ids = {f["candidate_id"] for f in head}
        tail = [f for f in pool if f["candidate_id"] not in head_ids]  # preserves sorted order
        ordered = head + tail
    else:
        head_ids = {f["candidate_id"] for f in elite}
        tail = [f for f in pool if f["candidate_id"] not in head_ids]
        ordered = elite + tail
    return ordered[:k]


def emit_scores(ordered):
    """
    Assign ranks 1..N and STRICTLY-DECREASING 6-dp scores:
        s_i = round(min(final_i, s_{i-1} - 1e-6), 6)
    This guarantees the validator's 'score non-increasing' rule and makes its
    equal-score tie-break vacuous (no two scores are ever equal).
    Returns list of (candidate_id, rank, score, feature_dict).
    """
    out = []
    prev = float("inf")
    for idx, f in enumerate(ordered):
        rank = idx + 1
        raw = f["final"]
        s = round(min(raw, prev - 1e-6), 6)
        prev = s
        out.append((f["candidate_id"], rank, s, f))
    return out
