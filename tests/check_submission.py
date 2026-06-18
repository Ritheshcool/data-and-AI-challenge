#!/usr/bin/env python3
"""
Pre-submit checks (spec §9.3 / §10): format + quality. Run before every upload.

  python tests/check_submission.py --submission submission.csv --candidates candidates.jsonl

Exits non-zero on any failure. Covers:
  - exactly 100 rows; ranks 1..100 each once; unique candidate_ids; all exist in pool
  - score strictly/ non-increasing; tie-break (equal score -> id asc) satisfied
  - zero honeypots in top-100 (recomputed)
  - reasoning: non-empty, all distinct, no skill/employer hallucination
  - known gold candidates land in the top ~40; honeypots absent
"""
import argparse
import csv
import re
import sys

sys.path.insert(0, ".")
from redrob_ranker import features as F
from redrob_ranker.io_utils import iter_candidates

GOLD_TOP = ["CAND_0077337", "CAND_0008425", "CAND_0018499", "CAND_0080766", "CAND_0033861",
            "CAND_0044855", "CAND_0079387", "CAND_0081846", "CAND_0005260", "CAND_0060054"]
CID_RX = re.compile(r"^CAND_[0-9]{7}$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--candidates", required=True)
    args = ap.parse_args()
    errors, warnings = [], []

    with open(args.submission, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if any(c.strip() for c in r)]

    if header != ["candidate_id", "rank", "score", "reasoning"]:
        errors.append(f"bad header: {header}")
    if len(rows) != 100:
        errors.append(f"expected 100 data rows, got {len(rows)}")

    ids, ranks, scores, reasons = [], [], [], []
    for i, r in enumerate(rows):
        if len(r) != 4:
            errors.append(f"row {i+2}: expected 4 cols, got {len(r)}"); continue
        cid, rank_s, score_s, reasoning = r
        if not CID_RX.match(cid):
            errors.append(f"row {i+2}: bad candidate_id {cid!r}")
        ids.append(cid); reasons.append(reasoning)
        try:
            ranks.append(int(rank_s))
        except ValueError:
            errors.append(f"row {i+2}: rank not int: {rank_s!r}")
        try:
            scores.append(float(score_s))
        except ValueError:
            errors.append(f"row {i+2}: score not float: {score_s!r}")

    if sorted(ranks) != list(range(1, 101)):
        errors.append("ranks must be exactly 1..100 each once")
    if len(set(ids)) != len(ids):
        errors.append("duplicate candidate_id in submission")
    for a, b in zip(scores, scores[1:]):
        if a < b:
            errors.append(f"score increased with rank ({a} < {b})")
            break
    # tie-break: equal scores -> candidate_id ascending
    for i in range(len(scores) - 1):
        if scores[i] == scores[i + 1] and ids[i] > ids[i + 1]:
            errors.append(f"equal scores at ranks {i+1},{i+2} but ids not ascending")
            break
    if any(not s.strip() for s in reasons):
        errors.append("empty reasoning present")
    if len(set(reasons)) != len(reasons):
        errors.append("duplicate reasoning strings present")

    # Load the top-100 candidate records + a sample for gold/honeypot checks.
    wanted = set(ids) | set(GOLD_TOP)
    recs = {}
    for c in iter_candidates(args.candidates):
        if c["candidate_id"] in wanted:
            recs[c["candidate_id"]] = c

    missing = [cid for cid in ids if cid not in recs]
    if missing:
        errors.append(f"{len(missing)} submitted ids not found in pool, e.g. {missing[:3]}")

    # honeypots in top-100 + no hallucination
    hp_in_top = 0
    halluc = 0
    for cid, reasoning in zip(ids, reasons):
        c = recs.get(cid)
        if not c:
            continue
        feat = F.compute_features(c)
        if feat["is_honeypot"]:
            hp_in_top += 1
        # hallucination guard: companies mentioned via "at X" and skills via "Name (prof)"
        companies = {(r.get("company") or "").lower() for r in c.get("career_history", [])}
        skillnames = {(s.get("name") or "").lower() for s in c.get("skills", [])}
        for m in re.findall(r" at ([A-Z][\w.&'-]+(?: [A-Z][\w.&'-]+)*)", reasoning):
            if m.lower() not in companies and not any(m.lower() in cc for cc in companies):
                # allow the JD-link phrases; only flag capitalized company-like tokens absent from profile
                if m.lower() not in ("scale",):
                    halluc += 1
        for m in re.findall(r"([A-Z][\w .&/+-]+?) \((expert|advanced|intermediate|beginner)\)", reasoning):
            if m[0].lower() not in skillnames:
                halluc += 1
    if hp_in_top > 0:
        errors.append(f"{hp_in_top} honeypot(s) in top-100 (must be 0)")
    if halluc > 0:
        errors.append(f"{halluc} possible hallucinated skill/employer mention(s) in reasoning")

    # gold placement
    rank_of = {cid: rnk for cid, rnk in zip(ids, ranks)}
    placed = {g: rank_of.get(g) for g in GOLD_TOP}
    not_top40 = [g for g, rk in placed.items() if rk is None or rk > 40]
    if len(not_top40) > 3:
        warnings.append(f"gold candidates not in top-40: {not_top40} (placements: {placed})")
    else:
        print(f"[ok] gold placements: {placed}")

    print(f"\n{'FAILED' if errors else 'PASSED'} — {len(errors)} error(s), {len(warnings)} warning(s)")
    for e in errors:
        print(f"  ERROR: {e}")
    for w in warnings:
        print(f"  warn:  {w}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
