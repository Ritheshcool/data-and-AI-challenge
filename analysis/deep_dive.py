#!/usr/bin/env python3
"""
Second-pass scout: isolate the REAL honeypots (rare strong impossibility checks)
and see what GENUINE top-fit candidates look like (production retrieval/ranking/recsys
language in career descriptions at product companies). Also map the career-description
template space so the designer knows the vocabulary it's matching against.
"""
import json, os, datetime, collections, re

BASE = r"C:\Users\Rithesh\Desktop\projects\smart recruiter\challenge_data\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"
CAND = os.path.join(BASE, "candidates.jsonl")
OUTDIR = r"C:\Users\Rithesh\Desktop\projects\smart recruiter\analysis"
ANCHOR = datetime.date(2026, 6, 1)

PRODUCT_COMPANIES = {"swiggy", "zomato", "cred", "razorpay", "flipkart", "meesho", "inmobi",
                     "nykaa", "zoho", "freshworks", "vedantu", "ola", "phonepe", "paytm",
                     "myntra", "dream11", "sharechat", "unacademy", "byju", "groww", "zepto"}
CONSULTING = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl",
              "tech mahindra", "mphasis", "mindtree", "ibm", "deloitte"}
# Fictional placeholder companies (ambiguous product/services)
FICTIONAL = {"wayne enterprises", "initech", "pied piper", "globex inc", "acme corp",
             "dunder mifflin", "hooli", "stark industries"}

PROD_ML_PHRASES = ["recommendation system", "recommender", "recsys", "search engine", "search system",
                   "ranking system", "ranking model", "learning to rank", "learning-to-rank",
                   "retrieval", "semantic search", "vector search", "embeddings", "personalization",
                   "personalisation", "relevance", "matching engine", "matching system", "candidate generation",
                   "two-tower", "feed ranking", "re-ranking", "reranking", "nearest neighbor", "ann ",
                   "information retrieval", "deployed", "in production", "production", "serving",
                   "a/b test", "ndcg", "embedding", "fine-tun"]
HOBBY_PHRASES = ["side project", "self-learner", "self learner", "experimented with chatgpt",
                 "online courses", "haven't done it in a professional", "curious about how ai",
                 "exploring how llms", "self-directed", "kaggle", "augment my work", "emerging ai"]
RELEVANT_TITLE_KEYS = ["ai engineer", "ml engineer", "machine learning", "ai research", "applied scientist",
                       "data scientist", "research engineer", "nlp", "ai specialist"]

def parse_date(s):
    if not s: return None
    try: return datetime.date.fromisoformat(s[:10])
    except: return None

def real_honeypot_reasons(c):
    """Only the rare, strong impossibility signals. Returns list of human-readable reasons."""
    reasons = []
    p = c.get("profile", {}); yoe = p.get("years_of_experience"); ch = c.get("career_history", [])
    for r in ch:
        sd, ed = parse_date(r.get("start_date")), parse_date(r.get("end_date"))
        dur = r.get("duration_months")
        if sd and dur is not None:
            end = ed or ANCHOR
            elapsed = (end.year - sd.year) * 12 + (end.month - sd.month)
            if dur - elapsed > 9:   # claims far more tenure than the dates allow
                reasons.append(f"role '{r.get('title')}@{r.get('company')}' claims {dur}mo but dates allow only ~{elapsed}mo")
        if sd and ed and ed < sd:
            reasons.append(f"role '{r.get('title')}' end<start")
    # career span >> YOE
    spans = [(parse_date(r.get("start_date")), parse_date(r.get("end_date")) or ANCHOR) for r in ch if parse_date(r.get("start_date"))]
    if spans and yoe is not None:
        earliest = min(s for s, _ in spans); latest = max(e for _, e in spans)
        span = (latest.year - earliest.year) * 12 + (latest.month - earliest.month)
        if span - yoe * 12 > 30:
            reasons.append(f"career span ~{span}mo >> stated YOE {yoe}y")
    # expert/advanced with 0 months used (>=3)
    ze = [s.get("name") for s in c.get("skills", []) if s.get("proficiency") in ("expert", "advanced") and s.get("duration_months") == 0]
    if len(ze) >= 3:
        reasons.append(f"{len(ze)} expert/advanced skills with 0 months used: {ze[:6]}")
    # skill used longer than entire career (strong threshold)
    if yoe is not None:
        long_sk = [(s.get("name"), s.get("duration_months")) for s in c.get("skills", []) if (s.get("duration_months") or 0) > yoe * 12 + 24]
        if long_sk:
            reasons.append(f"skill duration > career+2y: {long_sk[:4]}")
    return reasons

def fit_score_quick(c):
    """Cheap heuristic only to surface candidate examples worth reading (NOT the real ranker)."""
    p = c.get("profile", {}); ch = c.get("career_history", []); rs = c.get("redrob_signals", {})
    title = (p.get("current_title") or "").lower()
    car_desc = " ".join((r.get("description") or "") for r in ch).lower()
    summ = (p.get("summary") or "").lower()
    s = 0
    if any(k in title for k in RELEVANT_TITLE_KEYS): s += 5
    if "senior" in title and any(k in title for k in RELEVANT_TITLE_KEYS): s += 2
    s += sum(1 for ph in PROD_ML_PHRASES if ph in car_desc)
    s -= 2 * sum(1 for ph in HOBBY_PHRASES if ph in summ)
    comps = [(r.get("company") or "").lower() for r in ch]
    if any(any(pc in cc for pc in PRODUCT_COMPANIES) for cc in comps): s += 3
    yoe = p.get("years_of_experience") or 0
    if 5 <= yoe <= 9: s += 2
    loc = (p.get("location") or "").lower()
    if p.get("country") == "India": s += 1
    return s

def main():
    real_hp = []
    relevant_titles = collections.Counter()
    gold = []   # (score, candidate compact)
    desc_prefixes = collections.Counter()
    title_industry = collections.Counter()

    with open(CAND, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try: c = json.loads(line)
            except: continue
            p = c.get("profile", {})
            title = (p.get("current_title") or "")
            tl = title.lower()
            if any(k in tl for k in RELEVANT_TITLE_KEYS):
                relevant_titles[title] += 1
                title_industry[(title, p.get("current_industry"))] += 1
            # description template space
            for r in c.get("career_history", []):
                d = (r.get("description") or "")[:60]
                desc_prefixes[d] += 1
            # real honeypots
            reasons = real_honeypot_reasons(c)
            if reasons:
                real_hp.append((c.get("candidate_id"), title, p.get("years_of_experience"), reasons))
            # gold candidates
            sc = fit_score_quick(c)
            if sc >= 8:
                gold.append((sc, c))

    gold.sort(key=lambda x: -x[0])

    out = []
    out.append(f"# Deep Dive\n")
    out.append(f"## Real honeypot suspects (rare strong impossibility checks): {len(real_hp)} found")
    for cid, title, yoe, reasons in real_hp[:60]:
        out.append(f"- {cid} [{title}, {yoe}y]: " + " | ".join(reasons))
    out.append("")

    out.append(f"## Relevant-title distribution (AI/ML/DS/research titles)")
    for t, n in relevant_titles.most_common(40):
        out.append(f"- {t}: {n}")
    out.append("")

    out.append(f"## Top {min(25,len(gold))} GOLD candidates by quick heuristic (read their descriptions)")
    for sc, c in gold[:25]:
        p = c["profile"]; rs = c.get("redrob_signals", {})
        la = parse_date(rs.get("last_active_date"))
        days = (ANCHOR - la).days if la else None
        out.append(f"\n### {c['candidate_id']}  score~{sc}  | {p.get('current_title')} @ {p.get('current_company')} ({p.get('current_industry')})")
        out.append(f"  loc={p.get('location')}, {p.get('country')} | yoe={p.get('years_of_experience')} | "
                   f"resp_rate={rs.get('recruiter_response_rate')} | days_inactive={days} | "
                   f"gh={rs.get('github_activity_score')} | relocate={rs.get('willing_to_relocate')} | notice={rs.get('notice_period_days')}d")
        out.append(f"  headline: {p.get('headline')}")
        out.append(f"  summary: {(p.get('summary') or '')[:400]}")
        for r in c.get("career_history", []):
            out.append(f"   - {r.get('title')} @ {r.get('company')} ({r.get('duration_months')}mo, {r.get('industry')}): {(r.get('description') or '')[:220]}")
        sk = [(s.get('name'), s.get('proficiency'), s.get('duration_months')) for s in c.get('skills', [])]
        out.append(f"  skills: {sk}")
    out.append("")

    out.append(f"## Most common career-description prefixes (template space, top 30)")
    for d, n in desc_prefixes.most_common(30):
        out.append(f"- ({n}) {d!r}")

    txt = "\n".join(out)
    with open(os.path.join(OUTDIR, "deep_dive.md"), "w", encoding="utf-8") as o:
        o.write(txt)
    print(f"real_honeypots={len(real_hp)}  gold_candidates(score>=8)={len(gold)}")
    print("WROTE deep_dive.md")

if __name__ == "__main__":
    main()
