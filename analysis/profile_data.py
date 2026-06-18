#!/usr/bin/env python3
"""
Deep profiling of the Redrob candidate pool.

Streams candidates.jsonl (no full load into memory), accumulates distributions,
runs honeypot/trap consistency heuristics, and writes:
  - analysis/data_profile.md   (human/agent-readable summary)
  - analysis/samples.json       (concrete example candidates per category)

This is a *scouting* artifact: the scoring design reasons over this, not the raw 465MB.
"""
import json, os, sys, datetime, collections, statistics

BASE = r"C:\Users\Rithesh\Desktop\projects\smart recruiter\challenge_data\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"
CAND = os.path.join(BASE, "candidates.jsonl")
OUTDIR = r"C:\Users\Rithesh\Desktop\projects\smart recruiter\analysis"
os.makedirs(OUTDIR, exist_ok=True)

CONSULTING = {"tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
              "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "mphasis",
              "deloitte", "ibm"}

TARGET_CITIES = {"pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
                 "bengaluru", "bangalore", "ncr", "new delhi", "navi mumbai"}

# JD-relevant skill vocabularies (lowercased substring match)
VEC_DB = ["pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
          "elasticsearch", "vespa", "vector database", "vector db", "annoy", "hnsw", "scann"]
RETRIEVAL = ["embedding", "sentence-transformer", "sentence transformer", "bge", "e5 ",
             "retrieval", "semantic search", "rag", "dense retrieval", "bm25", "information retrieval",
             "dpr", "colbert", "reranking", "re-ranking", "reranker"]
RANKING = ["learning to rank", "learning-to-rank", "ltr", "recommendation", "recommender",
           "recsys", "ranking", "xgboost", "lambdamart", "ranknet", "two-tower", "collaborative filtering"]
EVAL = ["ndcg", "mrr", "mean reciprocal", "mean average precision", " map ", "a/b test", "ab test",
        "offline eval", "evaluation framework", "precision@", "recall@"]
LLM_NLP = ["nlp", "natural language", "llm", "fine-tun", "lora", "qlora", "peft", "transformer",
           "bert", "gpt", "language model", "huggingface", "hugging face", "named entity", "text classification"]
NEG_DOMAIN = ["image classification", "gans", "gan ", "speech recognition", "tts", "computer vision",
              "robotics", "object detection", "ocr", "segmentation", "speech-to-text", "asr", "yolo", "opencv"]
TECH_TITLES = ["engineer", "developer", "scientist", "ml ", "machine learning", "data", "ai ",
               "research", "architect", "programmer", "sde", "swe"]
NONTECH_TITLES = ["marketing", "sales", "hr ", "human resource", "recruiter", "accountant",
                  "operations manager", "content writer", "graphic designer", "customer support",
                  "business analyst", "project manager", "civil engineer", "mechanical engineer"]

def has_any(text, vocab):
    t = text.lower()
    return [k for k in vocab if k in t]

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.date.fromisoformat(s[:10])
    except Exception:
        return None

def profile_text(c):
    """All free-text from a candidate, concatenated lower."""
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for r in c.get("career_history", []):
        parts.append(r.get("title", ""))
        parts.append(r.get("description", ""))
    return " ".join(parts)

def skills_text(c):
    return " ".join(s.get("name", "") for s in c.get("skills", []))

# ---- accumulators ----
n = 0
countries = collections.Counter()
india_cities = collections.Counter()
titles = collections.Counter()
industries = collections.Counter()
companies = collections.Counter()
yoe_vals = []
completeness_vals = []
resp_rate_vals = []
ghscore_vals = []
last_active_dates = []
signup_dates = []
end_dates = []
open_to_work = 0
willing_relocate = 0
n_skills_vals = []
edu_tiers = collections.Counter()
work_modes = collections.Counter()

# how many candidates reference JD skill areas (in skills list vs career text)
skill_area_hits = collections.Counter()       # in skills[]
career_area_hits = collections.Counter()       # in career descriptions only

# honeypot / impossibility checks
hp_checks = collections.Counter()              # check_name -> count
hp_flag_distribution = collections.Counter()   # num_flags -> num_candidates

# trap categories
n_consulting_only = 0
n_keyword_stuffer = 0          # nontech title + many AI skills
n_plain_gem = 0                # career text shows recsys/retrieval at product co, few buzzword skills
n_inactive_perfect = 0

samples = collections.defaultdict(list)
SAMPLE_CAP = 8

ANCHOR_FALLBACK = datetime.date(2026, 6, 1)

def add_sample(cat, c, note=""):
    if len(samples[cat]) < SAMPLE_CAP:
        samples[cat].append({"candidate_id": c.get("candidate_id"), "note": note,
                              "profile": c.get("profile"),
                              "career_titles": [(r.get("title"), r.get("company"), r.get("duration_months"))
                                                for r in c.get("career_history", [])]})

def honeypot_flags(c, anchor):
    """Return list of failed-consistency check names (impossibility signals)."""
    flags = []
    p = c.get("profile", {})
    yoe = p.get("years_of_experience")
    ch = c.get("career_history", [])
    # 1. role date / duration consistency
    spans = []
    for r in ch:
        sd, ed = parse_date(r.get("start_date")), parse_date(r.get("end_date"))
        dur = r.get("duration_months")
        if sd:
            end = ed or anchor
            if end < sd:
                flags.append("end_before_start")
            months = (end.year - sd.year) * 12 + (end.month - sd.month)
            if dur is not None and abs(months - dur) > 6:
                flags.append("duration_vs_dates_mismatch")
            spans.append((sd, end))
        if r.get("is_current") and r.get("end_date"):
            flags.append("current_with_enddate")
        if (not r.get("is_current")) and r.get("end_date") is None:
            flags.append("noncurrent_without_enddate")
        if sd and sd > anchor:
            flags.append("start_in_future")
    # 2. career span vs YOE
    if spans and yoe is not None:
        earliest = min(s for s, _ in spans)
        latest = max(e for _, e in spans)
        span_months = (latest.year - earliest.year) * 12 + (latest.month - earliest.month)
        if span_months > (yoe * 12) + 24:
            flags.append("careerspan_gt_yoe")
        # single role longer than entire career
        for sd, ed in spans:
            rm = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if rm > (yoe * 12) + 18:
                flags.append("role_longer_than_career")
                break
    # 3. skill duration > career length
    if yoe is not None:
        for s in c.get("skills", []):
            d = s.get("duration_months")
            if d is not None and d > (yoe * 12) + 12:
                flags.append("skill_dur_gt_career")
                break
    # 4. expert/advanced proficiency with 0 months used
    zero_expert = sum(1 for s in c.get("skills", [])
                      if s.get("proficiency") in ("expert", "advanced") and s.get("duration_months") == 0)
    if zero_expert >= 3:
        flags.append("expert_zero_duration")
    # 5. education sanity
    for e in c.get("education", []):
        sy, ey = e.get("start_year"), e.get("end_year")
        if sy and ey and ey < sy:
            flags.append("edu_end_before_start")
    # 6. signal sanity
    rs = c.get("redrob_signals", {})
    su, la = parse_date(rs.get("signup_date")), parse_date(rs.get("last_active_date"))
    if su and la and la < su:
        flags.append("active_before_signup")
    sal = rs.get("expected_salary_range_inr_lpa", {})
    if sal.get("min") is not None and sal.get("max") is not None and sal["min"] > sal["max"]:
        flags.append("salary_min_gt_max")
    # 7. profile_completeness 100 but empty sections
    if rs.get("profile_completeness_score", 0) >= 99 and (not c.get("skills") or not ch):
        flags.append("complete100_but_empty")
    return flags

def main():
    global n, open_to_work, willing_relocate, n_consulting_only, n_keyword_stuffer, n_plain_gem, n_inactive_perfect
    # first pass to find anchor (max last_active)
    max_active = ANCHOR_FALLBACK
    with open(CAND, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
            except Exception:
                continue
            la = parse_date(c.get("redrob_signals", {}).get("last_active_date"))
            if la and la > max_active:
                max_active = la
    anchor = max_active
    print("recency anchor (max last_active):", anchor)

    with open(CAND, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
            except Exception:
                continue
            n += 1
            p = c.get("profile", {})
            rs = c.get("redrob_signals", {})
            ch = c.get("career_history", [])

            countries[p.get("country", "?")] += 1
            loc = (p.get("location") or "").lower()
            if p.get("country") == "India":
                city = loc.split(",")[0].strip()
                india_cities[city] += 1
            titles[p.get("current_title", "?")] += 1
            industries[p.get("current_industry", "?")] += 1
            comp = (p.get("current_company") or "")
            companies[comp] += 1
            if p.get("years_of_experience") is not None:
                yoe_vals.append(p["years_of_experience"])
            completeness_vals.append(rs.get("profile_completeness_score", 0))
            resp_rate_vals.append(rs.get("recruiter_response_rate", 0))
            ghscore_vals.append(rs.get("github_activity_score", -1))
            la = parse_date(rs.get("last_active_date"))
            if la:
                last_active_dates.append(la)
            if rs.get("open_to_work_flag"):
                open_to_work += 1
            if rs.get("willing_to_relocate"):
                willing_relocate += 1
            n_skills_vals.append(len(c.get("skills", [])))
            for e in c.get("education", []):
                edu_tiers[e.get("tier", "unknown")] += 1
            work_modes[rs.get("preferred_work_mode", "?")] += 1

            sk_t = skills_text(c)
            car_t = " ".join(r.get("description", "") + " " + r.get("title", "") for r in ch)
            for area, vocab in [("vec_db", VEC_DB), ("retrieval", RETRIEVAL), ("ranking", RANKING),
                                ("eval", EVAL), ("llm_nlp", LLM_NLP), ("neg_domain", NEG_DOMAIN)]:
                if has_any(sk_t, vocab):
                    skill_area_hits[area] += 1
                if has_any(car_t, vocab):
                    career_area_hits[area] += 1

            # honeypot flags
            flags = honeypot_flags(c, anchor)
            uniq = set(flags)
            for fl in uniq:
                hp_checks[fl] += 1
            hp_flag_distribution[len(uniq)] += 1
            if len(uniq) >= 2:
                add_sample("honeypot_suspect", c, note=f"flags={sorted(uniq)}")

            # consulting-only career
            comps = [(r.get("company") or "").lower() for r in ch]
            if comps and all(any(cf in cc for cf in CONSULTING) for cc in comps):
                n_consulting_only += 1
                add_sample("consulting_only", c, note=f"companies={[r.get('company') for r in ch]}")

            # keyword stuffer: nontech current title but many AI skills in skills[]
            title_l = (p.get("current_title") or "").lower()
            is_nontech = any(nt in title_l for nt in NONTECH_TITLES) and not any(tt in title_l for tt in ["engineer", "scientist", "ml", "ai ", "developer", "data"])
            ai_skill_hits = len(set(has_any(sk_t, RETRIEVAL + VEC_DB + RANKING + LLM_NLP)))
            if is_nontech and ai_skill_hits >= 4:
                n_keyword_stuffer += 1
                add_sample("keyword_stuffer", c, note=f"title={p.get('current_title')} ai_skill_areas={ai_skill_hits}")

            # plain-language gem: career text shows recsys/retrieval/ranking, but few buzzword skills, tech title
            car_signal = set(has_any(car_t, RETRIEVAL + VEC_DB + RANKING))
            sk_signal = set(has_any(sk_t, RETRIEVAL + VEC_DB + RANKING))
            is_tech = any(tt in title_l for tt in ["engineer", "scientist", "ml", "machine learning", "ai ", "developer", "data"])
            if is_tech and car_signal and len(sk_signal) <= 1:
                n_plain_gem += 1
                add_sample("plain_gem", c, note=f"career_signals={sorted(car_signal)} title={p.get('current_title')}")

            # perfect-on-paper but inactive/unresponsive
            if la:
                days_inactive = (anchor - la).days
                if ai_skill_hits >= 3 and (days_inactive > 120 or rs.get("recruiter_response_rate", 1) < 0.1):
                    n_inactive_perfect += 1
                    add_sample("inactive_perfect", c, note=f"days_inactive={days_inactive} resp={rs.get('recruiter_response_rate')}")

    # ---- write outputs ----
    def pct(x):
        return f"{100.0*x/n:.1f}%"

    def topn(counter, k=20):
        return counter.most_common(k)

    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return {}
        vals_s = sorted(vals)
        q = lambda p: vals_s[min(len(vals_s)-1, int(p*len(vals_s)))]
        return {"min": round(min(vals),2), "p10": round(q(0.10),2), "median": round(statistics.median(vals),2),
                "mean": round(statistics.mean(vals),2), "p90": round(q(0.90),2), "max": round(max(vals),2)}

    md = []
    md.append(f"# Redrob Candidate Pool — Data Profile\n")
    md.append(f"- Total candidates: **{n}**")
    md.append(f"- Recency anchor (max last_active_date): **{anchor}**")
    md.append(f"- open_to_work: {open_to_work} ({pct(open_to_work)}); willing_to_relocate: {willing_relocate} ({pct(willing_relocate)})\n")

    md.append("## Geography")
    md.append(f"- Countries (top): {topn(countries, 12)}")
    md.append(f"- India cities (top): {topn(india_cities, 20)}\n")

    md.append("## Titles / Industry / Company")
    md.append(f"- current_title (top 30): {topn(titles, 30)}")
    md.append(f"- current_industry (top 20): {topn(industries, 20)}")
    md.append(f"- current_company (top 30): {topn(companies, 30)}\n")

    md.append("## Distributions")
    md.append(f"- years_of_experience: {stats(yoe_vals)}")
    md.append(f"- profile_completeness: {stats(completeness_vals)}")
    md.append(f"- recruiter_response_rate: {stats(resp_rate_vals)}")
    md.append(f"- github_activity_score (incl -1): {stats(ghscore_vals)}")
    md.append(f"- num_skills per candidate: {stats(n_skills_vals)}")
    md.append(f"- education tiers: {dict(edu_tiers)}")
    md.append(f"- preferred_work_mode: {dict(work_modes)}\n")

    md.append("## JD-skill area coverage")
    md.append(f"- candidates with area term in SKILLS list: {dict(skill_area_hits)}")
    md.append(f"- candidates with area term in CAREER descriptions: {dict(career_area_hits)}\n")

    md.append("## Honeypot / impossibility checks")
    md.append(f"- per-check counts: {dict(hp_checks)}")
    md.append(f"- distribution of #flags-per-candidate: {dict(sorted(hp_flag_distribution.items()))}")
    md.append(f"  (spec says ~80 honeypots forced to tier 0; candidates with >=2 flags are prime suspects)\n")

    md.append("## Trap-category candidate counts")
    md.append(f"- consulting-only careers: {n_consulting_only} ({pct(n_consulting_only)})")
    md.append(f"- keyword stuffers (nontech title + >=4 AI skill areas): {n_keyword_stuffer} ({pct(n_keyword_stuffer)})")
    md.append(f"- plain-language gems (career shows retrieval/recsys, <=1 buzzword skill, tech title): {n_plain_gem} ({pct(n_plain_gem)})")
    md.append(f"- perfect-on-paper but inactive/unresponsive: {n_inactive_perfect} ({pct(n_inactive_perfect)})\n")

    with open(os.path.join(OUTDIR, "data_profile.md"), "w", encoding="utf-8") as o:
        o.write("\n".join(md))

    with open(os.path.join(OUTDIR, "samples.json"), "w", encoding="utf-8") as o:
        json.dump({k: v for k, v in samples.items()}, o, indent=2)

    print("WROTE data_profile.md and samples.json")
    print("\n".join(md))

if __name__ == "__main__":
    main()
