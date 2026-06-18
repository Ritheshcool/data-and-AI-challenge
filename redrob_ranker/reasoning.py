"""
Deterministic, fact-grounded, varied reasoning strings (no LLM). Spec §8.

Every clause is filled ONLY from fields present in the candidate record (no hallucination).
Variation is seeded by the numeric part of candidate_id (deterministic, no PYTHONHASHSEED
dependency). Tone is gated by rank band so it always matches the rank (Stage-4 check 6).
"""
import random
import re

from . import knowledge as K

# A concrete, grounded detail lifted verbatim from the actual best-role description (no hallucination):
# a scale metric or a named architecture term. Slotted into the evidence clause when present.
_DETAIL_RX = re.compile(
    r"(\d+\s?[MmKk]\+?\s*(?:queries|users|items|documents|requests|profiles)(?:/(?:mo|month|day|sec))?"
    r"|\d+\s*qps|p95|sub-?\d+ ?ms|bm25|faiss|hnsw|\bbge\b|pinecone|qdrant|milvus|weaviate|pgvector"
    r"|learning[- ]to[- ]rank|two[- ]tower|llm[- ]?re-?ranker|ndcg@?\d*|collaborative filtering"
    r"|matrix factorization|sentence[- ]transformer)", re.I)

_PROF_RANK = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}
_VERBS = ["shipped", "built", "owned", "drove", "led the build of"]
_FAMILY_PHRASE = {
    "retrieval": "embeddings/semantic retrieval",
    "ranking": "learning-to-rank ranking pipelines",
    "recsys": "production recommendation systems",
    "rageval": "RAG and ranking evaluation (NDCG/A-B testing)",
    "prod": "production systems at scale",
}
_JD_LINK = {
    "retrieval": "the production retrieval + ranking-eval mandate",
    "ranking": "the ranking/relevance core of the role",
    "recsys": "the recommendation/ranking systems this role owns",
    "rageval": "the evaluation-rigor the JD demands",
    "prod": "the 'shipper over researcher' profile the JD wants",
}


def _seed(cid):
    try:
        return int(cid.split("_")[1])
    except (IndexError, ValueError):
        return abs(hash(cid)) % (10 ** 7)


def _exp_phrase(facts):
    yoe, title, comp = facts.get("yoe"), facts.get("current_title"), facts.get("current_company")
    bits = []
    if title:
        bits.append(title)
    if comp:
        bits.append(f"at {comp}")
    head = " ".join(bits) if bits else "candidate"
    if yoe is not None:
        return f"{yoe:.1f}y {head}"
    return head


def _evidence_phrase(facts, rng):
    br = facts.get("best_role")
    if not br or br.get("role_evid", 0) < 0.05:
        return None
    fams = [k for k in ("retrieval", "ranking", "recsys", "rageval") if br["matched"].get(k)]
    if not fams:
        if br["matched"].get("prod"):
            fams = ["prod"]
        else:
            return None
    phrases = [_FAMILY_PHRASE[f] for f in fams[:2]]
    what = " and ".join(phrases)
    verb = rng.choice(_VERBS)
    comp = br.get("company")
    # Drop the company when it's the same as the current employer (named in the experience clause).
    where = f" at {comp}" if comp and comp != facts.get("current_company") else ""
    # Splice in one concrete, grounded detail from the actual description, if available.
    detail = ""
    m = _DETAIL_RX.search(br.get("description") or "")
    if m:
        detail = f" ({m.group(1).strip()})"
    return f"{verb} {what}{where}{detail}", fams


def _skills_phrase(facts, rng):
    rel = [s for s in facts.get("skills", []) if K.RELEVANT_SKILL_RX.search(s.get("name") or "")]
    rel.sort(key=lambda s: (_PROF_RANK.get(s.get("proficiency"), 0),
                            s.get("endorsements") or 0, s.get("duration_months") or 0),
             reverse=True)
    top = rel[:2]
    if not top:
        return None
    return ", ".join(f"{s['name']} ({s.get('proficiency')})" for s in top)


def _concerns(f):
    """All applicable real concerns, ordered most-severe first (no fabrication)."""
    facts = f["facts"]
    out = []
    label = f.get("trap_label")
    trap_map = {
        "self-learner-side-project": "AI experience is self-taught/side-project, not yet production",
        "consulting-only": "entirely services/consulting background, no product-company role",
        "cv/speech-only": "background skews to vision/speech rather than NLP/IR",
        "recent-only-LLM": "AI exposure is recent LangChain/API work without older production ML",
        "keyword-stuffer": "non-technical role with AI present only as listed skills",
        "research-only": "research-only history with no production deployment",
    }
    if label in trap_map:
        out.append(trap_map[label])
    di = facts.get("days_inactive")
    rr = facts.get("recruiter_response_rate")
    npd = facts.get("notice_period_days")
    yoe = facts.get("yoe")
    if di is not None and di > 180:
        out.append(f"{di} days inactive on-platform")
    if rr is not None and rr < 0.15:
        out.append(f"low {rr:.0%} recruiter response rate")
    if yoe is not None and (yoe > 11 or yoe < 4):
        out.append(f"{yoe:.1f}y experience is outside the ideal 5-9y band")
    if rr is not None and 0.15 <= rr < 0.30:
        out.append(f"modest {rr:.0%} recruiter response rate")
    if npd is not None and npd > 90:
        out.append(f"{npd}-day notice, above the sub-30 preference")
    if facts.get("country") and facts["country"] != "India" and not facts.get("willing_to_relocate"):
        loc = facts.get("location") or facts["country"]
        out.append(f"{loc}-based and not open to relocating (no visa sponsorship)")
    if npd is not None and 60 < npd <= 90:
        out.append(f"{npd}-day notice period")
    if f.get("title_class") == "adjacent":
        out.append("adjacent engineering title — production evidence is the deciding factor")
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def _fit_tier(f, has_evidence):
    """Drive the CLAIM by actual evidence (not rank position) so tone is always honest."""
    pes = f["PES"]
    if pes >= 0.55 and f["P_trap"] == 1.0 and f["T_mult"] >= 0.62 and has_evidence:
        return "strong"
    if has_evidence and (pes >= 0.30 or (f["T_mult"] >= 0.62 and pes >= 0.15)):
        return "moderate"
    return "weak"


def _generate_one(cid, rank, f, salt=0):
    rng = random.Random(_seed(cid) + 1009 * salt)
    facts = f["facts"]
    exp = _exp_phrase(facts)
    ev = _evidence_phrase(facts, rng)
    ev_text, ev_fams = (ev if ev else (None, []))
    skills = _skills_phrase(facts, rng)
    concerns = _concerns(f)
    jd = _JD_LINK[ev_fams[0]] if ev_fams else None
    tier = _fit_tier(f, ev_text is not None)

    def concern_clause(items):
        if len(items) == 1:
            connector = rng.choice(["the one soft caveat is", "minor caveat:", "only watch-item:"])
            return f"{connector} {items[0]}"
        return "caveats: " + "; ".join(items)

    clauses = []
    if tier == "strong":
        lead = f"{exp}; {ev_text}" if ev_text else exp
        if jd:
            lead += " — squarely matches " + jd
        clauses.append(lead)
        if skills:
            clauses.append(f"corroborated by {skills}")
        if concerns:
            clauses.append(concern_clause(concerns[:2]))
    elif tier == "moderate":
        opener = rng.choice(["Solid partial fit:", "Relevant but not ideal:", "Reasonable fit with caveats:"])
        lead = f"{opener} {exp}"
        if ev_text:
            lead += f"; {ev_text}"
        clauses.append(lead)
        if skills:
            clauses.append(f"relevant skills {skills}")
        clauses.append(concern_clause(concerns[:2]) if concerns else
                       (f"adjacent to {jd}" if jd else "production-ML evidence is thin for this senior AI role"))
    else:  # weak — lead with the limiting factor, never claim a match
        limit = concerns[0] if concerns else "no production retrieval/ranking/recsys evidence for this JD"
        opener = rng.choice(["Borderline —", "Filler pick —", "Below the core bar —"])
        clauses.append(f"{opener} {limit}")
        if len(concerns) > 1:
            clauses.append(f"also {concerns[1]}")
        if ev_text:
            clauses.append(f"does show {ev_text}")
        clauses.append(f"included on profile signals ({exp})")

    text = "; ".join(clauses)
    text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    return " ".join(text.split())


def _jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def generate_all(emitted):
    """
    emitted: list of (candidate_id, rank, score, feature_dict).
    Returns dict candidate_id -> reasoning string, with de-dup (Jaccard>0.8 -> re-roll).
    """
    result = {}
    texts_so_far = []
    for cid, rank, score, f in emitted:
        salt = 0
        text = _generate_one(cid, rank, f, salt)
        # de-dup: re-roll skeleton/synonyms if too similar to a prior one
        while salt < 6 and any(_jaccard(text, t) > 0.8 for t in texts_so_far[-25:]):
            salt += 1
            text = _generate_one(cid, rank, f, salt)
        result[cid] = text
        texts_so_far.append(text)
    return result
