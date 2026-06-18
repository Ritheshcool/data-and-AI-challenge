"""
Per-candidate feature extraction. Implements every formula in analysis/design_spec.md
§EXACT WEIGHTS A-I, plus the honeypot signals (§6) and trap penalties (§3.4).

compute_features(candidate) -> dict of scalars in [0,1] (PES, TFF_pos, SKC, EBF, LOC, BASE),
multipliers (T_mult, P_trap, B, P_hp), flags, and a `facts` sub-dict for reasoning.
EMB is computed separately in rank.py (needs the embedding matrix) and merged in scoring.
"""
import math
import re

from . import knowledge as K
from .io_utils import parse_date, months_between

# Leading "N years of ..." tenure stated in the free-text summary (honeypot signal D).
_SUMMARY_YEARS_RX = re.compile(r"([\d.]+)\s*\+?\s*years?\s+of", re.I)

PROF_WEIGHT = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.5, "beginner": 0.25}


def _role_evidence(candidate):
    """Per-role evidence + bookkeeping for PES, recency, and the strongest role (reasoning)."""
    anchor = K.RECENCY_ANCHOR
    roles = []
    families_seen = set()
    for r in candidate.get("career_history", []) or []:
        d = (r.get("description") or "")
        matched = {}
        role_raw = 0.0
        for key, rx, w in K.EVIDENCE_FAMILIES:
            if rx.search(d):
                matched[key] = True
                role_raw += w
                families_seen.add(key)
        role_raw = min(role_raw, 1.0)
        cls = K.company_class(r.get("company"))
        cmult = K.COMPANY_MULT[cls]
        m = r.get("duration_months") or 0
        dur_w = max(0.0, min(m / 24.0, 1.0))
        end = parse_date(r.get("end_date")) or anchor
        mb = months_between(end, anchor)
        recency_w = 1.0 if (mb is not None and mb <= 60) else 0.60
        role_evid = role_raw * cmult * dur_w * recency_w
        roles.append({
            "title": r.get("title"), "company": r.get("company"), "company_class": cls,
            "duration_months": m, "role_raw": role_raw, "role_evid": role_evid,
            "matched": matched, "ended_months_ago": mb if mb is not None else 999,
            "description": d,
        })
    return roles, families_seen


def _pes(roles, families_seen):
    prod = 1.0
    for r in roles:
        prod *= (1.0 - 0.60 * r["role_evid"])
    pes_raw = 1.0 - prod
    distinct_ireval = len(families_seen & K.IREVAL_FAMILY_KEYS)
    bonus = 0.05 * min(distinct_ireval, 2) / 2.0
    return max(0.0, min(pes_raw + bonus, 1.0)), distinct_ireval


def _title_mult(candidate, pes):
    p = candidate.get("profile", {}) or {}
    current = p.get("current_title") or ""
    tc = K.title_class(current)
    roles = candidate.get("career_history", []) or []
    prior_core = any(K.title_class(r.get("title")) == "core" for r in roles)
    if tc == "core":
        T = 1.00
    elif tc == "adjacent":
        if prior_core:
            T = 0.78
        elif pes >= 0.40:
            T = 0.62
        else:
            T = 0.30
    elif tc == "nontech":
        T = 0.06
    else:  # unknown title
        T = 0.55 if pes >= 0.40 else 0.25
    if K.JUNIOR_RX.search(current):
        T = min(T, 0.70)
    return T, tc


def _skill_corroboration(candidate):
    assess = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    assess_lc = {str(k).lower(): v for k, v in assess.items()}
    n_cred = 0
    a_count_expert_zero = 0          # ALL skills (honeypot signal A)
    rel_assess_vals = []
    for s in candidate.get("skills", []) or []:
        name = (s.get("name") or "")
        prof = s.get("proficiency")
        du = s.get("duration_months") or 0
        end = s.get("endorsements") or 0
        if prof in ("expert", "advanced") and du == 0:
            a_count_expert_zero += 1
        if not K.RELEVANT_SKILL_RX.search(name):
            continue
        if prof in ("expert", "advanced") and du == 0:
            continue  # zero credit (also flagged above)
        prof_w = PROF_WEIGHT.get(prof, 0.25)
        dur_w = 1.0 if du >= 12 else 0.7 if du >= 6 else 0.45 if du >= 1 else 0.15
        endor_w = 0.5 + 0.5 * min(end / 15.0, 1.0)
        a = assess_lc.get(name.lower())
        assess_bonus = 0.10 if (a is not None and a >= 70) else 0.0
        if a is not None:
            rel_assess_vals.append(a)
        credit = prof_w * dur_w * endor_w + assess_bonus
        if credit >= 0.5:
            n_cred += 1
    n_cred = min(n_cred, 6)
    rel_assess_avg = (sum(rel_assess_vals) / len(rel_assess_vals)) if rel_assess_vals else 0.0
    skc = 0.5 * math.log1p(n_cred) / math.log1p(6) + 0.5 * min(rel_assess_avg / 80.0, 1.0)
    return max(0.0, min(skc, 1.0)), a_count_expert_zero, n_cred


def _ebf(candidate):
    p = candidate.get("profile", {}) or {}
    y = p.get("years_of_experience")
    if y is None:
        return 0.25
    career_months = sum((r.get("duration_months") or 0) for r in candidate.get("career_history", []) or [])
    career_years = career_months / 12.0

    def band(v):
        if 6 <= v <= 8:
            return 1.00
        if 5 <= v < 6:
            return 0.6 + 0.4 * (v - 5)
        if 8 < v <= 9:
            return 0.6 + 0.4 * (9 - v)
        if 3.5 <= v < 5:
            return 0.35 + 0.25 * (v - 3.5) / 1.5
        if 9 < v <= 12:
            return 0.6 - 0.25 * (v - 9) / 3.0
        return 0.25

    # yoe/career-span contradiction: PENALIZE (never rescue into a favorable band). Score the
    # raw stated yoe and apply a harsh multiplier — such profiles are also gated by honeypot
    # signals C/D, so this is belt-and-suspenders for any that slip through.
    if career_years > 0 and abs(y - career_years) > 4:
        return max(0.0, min(band(y) * 0.6, 1.0))
    return max(0.0, min(band(y), 1.0))


def _loc(candidate):
    p = candidate.get("profile", {}) or {}
    rs = candidate.get("redrob_signals", {}) or {}
    loc = (p.get("location") or "").lower()
    country = p.get("country")
    reloc = bool(rs.get("willing_to_relocate"))
    if any(c in loc for c in K.PREFERRED_CITIES):
        return 1.00
    if any(c in loc for c in K.WELCOME_CITIES):
        return 0.80
    if country == "India" and reloc:
        return 0.65
    if country == "India":
        return 0.45
    if country != "India" and reloc:
        return 0.40
    return 0.20


def _base(candidate):
    rs = candidate.get("redrob_signals", {}) or {}
    gh = rs.get("github_activity_score", -1)
    gh_bucket = 1.0 if gh >= 40 else 0.5 if gh >= 15 else 0.0
    val = (0.5 * (rs.get("profile_completeness_score", 0) / 100.0)
           + 0.2 * bool(rs.get("verified_email"))
           + 0.1 * bool(rs.get("verified_phone"))
           + 0.1 * bool(rs.get("linkedin_connected"))
           + 0.1 * gh_bucket)
    return max(0.0, min(val, 1.0))


def _behavioral(candidate):
    rs = candidate.get("redrob_signals", {}) or {}
    la = parse_date(rs.get("last_active_date"))
    days_inactive = (K.RECENCY_ANCHOR - la).days if la else 999
    rr = rs.get("recruiter_response_rate", 0.0) or 0.0
    npd = rs.get("notice_period_days", 90)
    icr = rs.get("interview_completion_rate", 0.0) or 0.0

    f_resp = 1.05 if rr >= 0.60 else 1.00 if rr >= 0.30 else 0.92 if rr >= 0.15 else 0.84
    f_active = 1.04 if days_inactive <= 30 else 1.00 if days_inactive <= 90 else 0.92 if days_inactive <= 180 else 0.85
    f_notice = (1.05 if npd <= 15 else 1.02 if npd <= 30 else 0.98 if npd <= 60 else 0.92 if npd <= 90 else 0.86)
    f_otw = 1.04 if rs.get("open_to_work_flag") else 0.99
    f_follow = 0.97 + 0.06 * icr
    B = f_resp * f_active * f_notice * f_otw * f_follow
    return max(0.80, min(B, 1.12)), days_inactive


def _honeypot(candidate, a_count_expert_zero):
    """
    Impossibility / synthetic-stitch detection. Four rare, false-positive-free signals
    (see analysis/review_fixlist.md). Returns (is_honeypot, reasons).
      A: >=3 expert/advanced skills with 0 months used
      B: a role claiming more tenure than its start/end dates allow (gap >= 6mo)
      C: career-history duration sums to >3y MORE than stated years_of_experience
      D: the summary's stated tenure disagrees with years_of_experience by >4y
    C & D catch the cross-field "junior YoE stitched onto a senior multi-year career" honeypots
    that A & B structurally cannot (each role is internally consistent; the contradiction is
    across fields). Pool-wide counts: A~21, B~33, C~22, D~14 (union well under the 300 guardrail).
    """
    anchor = K.RECENCY_ANCHOR
    p = candidate.get("profile", {}) or {}
    yoe = p.get("years_of_experience")
    ch = candidate.get("career_history", []) or []
    reasons = []
    b_hit = c_hit = d_hit = False

    for r in ch:
        sd = parse_date(r.get("start_date"))
        ed = parse_date(r.get("end_date")) or anchor
        dur = r.get("duration_months")
        if sd is not None and dur is not None:
            elapsed = months_between(sd, ed)
            if elapsed is not None and (dur - elapsed) >= 6:
                b_hit = True
                reasons.append(f"{r.get('title')}@{r.get('company')} claims {dur}mo but dates allow ~{max(elapsed,0)}mo")
    if a_count_expert_zero >= 3:
        reasons.append(f"{a_count_expert_zero} expert/advanced skills with 0 months used")

    if yoe is not None and ch:
        career_years = sum((r.get("duration_months") or 0) for r in ch) / 12.0
        if career_years - yoe > 3.0:
            c_hit = True
            reasons.append(f"career history sums to {career_years:.1f}y but stated experience is {yoe:.1f}y")

    summ = p.get("summary") or ""
    m = _SUMMARY_YEARS_RX.search(summ)
    if yoe is not None and m:
        try:
            stated = float(m.group(1))
            if abs(stated - yoe) > 4:
                d_hit = True
                reasons.append(f"summary states {stated:.0f}y experience but profile field says {yoe:.1f}y")
        except ValueError:
            pass

    # Signal E: >=4 skills each used for MORE than (career + 2y). A few long skills can predate a
    # career (college/hobby), but 4+ — including tools younger than the claim (e.g. Pinecone for 6y
    # by a 3y professional) — is a synthetic stitch. Threshold tuned to the data's clean cliff
    # (exactly-3 over-cap = 54 candidates, but >=4 = only 18): >=4 converges the total honeypot
    # count to ~80 (matching the spec's seeded count) with near-zero genuine-candidate false positives.
    e_hit = False
    if yoe is not None:
        cap = yoe * 12 + 24
        over = sum(1 for s in candidate.get("skills", []) or [] if (s.get("duration_months") or 0) > cap)
        if over >= 4:
            e_hit = True
            reasons.append(f"{over} skills used longer than the candidate's entire career (+2y)")

    is_hp = (a_count_expert_zero >= 3) or b_hit or c_hit or d_hit or e_hit
    return is_hp, reasons


def _traps(candidate, tc, pes, roles, n_relevant_skills):
    """§3.4: compute applicable penalties; return (P_trap, trap_label, title_no_evidence, title_chaser_delta)."""
    p = candidate.get("profile", {}) or {}
    narrative = K.narrative_text(candidate).lower()
    desc_text = K.career_descriptions_text(candidate).lower()
    classes = [r["company_class"] for r in roles]
    max_role_evid = max((r["role_evid"] for r in roles), default=0.0)

    penalties = []  # (penalty, label)

    # KEYWORD_STUFFER
    if tc == "nontech" and n_relevant_skills >= 4 and pes < 0.15:
        penalties.append((0.25, "keyword-stuffer"))
    # SELF_LEARNER_SWE
    if tc == "adjacent" and K.SIDEPROJECT_RX.search(narrative) and max_role_evid < 0.4:
        penalties.append((0.45, "self-learner-side-project"))
    # RECENT_ONLY_LLM
    if K.LANGCHAIN_RECENT_RX.search(narrative):
        recent_evid = any(r["role_evid"] > 0 and r["ended_months_ago"] <= 12 for r in roles)
        older_strong = any(r["role_evid"] >= 0.4 and r["ended_months_ago"] > 12 for r in roles)
        if recent_evid and not older_strong and max_role_evid < 0.5:
            penalties.append((0.50, "recent-only-LLM"))
    # CONSULTING_ONLY (waived if any product/bigtech role)
    if classes and all(c == "consulting" for c in classes):
        penalties.append((0.45, "consulting-only"))
    # CV_SPEECH_ONLY
    if K.CV_SPEECH_RX.search(narrative) and pes < 0.10 and not K.NLP_IR_PRESENT_RX.search(desc_text):
        penalties.append((0.55, "cv/speech-only"))
    # RESEARCH_ONLY
    if K.RESEARCH_ONLY_RX.search(narrative) and not any(r["matched"].get("prod") for r in roles):
        if not K.PROD_RX.search(desc_text):
            penalties.append((0.50, "research-only"))

    if penalties:
        penalties.sort(key=lambda x: x[0])  # harshest (min) first
        P_trap, label = penalties[0]
    else:
        P_trap, label = 1.0, None

    # TITLE_NO_EVIDENCE (special: content cap, handled in scoring)
    title_no_evidence = (tc == "core" and pes < 0.10)
    # TITLE_CHASER (soft additive -0.04)
    comps = [r for r in roles if r["company_class"] in ("product_ai", "bigtech")]
    title_chaser_delta = 0.0
    if len(roles) >= 3 and comps:
        avg_tenure = sum(r["duration_months"] for r in comps) / len(comps)
        if avg_tenure < 18:
            title_chaser_delta = -0.04
    return P_trap, label, title_no_evidence, title_chaser_delta


def compute_features(candidate):
    roles, families_seen = _role_evidence(candidate)
    pes, distinct_ireval = _pes(roles, families_seen)
    T_mult, tc = _title_mult(candidate, pes)
    skc, a_count_expert_zero, n_cred = _skill_corroboration(candidate)
    n_relevant_skills = sum(1 for s in candidate.get("skills", []) or []
                            if K.RELEVANT_SKILL_RX.search(s.get("name") or ""))
    ebf = _ebf(candidate)
    loc = _loc(candidate)
    base = _base(candidate)
    B, days_inactive = _behavioral(candidate)
    is_hp, hp_reasons = _honeypot(candidate, a_count_expert_zero)
    P_hp = 0.02 if is_hp else 1.0
    P_trap, trap_label, title_no_evidence, title_chaser_delta = _traps(
        candidate, tc, pes, roles, n_relevant_skills)

    p = candidate.get("profile", {}) or {}
    rs = candidate.get("redrob_signals", {}) or {}
    best_role = max(roles, key=lambda r: r["role_evid"], default=None)

    return {
        "candidate_id": candidate["candidate_id"],
        "PES": pes, "TFF_pos": T_mult, "T_mult": T_mult, "SKC": skc, "EBF": ebf,
        "LOC": loc, "BASE": base, "B": B, "P_trap": P_trap, "P_hp": P_hp,
        "is_honeypot": is_hp, "trap_label": trap_label,
        "title_no_evidence": title_no_evidence, "title_chaser_delta": title_chaser_delta,
        "title_class": tc, "distinct_ireval": distinct_ireval,
        "n_cred_skills": n_cred, "n_relevant_skills": n_relevant_skills,
        "a_count_expert_zero": a_count_expert_zero, "hp_reasons": hp_reasons,
        "facts": {
            "yoe": p.get("years_of_experience"),
            "current_title": p.get("current_title"),
            "current_company": p.get("current_company"),
            "country": p.get("country"), "location": p.get("location"),
            "best_role": best_role,
            "roles": roles,
            "days_inactive": days_inactive,
            "recruiter_response_rate": rs.get("recruiter_response_rate"),
            "notice_period_days": rs.get("notice_period_days"),
            "willing_to_relocate": rs.get("willing_to_relocate"),
            "open_to_work": rs.get("open_to_work_flag"),
            "github_activity_score": rs.get("github_activity_score"),
            "skills": candidate.get("skills", []),
        },
    }
