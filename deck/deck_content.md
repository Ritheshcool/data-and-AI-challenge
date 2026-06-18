# Redrob Challenge — Idea Submission Deck (content)

Source of truth for the slide text. Rendered to PPTX → PDF. Follows the official 10-slide template.
Numbers marked ⟦…⟧ are confirmed from the final run.

---

## Slide 1 — Title
- **Team Name:** ⟦TEAM_NAME⟧
- **Problem Statement:** Intelligent Candidate Discovery & Ranking — rank the top-100 of 100,000 candidates for a *Senior AI Engineer (Founding Team)* JD, the way a great recruiter would: by understanding fit, not matching keywords.
- **Team Leader:** ⟦LEADER_NAME⟧

---

## Slide 2 — Solution Overview
**What we built:** a hybrid ranker = a transparent **rule layer** (what a recruiter actually reasons about) + a **local semantic-embedding** signal, combined into one explainable score per candidate, run over all 100K in **~1.6 seconds on CPU, offline**.

**What differentiates it from traditional keyword matching:**
- **Evidence over keywords.** The backbone signal reads *career-history descriptions* (what they actually built), not the skills list. "Vector DB" appears in 12,866 skill lists but only 108 career descriptions — we weight the credible source.
- **A role-class gate** makes a "Marketing Manager with 11 AI skills" structurally unable to reach the top — the exact trap the JD describes.
- **Behavioral availability** is a first-class multiplier: a perfect-on-paper candidate who's 206 days idle with a 7% response rate is *not actually hireable*, and ranks accordingly.
- **Impossible profiles are gated out.** Five consistency checks exclude the ~80 honeypots.
- **Every rank is explained** in plain, fact-grounded language — no LLM, no hallucination.

---

## Slide 3 — JD Understanding & Candidate Evaluation
**Key requirements extracted from the JD (it means more than it says):**
- Production **embeddings retrieval + vector DB/hybrid search + learning-to-rank/recsys** at *product* companies (not services, not pure research).
- **5–9 yrs** (ideal 6–8), ~4–5 in applied ML; has *shipped* a search/ranking/recsys system at scale.
- Ranking-eval rigor (NDCG/MRR/MAP, A/B). Strong Python.
- **Disqualifiers:** pure research w/o production; recent-only LangChain-on-OpenAI; "architect" who stopped coding.
- **Not wanted:** title-chasers, framework-tutorial people, pure-consulting careers (TCS/Infosys/Wipro/…), CV/speech/robotics w/o NLP/IR.
- **Logistics:** Pune/Noida or willing-to-relocate (Hyd/Mumbai/Delhi-NCR/Bangalore); outside India = no visa sponsorship; sub-30-day notice preferred.

**Signals that matter most (in order):** production-evidence in career descriptions ≫ genuine title/trajectory ≫ corroborated skills ≫ semantic JD fit ≫ experience band ≫ behavioral availability (as a multiplier) ≫ location.

**Fit beyond keywords:** we score the *gap between what the JD says and means* — a Tier-5 builder who never writes "RAG" still ranks high if their career shows a production recsys at a product company; a keyword-packed non-engineer does not.

---

## Slide 4 — Ranking Methodology
**Retrieve → score → rank, in one vectorized pass (no per-candidate LLM):**

```
content = 0.46·PES + 0.10·EMB + 0.14·TitleFit + 0.10·Skills + 0.08·Experience + 0.04·Location + 0.08·Base
gated   = (content + 0.08·TitleFit·PES) · (0.35 + 0.65·RoleClassMultiplier)
final   = gated · TrapPenalty · BehavioralAvailability · HoneypotGate
```

| Component | Wt | What it measures |
|---|---|---|
| **PES** (Production-Evidence Score) | 0.46 | retrieval/ranking/recsys/RAG/eval language in career *descriptions*, weighted by company type, role duration, recency |
| **EMB** | 0.10 | cosine of a `bge-small-en-v1.5` embedding vs a JD-intent vector (calibrated 0.66→0.80; genuine fits sit above pool p99) |
| **TitleFit + Role-class gate** | 0.14 + ×gate | core AI title ⇒ ×1.0; non-tech ⇒ ×0.06 (keeps only ~39% of content) |
| **Skills** | 0.10 | relevant skills, trust-weighted by endorsements+duration+proficiency; expert-at-0-months = red flag |
| **Experience / Location / Base** | 0.08/0.04/0.08 | band fit (6–8y), Pune-Noida-relocate, completeness/verification/GitHub |
| **Behavioral multiplier** | ×[0.80,1.12] | response rate, recency, notice, open-to-work, interview-completion |
| **Trap penalty / Honeypot gate** | × | demote stuffers/self-learners/consulting/CV-speech/research-only; exclude impossible profiles |

**Models/algorithms:** local sentence-embeddings (bge-small) + numpy cosine (no vector DB needed for one query over 100K) + transparent weighted heuristics. **No learning-to-rank training** — by design: explainable and defensible, no labels available.

**Combining signals:** additive content for graded factors, **multiplicative** for gates (role-class, availability, honeypot) so a single disqualifier can't be out-voted by keyword volume.

---

## Slide 5 — Explainability & Data Validation
**How decisions are explained:** each candidate gets a 1–2 sentence reason generated **programmatically** from its own scored features — citing real facts (years, title, company, the strongest evidence phrase, top corroborated skills), naming the matched JD requirement, and disclosing up to two honest concerns. Tone is gated by fit so it always matches the rank.

**Preventing hallucination:** the generator only fills slots from fields present in the record; a post-generation assertion confirms every named skill/employer exists in the profile; near-duplicate strings are re-rolled. **0 hallucinated mentions** across the top-100.

**Handling inconsistent / low-quality / suspicious profiles — 5 impossibility checks (gate to bottom, exclude from top-100):**
- A: ≥3 expert/advanced skills with 0 months used
- B: a role claiming more tenure than its dates allow
- C: career durations summing to >3y more than stated experience
- D: summary tenure disagreeing with the experience field by >4y
- E: ≥3 skills used longer than the entire career (+2y)

Deliberately **ignored as synthetic noise:** salary `min>max` (~19% of pool) and any single skill-duration overrun (~9%) — using them would bury genuine candidates.

---

## Slide 6 — End-to-End Workflow
```
JD ──▶ JD-intent vector (offline, once)
                                   │
candidates.jsonl ──▶ [PRECOMPUTE, offline]  ── bge-small embeddings (100K×384)  ─┐
                  └─ rule-feature cache (PES, title, skills, behavioral, traps) ─┤
                                                                                 ▼
                              [RANK, ≤5 min CPU, no network]
        load cached vectors+features ▶ combine score ▶ honeypot exclude
        ▶ top-10 safety lock ▶ sort + tie-break ▶ generate reasoning ▶ submission.csv
```
The expensive embedding work is one-time/offline; the judged ranking step only reads cached artifacts.

---

## Slide 7 — System Architecture
*(diagram slide — see architecture.svg)* Two stages:
- **Offline precompute** (`precompute.py`): bge-small encodes JD + candidates → `cand_embeddings.npy`; rule features → `features.npz`. Network/time allowed.
- **Online ranking** (`rank.py`): pure numpy + cached artifacts. Modules: `features` (per-candidate signals + honeypot/trap), `scoring` (weighting, top-10 lock, validator-safe emission), `reasoning` (grounded NL). CPU-only, offline, deterministic.

---

## Slide 8 — Results & Performance
**Ranking quality (top-100):**
- **100/100** are genuine core AI/ML/NLP/DS titles; **0** non-technical, **0** trap-penalized profiles.
- **0 honeypots** in the top-100 (⟦N_HP⟧ flagged pool-wide of the ~80 seeded; well under the 10% DQ line).
- Companies are all product/AI/big-tech (Ola, Zomato, Flipkart, Paytm, Razorpay, Meta, Google…); 90/100 India-based.
- Behavioral down-weighting verified: a PES-0.98 but 206-day-idle / 7%-response candidate correctly sinks to the tail.
- Reasoning: 0 hallucinations, all distinct, rank-consistent.

**Performance vs the challenge constraints:**
| Constraint | Limit | Ours |
|---|---|---|
| Ranking runtime | ≤ 5 min | **~1.6 s** |
| Memory | ≤ 16 GB | < 1 GB |
| Compute | CPU only | CPU only |
| Network during ranking | none | none (cached vectors) |

---

## Slide 9 — Technologies Used
- **Python 3.10**, **numpy** (vectorized scoring + cosine), **scikit-learn** utilities.
- **sentence-transformers + PyTorch (CPU)** with **`BAAI/bge-small-en-v1.5`** — small, fast, strong retrieval embeddings; used *only offline*.
- **No vector DB / FAISS at rank time** — a single query over 100K vectors is one matmul; an ANN index would add dependency + non-determinism for no benefit.
- Why these: maximize quality within a hard CPU/offline/5-min budget, keep everything **deterministic, explainable, and defensible** for the code-reproduction and interview stages.

---

## Slide 10 — Submission Assets
- **GitHub repo:** ⟦REPO_URL⟧ (rank.py, precompute.py, redrob_ranker/, tests/, README, requirements, submission_metadata.yaml)
- **Reproduce:** `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`
- **Sandbox/demo:** ⟦SANDBOX_URL⟧ (runs the ranker on a ≤100-candidate sample)
- **Ranked output:** `submission.csv` (top-100)
- **Walkthrough video:** ⟦VIDEO_URL⟧
