# Redrob — Intelligent Candidate Discovery & Ranking

A hybrid **(local embeddings + transparent rule layer)** ranker that scores 100,000 candidate
profiles against the released *"Senior AI Engineer — Founding Team"* job description and emits the
top-100 as `submission.csv` (`candidate_id,rank,score,reasoning`).

It is built to do what the JD explicitly asks: **rank the way a great recruiter would — by reading
who actually fits, not by counting AI keywords.** The dataset is seeded with keyword stuffers,
self-learner "side-project" engineers, plain-language gems, behavioral twins, and ~80 impossible
honeypots; the design defeats each one (see [analysis/design_spec.md](analysis/design_spec.md)).

## Reproduce the submission

```bash
# 1. (offline, once — network + time allowed) embed JD + all candidates with bge-small-en-v1.5
python precompute.py --candidates ./candidates.jsonl

# 2. (the ranking step — CPU-only, NO network, < 60s, < 1GB RAM) produce the CSV
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

For bit-identical reruns (already defaulted inside `rank.py`):
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`rank.py` loads **only** the cached embedding matrix (`artifacts/`) — it never loads a model and
never touches the network, so it satisfies the Stage-3 sandbox constraints (≤5 min, ≤16 GB, CPU-only,
offline) by construction. Pre-computed embeddings are the only precompute dependency, which the
challenge spec explicitly permits.

## How it works (one paragraph)

Every candidate gets a score in roughly `[0,1]`:

```
content   = 0.46·PES + 0.10·EMB + 0.14·Title + 0.10·Skills + 0.08·Experience + 0.04·Location + 0.08·Base
gated     = (content + 0.08·Title·PES) · (0.35 + 0.65·RoleClassMultiplier)
final     = gated · TrapPenalty · BehavioralAvailability · HoneypotGate
```

- **PES (Production-Evidence Score, weight 0.46)** — the backbone. Reads the candidate's *career
  descriptions* (the least-gameable field: "vector db" appears in 12,866 skill lists but only 108
  descriptions) for production retrieval / ranking / recsys / RAG / eval language, weighted by company
  type, role duration, and recency.
- **EMB (0.10)** — cosine of a `bge-small-en-v1.5` embedding of the candidate narrative against a
  hand-authored JD intent vector. Skills are deliberately **excluded** from the embedded text so
  keyword stuffing can't inflate it.
- **Role-class gate** — a non-technical current title (Marketing Manager, Graphic Designer…) keeps
  only ~39% of its content, so a keyword-packed profile structurally cannot reach the head.
- **Trap penalties** — keyword stuffer, self-learner side-project, recent-only-LLM, consulting-only,
  CV/speech-only, research-only (single harshest applied).
- **Behavioral availability multiplier `[0.80, 1.12]`** — response rate, recency, notice period,
  open-to-work, interview-completion: separates "behavioral twins" and down-weights perfect-on-paper
  but unreachable candidates.
- **Honeypot gate** — four rare, false-positive-free impossibility checks; flagged profiles are
  excluded from the top-100:
  - **A** ≥3 expert/advanced skills with 0 months used,
  - **B** a role claiming more tenure than its start/end dates allow (gap ≥ 6 months),
  - **C** career-history durations summing to >3y *more* than the stated `years_of_experience`,
  - **D** the summary's stated tenure disagreeing with `years_of_experience` by >4y.

  A & B catch within-role impossibilities; **C & D catch the cross-field "junior YoE stitched onto
  a senior multi-year career" honeypots** (each role is internally consistent, so the contradiction
  only shows across fields). Together they flag ~60–75 of the ~80 honeypots. (Salary `min>max` and
  `skill_duration>career` are pervasive synthetic *noise* — present in 10–20% of the pool — and are
  deliberately *not* used.)

Reasoning strings are generated **programmatically** (no LLM): fact-grounded, varied, tone-matched to
fit, and asserted to mention only skills/employers actually present in the profile.

## Repository layout

```
rank.py                     # ONLINE ranking step (the single reproduce command)
precompute.py               # OFFLINE embedding of JD + candidates -> artifacts/
redrob_ranker/
  knowledge.py              # JD/data-derived vocabularies (companies, cities, titles, phrase regexes)
  io_utils.py               # streaming jsonl loader + date math
  features.py               # per-candidate features (PES, Title, Skills, Experience, Location, Base, B, traps, honeypot)
  scoring.py                # weight combination, top-10 lock, validator-safe score emission
  reasoning.py              # deterministic fact-grounded reasoning generator
analysis/                   # data profiling, the synthesized design spec, debug dumps
artifacts/                  # cand_embeddings.npy, jd_embedding.npy, cand_ids.json, precompute_meta.json
tests/                      # pre-submit format + quality checks
requirements.txt, submission_metadata.yaml
```

## Constraints met
| Constraint | This system |
|---|---|
| ≤ 5 min runtime | rank step < 60s |
| ≤ 16 GB RAM | < 1 GB (fp32 upcast of the 100k×384 matrix ≈ 150 MB) |
| CPU only | no GPU; `CUDA_VISIBLE_DEVICES=""` |
| No network during ranking | no model load, no API — only cached `.npy` |
| ≤ 5 GB disk | artifacts < 200 MB |
