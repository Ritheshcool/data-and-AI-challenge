# Redrob "Senior AI Engineer â€” Founding Team" Ranking System â€” Final Scoring Design

**One JD, 100,000 candidates, emit top-100 CSV `candidate_id,rank,score,reasoning`. Scored by `0.50Â·NDCG@10 + 0.30Â·NDCG@50 + 0.15Â·MAP + 0.05Â·P@10` against hidden tiers 0â€“5 (relevant = tier 3+; ~80 forced-tier-0 honeypots).**

This synthesizes the four lenses (metrics, recruiter-judgment, robustness, engineering) into one implementable spec. Every conflict is resolved with a single decision rule, stated inline.

---

## AS-BUILT DELTAS (authoritative — code wins over the design narrative below)

A few values were corrected during implementation/review; the **code is the source of truth**. Differences from the narrative in later sections:

- **EMB calibration:** the narrative shows `clip((cos-0.20)/0.35)`. As-built is **`clip((cos-0.66)/0.14)`** (`EMB_FLOOR=0.66, EMB_CEIL=0.80` in `rank.py`). Reason: the actual `bge-small-en-v1.5` cosine distribution on this pool is compressed high — pool p50/p90/p99 = 0.643/0.669/0.689, while the known gold fits sit at 0.717–0.836 (all above pool p99). The 0.20/0.55 band saturated everyone to 1.0; 0.66/0.80 is the discriminative band.
- **Honeypot signals:** the narrative implements only A (expert-zero-duration) and B (duration-vs-dates). As-built adds **C** (career-sum − yoe > 3y) and **D** (|summary-stated-years − yoe| > 4) to catch the cross-field "junior-YoE-on-senior-career" stitches A/B structurally miss. Pool-wide union ≈ 60–75 flags (still ≪ the 300 over-flag guardrail). EBF's consistency case now **penalizes** (×0.6) rather than rescues such contradictions.
- **Artifacts:** as-built are `cand_embeddings.npy` (**float32**, ~153MB), `cand_ids.json`, `jd_embedding.npy`, `precompute_meta.json`, and **`features.npz`** (float64 rule-feature cache, <1MB) — not `features.parquet`/fp16.
- **Runtime:** the *ranking step* (`rank.py` with the cached artifacts) is **~1.6s** (well under 5 min). One-time offline precompute (embeddings) is ~25–35 min on CPU; the feature cache is ~3 min. Live-recompute fallback (no cache) is ~180s, still within budget.
- **Reasoning:** emits up to **two** honest concerns (not one), keyed including a raw-YoE out-of-band check, and splices one grounded concrete detail from the best role's description.

---

## 0. Design thesis & conflict resolutions

The four designs agreed on the spine; they differed on knobs. The resolutions adopted here:

| Conflict | Decision | Why |
|---|---|---|
| Score 0â€“1 squashed vs 0â€“100 vs linear | **Single linear score in `[0,1]`, no sigmoid.** Argsort is all the metrics consume; calibration sugar adds Stage-5 attack surface. | Transparency for the defend interview. |
| Score everyone vs recall-prefilter | **Score all 100k in one vectorized pass** (rule features are cheap). Embedding similarity is a *feature*, never a gate/prefilter. | Removes "recall miss" risk entirely; still well under budget. |
| Embedding weight (0.07 / 0.10 / 0.30) | **0.10**, and only on the `[0,1]` content scale, capped so it can never lift a gated profile. | Metrics-optimizer's 0.30 over-trusts the gamed surface; robustness lens's 0.03 wastes the gem-rescue signal. 0.10 is the middle that still rescues plain-language gems. |
| Title as multiplier vs additive | **Both**: small additive + a multiplicative role-class gate `(0.35 + 0.65Â·T)`. Non-tech titles lose ~60% structurally. | This is the marquee anti-stuffer lever; needs teeth. |
| Honeypot: penalize vs gate | **Hard gate to a sub-floor (`scoreâ‰ˆ0.02`), applied last, plus excluded from top-100.** | Stage-3 disqualifies at >10% honeypot rate; we aim for 0%. |
| Behavioral band width | **Multiplier `B âˆˆ [0.80, 1.12]`.** | Wide enough to separate twins, narrow enough that an available stuffer (`B=1.12`) never beats a real builder (whose content is multiples higher). |
| Which honeypot signals | **ONLY the two rare/strong ones** (expert-zero-duration; duration-vs-dates). Salary `min>max` (~19%) and `skill_dur>career` (~9%) are pervasive synthetic noise â€” **never** used. | Using noise checks would gate ~25% of the pool including golds. |

**Fixed deterministic recency anchor: `ANCHOR = 2026-06-01`** (max `last_active_date` in pool). No clock, no network.

---

## 1. Relevance/fit model & score composition

### 1.1 Internal tier model (target the argsort must reproduce â€” NOT emitted)

- **Tier 5** (target ranks ~1â€“15, score â‰¥ 0.85): 6â€“8y, CORE AI/ML/NLP/DS/Applied-Scientist title, career **descriptions** show production retrieval/ranking/recsys/RAG/semantic-search/LTR at a named product/AI/big-tech co, eval-framework language (NDCG/MRR/A-B), strong availability. Prototypes: `CAND_0077337`, `CAND_0008425`, `CAND_0018499`, `CAND_0080766`.
- **Tier 4** (~10â€“50, 0.72â€“0.85): same core production fit, one real friction (abroad/no-relocate, 90â€“120d notice, low response/long inactivity, yoe out of band e.g. `CAND_0039754` 16.2y, or recsys-heavy & eval-light). E.g. `CAND_0092278` (strong, 7% response, 206d idle), `CAND_0055905` (London).
- **Tier 3** (relevant floor, 0.55â€“0.72): genuine applied-ML/DS at a product co with adjacent production evidence but missing a pillar; OR a plain-language gem with generic-engineer title but descriptions clearly show production ranking (`CAND_0080766`-style "Search & Discovery").
- **Tier 1â€“2** (0.25â€“0.55): partial fits to down-weight â€” self-learner SWEs (RAG side project), consulting-only, CV/speech/robotics w/o NLP/IR, research-only, generic-churn-MLOps "AI" people.
- **Tier 0** (gated, â‰¤0.05): ~80 honeypots + keyword-stuffers.

### 1.2 Final score composition (overview; exact numbers in Â§EXACT WEIGHTS)

```
content      = 0.46Â·PES + 0.10Â·EMB + 0.14Â·TFF_pos + 0.10Â·SKC + 0.08Â·EBF + 0.04Â·LOC + 0.08Â·BASE   # all terms in [0,1]
synergy      = 0.08 Â· TFF_pos Â· PES                                                               # title Ã— evidence reinforcement
gated        = (content + synergy) Â· (0.35 + 0.65Â·T_mult)                                          # role-class gate
penalized    = gated Â· P_trap                                                                      # softest single trap penalty (min)
available    = penalized Â· B                                                                       # behavioral availability multiplier
final        = available Â· P_hp                                                                    # honeypot hard gate (0.02 or 1.0)
```

`final` lies in roughly `[0, 1.0]`. **Components (`PES,EMB,TFF_pos,SKC,EBF,LOC,BASE`) are each normalized to `[0,1]`** so weights are interpretable; `T_mult, P_trap, B, P_hp` are multipliers. All inputs are precomputed fields â€” one vectorized numpy pass over 100k.

---

## EXACT WEIGHTS & FORMULAS (implement directly)

> All regex matching is **case-insensitive on lowercased text**, precompiled. `clip(x,lo,hi)` clamps. Career-description text = concatenation of all `career_history[i].description`. "named co" lists in Â§3.3.

### A. Production-Evidence Score `PES âˆˆ [0,1]` â€” weight **0.46** (the backbone)

Computed from **career descriptions only** (the least-gameable field: "vector db" appears in 12,866 skill lists but only 108 descriptions).

Per role `r` with description `d_r`, title `t_r`, company `c_r`, `duration_months m_r`:

```
hits_retrieval_r = 1 if d_r matches RETRIEVAL_RX else 0
hits_ranking_r   = 1 if d_r matches RANKING_RX   else 0
hits_recsys_r    = 1 if d_r matches RECSYS_RX    else 0
hits_rageval_r   = 1 if d_r matches RAGEVAL_RX   else 0
hits_prod_r      = 1 if d_r matches PROD_RX      else 0

role_raw_r = clip(0.45Â·hits_retrieval_r + 0.40Â·hits_ranking_r + 0.35Â·hits_recsys_r
                 + 0.30Â·hits_rageval_r + 0.20Â·hits_prod_r, 0, 1.0)

company_mult_r = 1.05 if c_r in BIGTECH
                 1.00 if c_r in PRODUCT_AI
                 0.85 if c_r in FICTIONAL_FILLER or unknown
                 0.55 if c_r in CONSULTING
dur_w_r     = clip(m_r / 24, 0, 1)
recency_w_r = 1.00 if role ended within 60 months of ANCHOR else 0.60   # most recent ~5y full weight
role_evid_r = role_raw_r Â· company_mult_r Â· dur_w_r Â· recency_w_r

PES_raw = 1 - Î _r (1 - 0.60Â·role_evid_r)        # diminishing-returns OR across roles
PES     = clip(PES_raw + 0.05Â·min(distinct_ireval_families_in_descriptions, 2)/2, 0, 1)
```

The diminishing-returns OR lets multiple strong roles compound but saturate, so a single 6-month cameo cannot fake a top score. `distinct_ireval_families` bonus rewards breadth (retrieval+ranking+eval together) over keyword repetition.

**Phrase lists** (families; broad so plain-language gems fire):
```
RETRIEVAL_RX = semantic search | hybrid retrieval | dense (and|\+|/) ?sparse | bm25 | dense vector | vector recall
             | embedding-based (search|retrieval) | faiss | hnsw | nearest[- ]neighbor | bge | sentence[- ]transformer
             | index refresh | embedding drift
RANKING_RX   = learning[- ]to[- ]rank | \bltr\b | ranking (layer|pipeline|model|system) | re[- ]rank
             | xgboost|lightgbm | relevance (labeling|tuning) | candidate sourcing
RECSYS_RX    = recommendation system | recommender | collaborative filtering | matrix factorization
             | discovery feed | personalization | content[- ]based ranking | item[- ]item
RAGEVAL_RX   = \brag\b | retrieval[- ]augmented | ndcg | \bmrr\b | \bmap\b | a/?b test | offline[- ]to[- ]online
             | eval (framework|harness|infrastructure) | offline (experiment|benchmark)
PROD_RX      = shipped | deployed | in production | serving \d | \bqps\b | p95 | latency | 50m\+|10m\+|35m\+|million
             | live a/?b | rolled out | end[- ]to[- ]end
```

### B. Semantic-embedding component `EMB âˆˆ [0,1]` â€” weight **0.10**

- **Model:** `BAAI/bge-small-en-v1.5` (384-d, ~67MB fp16, CPU). Chosen over MiniLM (weaker MTEB retrieval) and e5-small (needs prefix juggling); bge natively supports the asymmetric query instruction for one-query/many-docs. Shipped as a **pinned local snapshot / ONNX export** in the repo (no download at runtime).
- **Candidate text embedded (OFFLINE):** `headline + " " + summary + " " + all career_history[].description`, truncated to 512 tokens, mean-pooled, L2-normalized. **Skills list is deliberately EXCLUDED** â€” it is the gamed surface; embedding it inflates stuffers.
- **JD text embedded (OFFLINE, once):** a hand-authored ~120-word *intent* string prefixed with bge's query instruction:
  > *"Represent this sentence for searching relevant passages: Senior applied ML/AI engineer, 6â€“8 years at product companies, who shipped production embeddings-based retrieval, hybrid dense+sparse search, vector databases, learning-to-rank and recommendation systems to real users at scale; handled embedding drift, index refresh, retrieval-quality regression; designed ranking evaluation with NDCG, MRR, MAP, offline-to-online correlation and A/B tests; tilts shipper over researcher."*
- **Combination:** cosine = single matmul of the `(100000, 384)` fp16 matrix (upcast to fp32 in RAM, ~150MB) against the `(384,)` query. Then:
  ```
  EMB = clip((cosine - 0.20) / (0.55 - 0.20), 0, 1)
  ```
  The 0.20 floor strips the baseline similarity even off-domain templated profiles share; 0.55 â‰ˆ a clearly on-domain profile. Both percentiles are **calibrated offline on the 25 gold + 100 random candidates and frozen as constants.**

**How keyword-stuffer summaries are prevented from scoring high on cosine â€” three independent locks:**
1. We embed **career descriptions** (which for a stuffer are about graphic-design/operations) far more than the summary; their doc cosine is *low*, not high.
2. `EMB` weight is **0.10** and on the content scale only â€” it cannot outweigh `PES` (0.46) or survive the title gate.
3. The keyword-stuffer trap (`P_trap=0.25`) and role-class gate (`T_mult=0.06 â†’ Â·0.39`) fire regardless of cosine. Even `EMB=1.0` cannot rescue a non-tech-title profile with `PESâ‰ˆ0`.

### C. Title / Role-Class â€” additive `TFF_pos âˆˆ [0,1]` (weight **0.14**) + multiplier `T_mult âˆˆ [0.06,1.0]`

Match `current_title` (weight 2Ã—) and recent `career_history[].title` against three sets:

```
CORE     = senior/staff/lead/principal? (ai|ml|machine learning|nlp|applied) engineer | applied (ml )?scientist
         | data scientist | (search|ranking|recommendation[s]?) engineer | ai specialist | ai engineer
ADJACENT = software|backend|cloud|devops|frontend|full ?stack|data|analytics|mobile engineer | developer
NONTECH  = business analyst | hr manager | (marketing|operations|project) manager | accountant | sales
         | content writer | graphic designer | (civil|mechanical) engineer | customer support | qa engineer

T_mult = 1.00  if current_title âˆˆ CORE
         0.78  if current_title âˆˆ ADJACENT AND a PRIOR role âˆˆ CORE
         0.62  if current_title âˆˆ ADJACENT AND PES â‰¥ 0.40            # plain-language-gem path
         0.30  if current_title âˆˆ ADJACENT AND PES < 0.40
         0.06  if current_title âˆˆ NONTECH                            # hard anti-stuffer ceiling
Junior ("junior ...") caps T_mult at 0.70.

TFF_pos = T_mult                                                     # reuse (0..1), enters additive block
```

Applied as the gate `(0.35 + 0.65Â·T_mult)`: a NONTECH profile keeps only `0.35 + 0.65Â·0.06 = 0.39` of content; a CORE profile keeps 1.00. The `+0.08Â·TFF_posÂ·PES` synergy term rewards *title AND evidence together* (the founding-team signal).

### D. Skill-Corroboration `SKC âˆˆ [0,1]` â€” weight **0.10** (corroboration only)

Skills are the most-gamed field; credited only when corroborated. Relevant-skill families: vector DB (Pinecone/Weaviate/Qdrant/Milvus/pgvector/FAISS/Elasticsearch/OpenSearch), embeddings, sentence-transformers, semantic search, BM25, information retrieval, learning-to-rank, recommendation systems, RAG, NDCG/MRR.

```
for each relevant skill s with proficiency p_s, endorsements e_s, duration_months du_s:
    if p_s âˆˆ {expert, advanced} and du_s == 0:           # HONEYPOT red flag
        credit_s = 0 ; raise expert_zero_flag
    else:
        prof   = {expert:1.0, advanced:0.85, intermediate:0.5, beginner:0.25}[p_s]
        dur    = 1.0 if du_sâ‰¥12 else 0.7 if du_sâ‰¥6 else 0.45 if du_sâ‰¥1 else 0.15
        endor  = 0.5 + 0.5Â·min(e_s/15, 1)
        assess = +0.10 if skill_assessment_scores[s] â‰¥ 70 else 0
        credit_s = profÂ·durÂ·endor + assess

n_cred = number of relevant skills with credit_s â‰¥ 0.5, CAPPED AT 6      # stuffing 11 terms â‰¤ 6 counts
SKC = clip(0.5Â·log1p(n_cred)/log1p(6) + 0.5Â·min(relevant_assessment_avg/80, 1), 0, 1)
```
Sublinear (`log`) count so packing skills barely moves `SKC`; expert-at-0-duration contributes zero AND flags the honeypot detector.

### E. Experience-Band Fit `EBF âˆˆ [0,1]` â€” weight **0.08**

```
y = years_of_experience
EBF = 1.00                       if 6 â‰¤ y â‰¤ 8
      0.6 + 0.4Â·(y-5)            if 5 â‰¤ y < 6
      0.6 + 0.4Â·(9-y)           if 8 < y â‰¤ 9
      0.35 + 0.25Â·(y-3.5)/1.5   if 3.5 â‰¤ y < 5
      0.6 - 0.25Â·(y-9)/3        if 9 < y â‰¤ 12
      0.25                       otherwise (never 0 â€” JD allows strong out-of-band)
Consistency check: if |y - sum(career duration_months)/12| > 4, use min(y, career_years) and apply EBF Â·= 0.9
```
Keeps `CAND_0039754` (16.2y Applied Scientist) dinged but rankable; `CAND_0005260` (5.2y Netflix gem) barely touched.

### F. Location `LOC âˆˆ [0,1]` â€” additive weight **0.04** (see also Â§5)

```
LOC = 1.00 if location âˆˆ {Pune, Noida}
      0.80 if location âˆˆ {Hyderabad, Mumbai, Delhi, Delhi NCR, Gurgaon, Bangalore}
      0.65 if country==India and willing_to_relocate
      0.45 if country==India and not willing_to_relocate
      0.40 if country!=India and willing_to_relocate            # case-by-case, no visa sponsorship
      0.20 if country!=India and not willing_to_relocate
```
City match = lowercased substring against a dictionary (handles "Delhi NCR", "Gurgaon", "Noida, Uttar Pradesh"). Soft â€” never excludes an elite abroad fit (`CAND_0055905` London still ranks if content is elite).

### G. Base-Quality `BASE âˆˆ [0,1]` â€” weight **0.08** (small completeness reward)

```
BASE = clip(0.5Â·(profile_completeness_score/100)
          + 0.2Â·(verified_email) + 0.1Â·(verified_phone) + 0.1Â·(linkedin_connected)
          + 0.1Â·(github_activity_scoreâ‰¥40 ? 1 : (github_activity_scoreâ‰¥15 ? 0.5 : 0)), 0, 1)
# github == -1  -> contributes 0 here, NEVER negative (half the pool has no GitHub)
```

### H. Behavioral Availability multiplier `B âˆˆ [0.80, 1.12]` (see Â§4 for shape)

### I. Trap penalty `P_trap âˆˆ [0.25, 1.0]` (see Â§3.4) and Honeypot gate `P_hp âˆˆ {0.02, 1.0}` (see Â§6)

---

## 2. Semantic-embedding component â€” consolidated

Covered in **Â§EXACT WEIGHTS B**. Summary of the contract:

- **Model:** bge-small-en-v1.5, fp16 matrix `(100000Ã—384)` â‰ˆ 73MB on disk, shipped as an artifact. Rank-time op = **one matmul + scalar arithmetic**, milliseconds. **No FAISS** (one query over 100k vectors does not justify an ANN index's dependency/non-determinism).
- **Embed for JD:** the frozen intent string above (query instruction prefix).
- **Embed for candidate:** headline + summary + all descriptions, **never the skills list**.
- **Combine:** `EMB = clip((cosâˆ’0.20)/0.35, 0,1)`, weight 0.10 on the `[0,1]` content scale.
- **Anti-stuffer:** the three locks in Â§B (embed descriptions not skills; 0.10 cap; gate+trap dominate).

---

## 3. Rule-layer logic â€” consolidated

### 3.1 Title/role fit â€” Â§EXACT WEIGHTS C.

### 3.2 Production-evidence detection â€” Â§EXACT WEIGHTS A (phrase lists inline).

### 3.3 Company lists (lowercased substring match)

```
PRODUCT_AI   = paytm, ola, zomato, flipkart, razorpay, swiggy, cred, freshworks, meesho, dream11, nykaa,
               inmobi, zoho, vedantu, byju, upgrad, sarvam, mad street den, glance, yellow.ai, verloop,
               haptik, rephrase, krutrim, locobuzz, saarthi, aganitha, observe.ai, niramai, wysa, genpact ai
BIGTECH      = amazon, microsoft, google, meta, netflix, uber, linkedin, salesforce, apple, adobe
CONSULTING   = tcs, infosys, wipro, accenture, cognizant, capgemini, hcl, tech mahindra, mphasis, mindtree
FICTIONAL_FILLER = wayne, initech, pied piper, globex, acme, dunder mifflin, stark, hooli
```
Note: fictional filler companies are the synthetic noise carriers; they get `company_mult=0.85` (neutral-unknown), not a bonus, so a strong description at a fake co is discounted vs the same description at Paytm/Amazon.

### 3.4 Disqualifier / trap penalties â€” `P_trap`

Compute each applicable penalty; **apply only the single MINIMUM (harshest) one** (no double-jeopardy that would distort reasoning tone):

```
KEYWORD_STUFFER : current_title âˆˆ NONTECH AND (#relevant skill areas â‰¥ 4) AND PES < 0.15      -> 0.25
SELF_LEARNER_SWE: current_title âˆˆ ADJACENT AND SUMMARY_RX_SIDEPROJECT AND no STRONG/MED PES role -> 0.45
RECENT_ONLY_LLM : domain evidence ONLY in roles ending within 12mo AND mentions langchain/openai
                  AND NO older role with PES role_evid â‰¥ 0.4                                   -> 0.50
CONSULTING_ONLY : EVERY career company âˆˆ CONSULTING (no product/AI/bigtech role ever)          -> 0.45
                  (WAIVED entirely if any prior product-co role exists â€” JD carve-out)
CV_SPEECH_ONLY  : CV/speech/robotics tokens dominate skills+descriptions AND RETRIEVAL/RANKING/NLP-IR
                  evidence absent (PES < 0.10)                                                  -> 0.55
RESEARCH_ONLY   : titles/desc are academic/research-only AND zero PROD_RX hits anywhere         -> 0.50
TITLE_NO_EVIDENCE: current_title âˆˆ CORE AND PES < 0.10  -> cap content at its 60th-percentile value (not a multiplier)
TITLE_CHASER    : â‰¥3 companies, avg product-co tenure < 18 months  -> additive âˆ’0.04 to content (soft)
otherwise P_trap = 1.0
```

```
SUMMARY_RX_SIDEPROJECT = side[- ]project | personal project | hobby project | online course[s]?
   | tutorial | experimenting with | haven'?t done .* (in a )?professional | not .* professional
   | built a small (rag|llm) | exploring | learning .* in my spare time | transitioning
CV_SPEECH_TOKENS = opencv | yolo | object detection | image classification | \bcnn\b | diffusion
   | \bgan[s]?\b | asr | tts | speech recognition | robotics
```
**Note:** golds legitimately carry a few CV/speech skills as noise (e.g. `CAND_0080766` has OpenCV/YOLO/ASR). `CV_SPEECH_ONLY` fires **only** when CV/speech is primary AND IR/ranking evidence is absent â€” so it never touches a real retrieval fit.

### 3.5 Skill trust-weighting â€” Â§EXACT WEIGHTS D (endorsements + duration + proficiency; expert-with-0-duration â†’ zero credit + honeypot flag).

### 3.6 Experience-band fit â€” Â§EXACT WEIGHTS E.

---

## 4. Behavioral availability MULTIPLIER `B`

Multiplicative, bounded, applied **after** content scoring so it re-orders within a fit tier but can never substitute for fit. Inputs from `redrob_signals`; `days_inactive = (ANCHOR âˆ’ last_active_date).days`.

```
f_resp  = 1.05 if recruiter_response_rate â‰¥ 0.60
          1.00 if â‰¥ 0.30
          0.92 if â‰¥ 0.15
          0.84 if < 0.15
f_active= 1.04 if days_inactive â‰¤ 30
          1.00 if â‰¤ 90
          0.92 if â‰¤ 180
          0.85 if > 180
f_notice= 1.05 if notice_period_days â‰¤ 15
          1.02 if â‰¤ 30
          0.98 if â‰¤ 60
          0.92 if â‰¤ 90
          0.86 if > 90
f_otw   = 1.04 if open_to_work_flag else 0.99
f_follow= 0.97 + 0.06Â·interview_completion_rate        # offer_acceptance_rate == -1 ignored (neutral)

B = clip(f_resp Â· f_active Â· f_notice Â· f_otw Â· f_follow, 0.80, 1.12)
```

**Why this shape (the requested ordering guarantee):** a strong fit (`contentÂ·gates â‰ˆ 0.85`) that is fully unreachable scores `0.85Â·0.80 = 0.68`; a mediocre fit (`â‰ˆ0.42`) fully available scores `0.42Â·1.12 = 0.47`. So **available-strong > unreachable-strong > available-weak**, and an available stuffer (`B=1.12`) still cannot reach a real builder. `github_activity_score==-1` is **not** in `B` (half the pool has no GitHub). `saved_by_recruiters_30d` / `search_appearance_30d` are not used (redundant with response/activity and noisier).

---

## 5. Location handling

Primary effect is the additive `LOC` term (Â§EXACT WEIGHTS F, weight 0.04). Max swing â‰ˆ 4% â€” a tilt, not a gate, matching the JD's "Pune/Noida-preferred but flexibleâ€¦ Hyderabad/Mumbai/Delhi-NCR/Bangalore welcomeâ€¦ outside India case-by-case, no visa sponsorship." India-based candidates (75% of pool, large majority of golds) naturally dominate without any hard cut that could drop a genuine top-10 fit abroad. Relocation willingness substantially recovers the term for non-Tier-1-India and abroad candidates.

---

## 6. Honeypot detection

**ONLY the two rare/strong impossibility signals** (data-profile counts: `expert_zero_duration=21`, `duration_vs_dates_mismatch=33`, `careerspan_gt_yoe=4`). **Salary `min>max` (18,865, ~19%) and `skill_dur>career` (9,231, ~9%) are PERVASIVE SYNTHETIC NOISE and are never used** (they would gate ~25% of the pool including golds like `CAND_0033861`).

```
# Signal A â€” expert-zero-duration
A_count = #{skills with proficiency âˆˆ {expert, advanced} AND duration_months == 0}

# Signal B â€” duration-vs-dates impossibility (per role)
for role r: elapsed_r = months_between(start_date_r, end_date_r or ANCHOR)
            gap_r = duration_months_r - elapsed_r        # months claimed beyond what dates allow
B_hit = any role with gap_r â‰¥ 6                           # clean cliff: 100k have gap 0â€“1; 19 have gap â‰¥48; nothing between

HONEYPOT if  (A_count â‰¥ 3) OR (B_hit) OR (A_count â‰¥ 1 AND B_hit)
P_hp = 0.02 if HONEYPOT else 1.0
```

**Pipeline placement:** `P_hp` is the **last multiplier** (`final = available Â· P_hp`) AND any HONEYPOT candidate is **excluded from the top-100 emission set entirely**. This is a **hard gate, not a soft penalty** â€” Stage-3 disqualifies at >10% honeypot rate; we target 0%. The gap-`â‰¥6` threshold sits inside the empty `[2,47]` band, so it catches the ~19â€“33 impossible profiles with **zero false positives**. Offline validation requirement: the rule must flag on the order of ~80 (combined with indirect catches), **never thousands** â€” if it flags >300, tighten before submit.

---

## 7. Tie-breaking

`final` is a float; exact ties are rare but the validator requires unique ranks and non-increasing score. **Deterministic cascade:**

1. higher `final`
2. higher `PES` (most ground-truth-aligned: production-description evidence)
3. higher `B` (more reachable)
4. higher `EBF` (closer to 6â€“8y)
5. **`candidate_id` ascending** (final, validator-mandated fallback)

Emit `score = round(final, 6)`. For ranking we use the full-precision cascade; ranks 1..100 are assigned as a dense sequence after sorting, so **the displayed score column is guaranteed monotonically non-increasing** and any exactly-equal displayed scores are ordered by `candidate_id` ascending. Set `OMP/MKL/OPENBLAS_NUM_THREADS=1` and `PYTHONHASHSEED=0` for bit-identical reruns.

**Top-10 safety lock (protects the 0.50-weighted NDCG@10 and P@10):** only candidates with `PES â‰¥ 0.55`, `P_trap == 1.0`, `P_hp == 1.0`, and `B â‰¥ 0.92` are eligible for ranks 1â€“10. If fewer than 10 qualify, fill from the next-best by `final`.

---

## 8. Reasoning-string generation (local, fact-grounded, varied)

100% programmatic, no LLM. **Slot-filled multi-skeleton grammar** drawing only from fields actually present in the candidate record (no hallucination), with per-candidate variation seeded by `hash(candidate_id)`.

**Fact-bank assembled per candidate (all from parsed profile):**
- `yoe + current_title + current_company` â†’ e.g. *"7.0y Staff ML Engineer at Paytm"*
- strongest career-description evidence phrase, paraphrased from the actual highest-`role_evid` role â†’ *"shipped a hybrid BM25+dense retrieval system serving 50M+ queries/mo at Ola"*
- top 1â€“2 corroborated relevant skills with proficiency â†’ *"Vector Search (expert), Learning to Rank (expert)"*
- one JD-linkage â†’ *"matches the production retrieval + ranking-eval mandate"*
- **one honest concern when a real one exists**, pulled from penalty/availability features â†’ *"90-day notice above the sub-30 preference"*, *"206 days inactive, 7% recruiter response"*, *"London-based, no visa sponsorship"*, *"recsys-heavy, light on explicit eval framework"*, *"adjacent SWE title â€” production evidence is the deciding factor"*.

**Generation rules (pass all six Stage-4 checks):**
1. **Specific facts** â€” every clause inserts a literal field value.
2. **JD connection** â€” always name the specific matched requirement.
3. **Honest concerns** â€” the concern clause is emitted **iff** a driving concern feature is present.
4. **No hallucination** â€” post-generation assertion verifies every named skill/employer exists in the source JSON; absent slots are dropped, never invented.
5. **Variation** â€” ~12 sentence skeletons + synonym banks (`shipped/built/owned/drove`; `concern/caveat/watch-item`) chosen by `hash(candidate_id)`; clause order varied; facts differ per candidate. Post-gen de-dup: if Jaccard(two strings) > 0.8, re-roll the skeleton. Assert no two of the 100 are identical and none is empty.
6. **Rank consistency** â€” tone gated by rank band: **1â€“15** lead with strongest evidence + minor caveat; **16â€“50** "solid/partial fit with caveats"; **51â€“100** lead with the limiting factor ("adjacent/borderline, included as filler given <specific signal>"). The same features set both rank and tone, so consistency is automatic.

Example outputs (illustrative, varied):
> rank 2 â€” *"Senior ML Engineer at Ola, 7.8y; owned a BM25â†’hybrid dense semantic-search migration plus an Amazon learning-to-rank pipeline â€” squarely the production retrieval/ranking profile; responsive (0.66), 90-day notice the only soft caveat."*
> rank 78 â€” *"Applied ML Engineer at Dream11, 5.7y; real recsys experience but evidence skews to recommendation feeds over the search/IR core, no GitHub linked, 90-day notice â€” adjacent rather than ideal."*

---

## 9. Pipeline & engineering

### 9.1 Precompute (OFFLINE, unbounded time) â†’ `rank` (ONLINE, â‰¤5 min)

**Precompute step** (`precompute.py`, run once, may exceed 5 min):
- Parse `candidates.jsonl` â†’ compact per-candidate feature dict; precompile all regexes; run them over descriptions/summaries/skills; compute `PES, TFF, T_mult, SKC, EBF, LOC, BASE, B, P_trap, P_hp` raw inputs.
- Run bge-small-en-v1.5 over candidate texts â†’ `(100000Ã—384)` fp16 matrix; embed the frozen JD intent string â†’ `(384,)` query vector; compute & freeze the `EMB` calibration percentiles (0.20/0.55).
- Write artifacts: `features.parquet` (~30â€“60MB), `emb_matrix.fp16.npy` (~73MB), `jd_query.npy`, `calib.json`, `phrase_lists.json`, `company_lists.json`. All â‰ª 5GB.

**Rank step** (`rank.py --candidates ./candidates.jsonl --out ./submission.csv`, â‰¤5 min):
- Load artifacts (memmap the embedding matrix). One matmul for cosine â†’ `EMB`. Pure numpy arithmetic for the linear `content`, gates, multipliers â†’ `final` for all 100k.
- Apply honeypot exclusion + top-10 safety lock; sort by the Â§7 cascade; take top 100; generate reasoning; write CSV.
- **Budget:** matmul ~100ms; vectorized scoring over 100k ~seconds; reasoning for 100 ~ms. **Total < 60s, RAM < 1GB** (fp32 upcast of the matrix ~150MB) â€” comfortably inside 5 min / 16GB / CPU-only / no-network. *(Note: `rank.py` re-derives rule features from the jsonl itself so the ranking step is self-contained and reproducible in the Stage-3 sandbox; the parquet is an optional speed cache. Embeddings are the only true precompute dependency, which the spec explicitly permits.)*

### 9.2 Model/index choice
- bge-small-en-v1.5 (justified Â§B). **No FAISS/HNSW** â€” brute-force GEMV over 100kÃ—384 is faster, deterministic, and dependency-light for a single query.

### 9.3 Repo module breakdown
```
rank.py                  # single reproduce entrypoint (CLI as in spec Â§10.3)
precompute.py            # offline: embeddings + frozen calibration + optional feature cache
src/parse.py             # jsonl -> feature dicts, date math, ANCHOR
src/evidence.py          # PES + phrase lists + company classification
src/titles.py            # TFF / T_mult
src/skills.py            # SKC + expert-zero flag
src/behavioral.py        # B, EBF, LOC, BASE
src/traps.py             # P_trap rules
src/honeypots.py         # P_hp (two rare signals only)
src/embed.py             # bge load (local snapshot/ONNX), candidate+JD text builders, EMB calibration
src/score.py             # combination formula Â§EXACT WEIGHTS, top-10 lock, tie-break
src/reasoning.py         # slot-filled grammar, de-dup, no-hallucination assertion
artifacts/               # emb_matrix.fp16.npy, jd_query.npy, calib.json, *_lists.json, model snapshot
tests/                   # gold-set sanity, honeypot-count, reasoning de-dup/no-hallucination, monotonic-score
README.md, requirements.txt, submission_metadata.yaml
```
**Single reproduce command:** `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`
Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0` for bit-identical output.

**Pre-submit checks (run locally):** exactly 100 rows; ranks 1..100 each once; unique `candidate_id`s; all IDs exist in jsonl; score non-increasing; all 25 known golds land in top ~40; zero honeypots in top 100; no empty/duplicate reasoning; reasoning skills/employers all present in profiles; full run < 5 min on a 16GB CPU box.

---

## 10. Top residual risks & per-trap mitigations

**Trap-by-trap (how each is defeated):**

| Trap | Mechanism that defeats it |
|---|---|
| **Keyword stuffer** (non-tech title + â‰¤11 AI skills + buzzword summary) | `T_mult=0.06` gate (Â·0.39) + `P_trap=0.25` + `SKC` count-cap 6 & log-sublinear + `EMB` embeds descriptions (theirs are non-technical, low cosine). Cannot reach relevance. |
| **Self-learner SWE** (RAG side project, not professional) | `SUMMARY_RX_SIDEPROJECT` + ADJACENT title + no production `PES` â†’ `P_trap=0.45`; partial fit, demoted but eligible. |
| **Plain-language gem** (production recsys, no buzzword skills, generic title) | `PES` is description-driven and fires on broad phrase families even with â‰¤1 buzzword skill; ADJACENT+`PESâ‰¥0.4` gem path (`T_mult=0.62`); `EMB` corroborates semantically. Ranks high. |
| **Behavioral twins** | Equal content â†’ `B` decides; available/responsive/low-notice/local wins by a clear, explainable gap. |
| **Honeypot** | Two rare impossibility checks â†’ `P_hp=0.02` + excluded from top-100. The ~40 not caught directly also fail PES/title/skill checks and land far below rank 100. |
| **Consulting-only** | `P_trap=0.45` only if EVERY company is consulting; **waived** if any prior product-co role (JD carve-out). |
| **CV/speech/robotics w/o NLP/IR** | `P_trap=0.55` only when CV/speech is primary AND `PES<0.10`; never touches golds carrying CV skills as noise. |
| **Research-only** | `P_trap=0.50` when academic/research titles AND zero `PROD_RX` hits anywhere (JD hard disqualifier). |
| **Location (abroad/no-relocate)** | Soft additive `LOC` (max ~4% swing); never a gate â€” elite abroad fits remain in top-50. |

**Residual risks & mitigations:**
1. **Calibration drift** (weights hand-set from 25 golds). â†’ All weights are constants in one config block; pre-submit assert all 25 golds in top ~40 and no honeypot/NONTECH in top 100.
2. **False-positive trap penalty** burying a true tier-5 (gold whose summary says "side project", ex-product engineer now at TCS). â†’ Consulting requires *every* company consulting; self-learner requires ADJACENT title *and* explicit non-professional phrasing; apply only the single harshest penalty; floors â‰¥0.25 (never zero), so strong `PES` survives.
3. **Plain-gem under-ranking** (ADJACENT cap 0.62). â†’ gem path + `EMB` 0.10 rescue; verify a sample lands above rank 50.
4. **Honeypot off-by-one** in date math. â†’ `gap â‰¥ 6` sits inside the empty `[2,47]` band; use `ANCHOR` for current roles; log all flagged IDs for spot-check; assert count â‰ˆ 80 not thousands.
5. **Reasoning Stage-4 failure** (templated/hallucinated). â†’ multi-skeleton + synonym banks + Jaccard de-dup + post-gen field-existence assertion; concern clause only when a real concern exists.
6. **Determinism/repro in Docker.** â†’ thread-count=1, `PYTHONHASHSEED=0`, fp16 frozen matrix, score rounded to 6 dp, `candidate_id`-ascending final tie-break â†’ bit-identical CSV.
7. **Compute overrun.** â†’ embeddings fully precomputed; rank step is one matmul + vectorized arithmetic + 100 reasoning strings (< 60s); benchmark on 16GB CPU before each of the 3 allowed submissions.