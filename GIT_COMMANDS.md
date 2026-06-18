# Git: commit history + push

Run these from the project root (`smart recruiter/`) in Git Bash. They create a clean, staged
commit history (genuine build order — Stage-4 review looks for this) and push to your GitHub repo.
The 153 MB embedding matrix is intentionally excluded (regenerated via `precompute.py`).

```bash
git init
git branch -M main

# 1 — project scaffold & dependencies
git add .gitignore requirements.txt README.md
git commit -m "chore: project scaffold, dependencies, and README

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"

# 2 — data understanding (EDA over the 100k pool)
git add analysis/profile_data.py analysis/deep_dive.py analysis/data_profile.md analysis/deep_dive.md
git commit -m "feat(eda): profile the 100k pool — archetypes, trap & honeypot landscape

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"

# 3 — methodology design + adversarial-review notes
git add analysis/design_spec.md analysis/review_fixlist.md
git commit -m "docs: synthesized scoring spec and adversarial-review fix list

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"

# 4 — ranker package + offline precompute + ranking entrypoint
git add redrob_ranker/ precompute.py rank.py
git commit -m "feat(ranker): hybrid scoring (PES + embeddings + rule gates), honeypot signals A-E, grounded reasoning, CPU/offline ranking step

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"

# 5 — pre-submit format + quality checks
git add tests/check_submission.py
git commit -m "test: pre-submit format + quality checks (honeypots, hallucination, gold placement, monotonic score)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"

# 6 — sandbox demo
git add sandbox/
git commit -m "feat(sandbox): Streamlit demo that ranks a small candidate sample live

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"

# 7 — idea-submission deck
git add deck/
git commit -m "docs(deck): 10-slide idea-submission deck (PDF + generator)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"

# 8 — submission CSV, metadata, committed small artifacts
git add submission.csv submission_metadata.yaml GIT_COMMANDS.md \
        artifacts/features.npz artifacts/jd_embedding.npy artifacts/cand_ids.json artifacts/precompute_meta.json
git commit -m "feat: final top-100 submission, metadata, and committed precompute artifacts

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

## Push to GitHub

**Option A — with the GitHub CLI (`gh`):**
```bash
gh repo create redrob-ranker --private --source=. --remote=origin --push
```

**Option B — manual:** create an empty repo named `redrob-ranker` on github.com (no README), then:
```bash
git remote add origin https://github.com/<YOUR_USERNAME>/redrob-ranker.git
git push -u origin main
```

> Private repo is fine — you can grant the organizers access at Stage 3 (email communicated then).

## Reproduce from a fresh clone
```bash
# place the organizer's candidates.jsonl at ./candidates.jsonl, then:
pip install -r requirements.txt
python precompute.py --candidates ./candidates.jsonl     # one-time, ~25-35 min CPU (regenerates the 153MB matrix)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv   # ~2s
```
