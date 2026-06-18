#!/usr/bin/env python3
"""
Render the 10-slide Idea Submission deck to PDF (+ per-slide PNGs for QA) with matplotlib.
No LibreOffice/PowerPoint needed. Content mirrors deck/deck_content.md.

    python deck/build_deck.py
"""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

# ---- fill these from final run / packaging ----
TEAM_NAME = "TODO — your team name"
LEADER = "TODO — team leader"
N_HP = 82          # honeypots flagged pool-wide
REPO_URL = "github.com/<you>/redrob-ranker"
SANDBOX_URL = "huggingface.co/spaces/<you>/redrob-ranker"
VIDEO_URL = "<walkthrough video link>"

OUT = os.path.dirname(os.path.abspath(__file__))
SLIDES = os.path.join(OUT, "slides")
os.makedirs(SLIDES, exist_ok=True)

# Palette — "deep navy + ice + mint accent" (talent-intelligence / AI product)
INK = "#0E1B33"
NAVY = "#1E2761"
ICE = "#CADCFC"
MINT = "#00C39A"
TEAL = "#1C7293"
WHITE = "#FFFFFF"
TEXT = "#15203B"
MUTED = "#5C6788"
CARD = "#F3F6FC"

W, H = 13.333, 7.5  # 16:9


def slide(dark=False):
    fig = plt.figure(figsize=(W, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, color=(INK if dark else WHITE), zorder=-10))
    return fig, ax


def num_badge(ax, n, x=0.85, y=6.75):
    ax.add_patch(Circle((x, y), 0.28, color=MINT, zorder=5))
    ax.text(x, y, str(n), ha="center", va="center", color=INK, fontsize=16, fontweight="bold", zorder=6)


def title(ax, t, x=1.35, y=6.72, color=NAVY, size=30):
    ax.text(x, y, t, ha="left", va="center", color=color, fontsize=size, fontweight="bold")


def card(ax, x, y, w, h, color=CARD, ec="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=color, ec=ec, lw=1.2, zorder=1))


def wraptext(s, usable_in, size):
    """Wrap to the box width (matplotlib's wrap=True uses axes width, not the card)."""
    cpl = max(10, int(usable_in * 118.0 / size))
    return textwrap.fill(s, cpl)


def cardtext(ax, s, x, y, usable_in, size, color=TEXT, va="top"):
    ax.text(x, y, wraptext(s, usable_in, size), ha="left", va=va, color=color,
            fontsize=size, linespacing=1.3, zorder=3)


def bullets(ax, items, x, y, w=11.3, gap=0.18, size=14.5, color=TEXT, head=MINT):
    """w = usable text width in inches. Returns the y just below the last bullet."""
    cy = y
    line_h = size / 72.0 * 1.34
    for it in items:
        wrapped = wraptext(it, w - 0.35, size)
        n = wrapped.count("\n") + 1
        ax.add_patch(Circle((x + 0.07, cy - 0.11), 0.05, color=head, zorder=3))
        ax.text(x + 0.30, cy, wrapped, ha="left", va="top", color=color, fontsize=size,
                linespacing=1.34, zorder=3)
        cy -= n * line_h + gap
    return cy


def save(fig, n):
    fig.savefig(os.path.join(SLIDES, f"slide-{n:02d}.png"))
    return fig


figs = []

# ---- Slide 1 — Title (dark) ----
fig, ax = slide(dark=True)
ax.add_patch(Circle((11.7, 6.2), 1.7, color=NAVY, zorder=0))
ax.add_patch(Circle((12.6, 1.1), 1.1, color=TEAL, alpha=0.5, zorder=0))
ax.text(1.0, 5.2, "Intelligent Candidate", color=WHITE, fontsize=46, fontweight="bold", va="center")
ax.text(1.0, 4.35, "Discovery & Ranking", color=MINT, fontsize=46, fontweight="bold", va="center")
ax.text(1.0, 3.45, "Ranking 100,000 candidates for a Senior AI Engineer role —",
        color=ICE, fontsize=17, va="center")
ax.text(1.0, 3.05, "by understanding fit, not matching keywords.", color=ICE, fontsize=17, va="center")
ax.text(1.0, 1.5, f"Team   {TEAM_NAME}", color=WHITE, fontsize=15, va="center", fontweight="bold")
ax.text(1.0, 1.05, f"Lead   {LEADER}", color=MUTED, fontsize=13, va="center")
ax.text(1.0, 0.6, "Redrob Data & AI Challenge", color=MUTED, fontsize=12, va="center")
figs.append(save(fig, 1))

# ---- Slide 2 — Solution Overview ----
fig, ax = slide()
num_badge(ax, 1); title(ax, "Solution Overview")
ax.text(1.35, 6.05, "A hybrid ranker — transparent rule layer + local semantic embeddings — scoring all",
        fontsize=14.5, color=MUTED, va="center")
ax.text(1.35, 5.68, "100K candidates in ~1.6s on CPU, offline.", fontsize=14.5, color=MUTED, va="center")
cards = [
    ("Evidence over keywords", "Backbone signal reads career-history descriptions (what they built) — 'vector DB' is in 12,866 skill lists but only 108 descriptions."),
    ("Role-class gate", "A 'Marketing Manager with 11 AI skills' keeps only ~39% of its score — structurally can't reach the top."),
    ("Availability matters", "206 days idle + 7% response = not hireable. Behavioral signals are a first-class multiplier."),
    ("Impossible profiles gated", "Five consistency checks exclude the ~80 honeypots before they reach the shortlist."),
]
xs = [0.85, 6.95]; ys = [3.0, 3.0, 0.55, 0.55]
for i, (h, b) in enumerate(cards):
    x = xs[i % 2]; y = ys[i]
    card(ax, x, y, 5.55, 2.1)
    ax.text(x + 0.3, y + 1.75, h, fontsize=16, fontweight="bold", color=NAVY, va="top")
    cardtext(ax, b, x + 0.3, y + 1.25, usable_in=4.95, size=12.3)
figs.append(save(fig, 2))

# ---- Slide 3 — JD Understanding ----
fig, ax = slide()
num_badge(ax, 2); title(ax, "JD Understanding & Candidate Evaluation")
ax.text(0.95, 6.0, "What the role actually needs", fontsize=17, fontweight="bold", color=NAVY, va="center")
cy = bullets(ax, [
    "Production embeddings retrieval + vector DB / hybrid search + learning-to-rank / recsys, at product companies.",
    "5–9 yrs (ideal 6–8); has shipped a search/ranking/recsys system at scale; NDCG/MRR/A-B eval rigor.",
    "Pune/Noida or willing to relocate; sub-30-day notice preferred; no visa sponsorship outside India.",
], 0.95, 5.5, w=11.4, size=13.5)
cy -= 0.25
ax.text(0.95, cy, "Traps it deliberately rejects", fontsize=17, fontweight="bold", color="#B23A48", va="center")
bullets(ax, [
    "Keyword stuffers — non-engineers with AI skills listed; self-learner 'side-project' RAG; recent-only LangChain.",
    "Pure-consulting careers (TCS/Infosys/Wipro…); CV/speech/robotics without NLP/IR; research-only without production.",
    "Our answer: score the gap between what the JD says and means — production evidence ≫ title ≫ skills ≫ semantic fit.",
], 0.95, cy - 0.55, w=11.4, size=13.5, head="#B23A48")
figs.append(save(fig, 3))

# ---- Slide 4 — Ranking Methodology ----
fig, ax = slide()
num_badge(ax, 3); title(ax, "Ranking Methodology")
card(ax, 0.85, 5.0, 11.6, 1.35, color=INK)
ax.text(1.15, 5.95, "content = 0.46·PES + 0.10·EMB + 0.14·Title + 0.10·Skills + 0.08·Exp + 0.04·Loc + 0.08·Base",
        fontsize=12.5, color=ICE, family="monospace", va="center")
ax.text(1.15, 5.5, "final   = (content + synergy) · RoleClassGate · TrapPenalty · Availability · HoneypotGate",
        fontsize=12.5, color=MINT, family="monospace", va="center")
rows = [
    ("PES — Production-Evidence", "0.46", "retrieval/ranking/recsys/RAG/eval in career descriptions × company × duration × recency"),
    ("EMB — semantic", "0.10", "bge-small cosine vs JD-intent; calibrated so genuine fits sit above pool p99"),
    ("Title + role-class gate", "0.14", "core AI title ×1.0; non-tech ×0.06 — the anti-keyword-stuffer lever"),
    ("Skills (corroborated)", "0.10", "endorsements+duration+proficiency; expert-at-0-months = red flag"),
    ("Behavioral availability", "×", "response, recency, notice, open-to-work → multiplier [0.80, 1.12]"),
    ("Trap / honeypot gates", "×", "demote stuffers/self-learner/consulting/CV-only; exclude impossible profiles"),
]
y = 4.35
for name, w, desc in rows:
    ax.text(0.95, y, name, fontsize=13.2, fontweight="bold", color=NAVY, va="center")
    ax.text(4.7, y, w, fontsize=13.2, color=MINT, va="center", fontweight="bold", ha="center")
    ax.text(5.25, y, desc, fontsize=11.8, color=TEXT, va="center")
    y -= 0.62
ax.text(0.95, 0.45, "Additive for graded factors; multiplicative for gates — one disqualifier can't be out-voted by keyword volume.",
        fontsize=11.5, color=MUTED, style="italic", va="center")
figs.append(save(fig, 4))

# ---- Slide 5 — Explainability & Data Validation ----
fig, ax = slide()
num_badge(ax, 4); title(ax, "Explainability & Data Validation")
card(ax, 0.85, 3.6, 5.6, 2.85)
ax.text(1.15, 6.15, "Explainable & grounded", fontsize=16, fontweight="bold", color=NAVY, va="top")
bullets(ax, [
    "Fact-grounded reason per rank, no LLM: years, title, company, evidence, skills.",
    "Up to two honest concerns disclosed.",
    "Assertion: every named skill/employer exists — 0 hallucinations.",
], 1.15, 5.6, w=4.55, gap=0.16, size=11.5)
card(ax, 6.75, 3.6, 5.7, 2.85, color="#FBEEEE")
ax.text(7.05, 6.15, "5 impossibility checks", fontsize=16, fontweight="bold", color="#B23A48", va="top")
bullets(ax, [
    "≥3 expert skills with 0 months used",
    "role tenure > what its dates allow",
    "career sum > stated experience +3y",
    "summary tenure vs field off by >4y",
    "≥4 skills used longer than the career",
], 7.05, 5.6, w=4.65, gap=0.14, size=11.5, head="#B23A48")
card(ax, 0.85, 0.5, 11.6, 2.9, color=CARD)
ax.text(1.15, 3.0, "Deliberately ignored as synthetic noise", fontsize=15, fontweight="bold", color=NAVY, va="top")
ax.text(1.15, 2.45, "Salary min>max (~19% of pool) and any single skill-duration overrun (~9%) are pervasive in the data —",
        fontsize=13, color=TEXT, va="top")
ax.text(1.15, 2.0, "using them as honeypot signals would bury thousands of genuine candidates. We use only rare, false-positive-free checks.",
        fontsize=13, color=TEXT, va="top")
ax.text(1.15, 1.15, "Result: the ~80 honeypots are flagged and excluded; 0 reach the top-100.", fontsize=13.5,
        fontweight="bold", color=MINT, va="top")
figs.append(save(fig, 5))

# ---- Slide 6 — End-to-End Workflow ----
fig, ax = slide()
num_badge(ax, 5); title(ax, "End-to-End Workflow")
steps = [
    ("Job Description", "reduced to an\nintent vector (once)"),
    ("PRECOMPUTE\n(offline)", "bge-small embeddings\n+ rule-feature cache"),
    ("RANK\n(≤5 min, CPU, offline)", "load cached artifacts →\ncombine score"),
    ("Gate & order", "honeypot exclude →\ntop-10 lock → tie-break"),
    ("Output", "fact-grounded reasoning\n→ submission.csv"),
]
n = len(steps); bw = 2.15; gap = (W - 1.7 - n * bw) / (n - 1); x = 0.85; y = 3.4
for i, (h, b) in enumerate(steps):
    dark = i in (1, 2)
    card(ax, x, y, bw, 1.9, color=(NAVY if dark else CARD))
    ax.text(x + bw / 2, y + 1.5, h, ha="center", va="top", fontsize=12.5, fontweight="bold",
            color=(WHITE if dark else NAVY), linespacing=1.1)
    ax.text(x + bw / 2, y + 0.75, b, ha="center", va="top", fontsize=10.8,
            color=(ICE if dark else TEXT), linespacing=1.2)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x + bw + 0.06, y + 0.95), (x + bw + gap - 0.06, y + 0.95),
                     arrowstyle="-|>", mutation_scale=18, color=MINT, lw=2))
    x += bw + gap
ax.text(0.95, 1.7, "The expensive embedding work is one-time and offline; the judged ranking step only reads cached artifacts —",
        fontsize=12.5, color=MUTED, va="center")
ax.text(0.95, 1.3, "so it is CPU-only, network-free, and finishes in ~1.6 seconds.", fontsize=12.5, color=MUTED, va="center")
figs.append(save(fig, 6))

# ---- Slide 7 — System Architecture ----
fig, ax = slide()
num_badge(ax, 6); title(ax, "System Architecture")
# offline box
card(ax, 0.85, 3.7, 11.6, 2.55, color=CARD)
ax.text(1.15, 6.05, "OFFLINE  ·  precompute.py   (network + time allowed)", fontsize=14, fontweight="bold", color=TEAL, va="top")
for i, (t, s) in enumerate([("bge-small-en-v1.5", "JD + candidate → cand_embeddings.npy"),
                            ("rule features", "PES · title · skills · behavioral · traps → features.npz")]):
    cx = 1.3 + i * 5.7
    card(ax, cx, 3.95, 5.2, 1.5, color=WHITE, ec=ICE)
    ax.text(cx + 0.25, 5.25, t, fontsize=13, fontweight="bold", color=NAVY, va="top")
    ax.text(cx + 0.25, 4.75, s, fontsize=11.2, color=TEXT, va="top")
# online box
card(ax, 0.85, 0.5, 11.6, 2.75, color=INK)
ax.text(1.15, 3.0, "ONLINE  ·  rank.py   (CPU-only · no network · ~1.6s)", fontsize=14, fontweight="bold", color=MINT, va="top")
for i, (t, s) in enumerate([("features", "signals + honeypot/trap"),
                            ("scoring", "weights · top-10 lock · valid emit"),
                            ("reasoning", "grounded NL, no LLM")]):
    cx = 1.3 + i * 3.75
    card(ax, cx, 0.75, 3.4, 1.5, color=NAVY)
    ax.text(cx + 0.22, 2.05, t, fontsize=12.5, fontweight="bold", color=ICE, va="top")
    ax.text(cx + 0.22, 1.55, s, fontsize=10.6, color=WHITE, va="top")
figs.append(save(fig, 7))

# ---- Slide 8 — Results & Performance ----
fig, ax = slide()
num_badge(ax, 7); title(ax, "Results & Performance")
stats = [("100/100", "top-100 are genuine\nAI/ML/NLP/DS roles"), ("0", "honeypots & traps\nin the top-100"),
         ("~1.6 s", "ranking step\n(limit: 5 min)"), ("90/100", "India-based;\nall product/AI/big-tech")]
x = 0.85
for big, lab in stats:
    card(ax, x, 4.35, 2.78, 2.0, color=CARD)
    ax.text(x + 1.39, 5.75, big, ha="center", va="center", fontsize=34, fontweight="bold", color=MINT)
    ax.text(x + 1.39, 4.85, lab, ha="center", va="center", fontsize=11.5, color=TEXT, linespacing=1.2)
    x += 2.93
ax.text(0.95, 3.85, "Ranking quality", fontsize=15, fontweight="bold", color=NAVY, va="top")
bullets(ax, [
    f"{N_HP} honeypots flagged pool-wide (of ~80 seeded); 0 in the top-100 — far under the 10% disqualification line.",
    "Behavioral down-weighting verified: a PES-0.98 but 206-day-idle / 7%-response candidate correctly sinks to the tail.",
    "Reasoning: 0 hallucinations, all distinct, tone matches rank. Output passes the official format validator.",
], 0.95, 3.45, w=11.4, gap=0.16, size=13.5)
card(ax, 0.85, 0.45, 11.6, 1.05, color=INK)
ax.text(1.15, 0.97, "Constraints met:  ≤5 min → ~1.6s   ·   ≤16 GB → <1 GB   ·   CPU-only ✓   ·   no network during ranking ✓",
        fontsize=12.5, color=ICE, va="center", family="monospace")
figs.append(save(fig, 8))

# ---- Slide 9 — Technologies ----
fig, ax = slide()
num_badge(ax, 8); title(ax, "Technologies Used")
techs = [
    ("Python · numpy", "Vectorized scoring + a single matmul for cosine over 100K vectors."),
    ("sentence-transformers · PyTorch (CPU)", "BAAI/bge-small-en-v1.5 — small, fast, strong retrieval embeddings; used offline only."),
    ("No vector DB / FAISS at rank time", "One query over 100K vectors is one matmul; an ANN index adds dependency + non-determinism for no gain."),
    ("Deterministic, explainable heuristics", "No black-box LTR training (no labels); every weight is auditable for the code-reproduction & interview."),
]
y = 5.7
for h, b in techs:
    ax.add_patch(Circle((1.15, y - 0.02), 0.16, color=MINT))
    ax.text(1.6, y, h, fontsize=15.5, fontweight="bold", color=NAVY, va="center")
    ax.text(1.6, y - 0.5, b, fontsize=12.5, color=TEXT, va="center")
    y -= 1.35
ax.text(0.95, 0.5, "Chosen to maximize quality within a hard CPU / offline / 5-minute budget — and to stay defensible.",
        fontsize=12.5, color=MUTED, style="italic", va="center")
figs.append(save(fig, 9))

# ---- Slide 10 — Submission Assets (dark) ----
fig, ax = slide(dark=True)
ax.add_patch(Circle((12.0, 1.0), 1.5, color=NAVY, zorder=0))
title(ax, "Submission Assets", color=WHITE)
ax.add_patch(Circle((0.85, 6.72), 0.28, color=MINT, zorder=5))
ax.text(0.85, 6.72, "9", ha="center", va="center", color=INK, fontsize=16, fontweight="bold", zorder=6)
items = [
    ("GitHub repo", REPO_URL),
    ("Reproduce", "python rank.py --candidates ./candidates.jsonl --out ./submission.csv"),
    ("Sandbox / demo", SANDBOX_URL),
    ("Ranked output", "submission.csv  (top-100)"),
    ("Walkthrough video", VIDEO_URL),
]
y = 5.5
for h, b in items:
    ax.text(1.0, y, h, fontsize=16, fontweight="bold", color=MINT, va="center")
    ax.text(4.4, y, b, fontsize=13.5, color=ICE, va="center", family=("monospace" if "python" in b or "submission" in b else "sans-serif"))
    y -= 0.92
ax.text(1.0, 0.55, "Build something real. Make hiring smarter.", fontsize=14, color=MUTED, style="italic", va="center")
figs.append(save(fig, 10))

# ---- combine into PDF ----
pdf_path = os.path.join(OUT, "Redrob_Idea_Submission.pdf")
with PdfPages(pdf_path) as pdf:
    for f in figs:
        pdf.savefig(f)
        plt.close(f)
print("wrote", pdf_path)
print("wrote", len(figs), "slide PNGs to", SLIDES)
