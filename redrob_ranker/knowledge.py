"""
Domain knowledge for the Redrob "Senior AI Engineer — Founding Team" ranker.

Two kinds of fact live here, both auditable in one place:
  (1) entity lists derived from profiling the 100k pool (companies, cities, titles);
  (2) raw regex source strings for the phrase families the evidence/trap detectors use.

The scoring WEIGHTS live in scoring.py; this file is purely the vocabulary.
All matching is case-insensitive on lowercased text.

References: analysis/design_spec.md (§EXACT WEIGHTS), analysis/data_profile.md, analysis/deep_dive.md
"""
import datetime
import re

# Fixed, deterministic "now" for recency math (= max last_active_date in pool). No clock, no network.
RECENCY_ANCHOR = datetime.date(2026, 6, 1)

# ----------------------------------------------------------------------------
# Company classification (lowercased substring match). One canonical class each.
# ----------------------------------------------------------------------------
BIGTECH = ["amazon", "microsoft", "google", "meta", "facebook", "netflix", "uber",
           "linkedin", "salesforce", "apple", "adobe", "nvidia"]

PRODUCT_AI = ["paytm", "ola", "zomato", "flipkart", "razorpay", "swiggy", "cred",
              "freshworks", "meesho", "dream11", "nykaa", "inmobi", "zoho", "vedantu",
              "byju", "upgrad", "phonepe", "myntra", "sharechat", "unacademy", "groww",
              "zepto", "sarvam", "mad street den", "glance", "yellow.ai", "verloop",
              "haptik", "rephrase", "krutrim", "locobuzz", "saarthi", "aganitha",
              "observe.ai", "niramai", "wysa", "gan.ai", "uniphore", "niki.ai"]

CONSULTING = ["tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
              "capgemini", "hcl", "tech mahindra", "mphasis", "mindtree", "ltimindtree",
              "ibm", "deloitte", "dxc"]

FICTIONAL_FILLER = ["wayne enterprises", "wayne", "initech", "pied piper", "globex",
                    "acme corp", "acme", "dunder mifflin", "stark industries", "stark", "hooli"]


def company_class(company):
    """Return one of: 'bigtech', 'product_ai', 'consulting', 'fictional', 'unknown'."""
    c = (company or "").lower()
    if not c:
        return "unknown"
    for name in CONSULTING:          # consulting checked first (JD's explicit negative)
        if name in c:
            return "consulting"
    for name in BIGTECH:
        if name in c:
            return "bigtech"
    for name in PRODUCT_AI:
        if name in c:
            return "product_ai"
    for name in FICTIONAL_FILLER:
        if name in c:
            return "fictional"
    return "unknown"


COMPANY_MULT = {"bigtech": 1.05, "product_ai": 1.00, "fictional": 0.85,
                "unknown": 0.85, "consulting": 0.55}

# ----------------------------------------------------------------------------
# Locations (JD: Pune/Noida preferred; Hyd/Mumbai/Delhi-NCR/Bangalore welcome).
# ----------------------------------------------------------------------------
PREFERRED_CITIES = ["pune", "noida"]
WELCOME_CITIES = ["hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "bangalore",
                  "bengaluru", "ncr"]

# ----------------------------------------------------------------------------
# Title classes (precedence CORE > NONTECH > ADJACENT). Matched on lowercased title.
# ----------------------------------------------------------------------------
CORE_TITLE_RX = re.compile(
    r"\b(ai|ml|machine[- ]learning|nlp|applied)\b[\w &/.-]*\bengineer\b"          # ai/ml/nlp/applied engineer
    r"|\b(ai|ml|machine[- ]learning|nlp|applied|data|research)\b[\w &/.-]*\bscientist\b"  # data/applied/research scientist
    r"|\b(search|ranking|recommendation|relevance|recsys|discovery|personalization)\b[\w &/.-]*\bengineer\b"
    r"|\bai specialist\b|\bai research\b|\bresearch engineer\b", re.I)
ADJACENT_TITLE_RX = re.compile(
    r"(software|backend|cloud|devops|frontend|full ?stack|data|analytics|mobile|platform) engineer"
    r"|developer|sde|swe", re.I)
NONTECH_TITLE_RX = re.compile(
    r"business analyst|hr manager|human resource|(marketing|operations|project) manager"
    r"|accountant|sales|content writer|graphic designer|(civil|mechanical) engineer"
    r"|customer support|qa engineer|tester|recruiter", re.I)
JUNIOR_RX = re.compile(r"\bjunior\b|\bjr\.?\b|\bintern\b|\btrainee\b|\bassociate\b", re.I)


def title_class(title):
    """CORE > NONTECH > ADJACENT > unknown."""
    t = (title or "")
    if CORE_TITLE_RX.search(t):
        return "core"
    if NONTECH_TITLE_RX.search(t):
        return "nontech"
    if ADJACENT_TITLE_RX.search(t):
        return "adjacent"
    return "unknown"


# ----------------------------------------------------------------------------
# Production-evidence phrase families (matched on career DESCRIPTIONS only).
# ----------------------------------------------------------------------------
RETRIEVAL_RX = re.compile(
    r"semantic search|hybrid retrieval|dense (and|\+|/) ?sparse|sparse (and|\+|/) ?dense|\bbm25\b"
    r"|dense vector|vector recall|vector search|embedding-based (search|retrieval)|\bfaiss\b|\bhnsw\b"
    r"|nearest[- ]neighbor|\bbge\b|sentence[- ]transformer|index refresh|embedding drift"
    r"|dense retrieval|retrieval[- ]quality", re.I)
RANKING_RX = re.compile(
    r"learning[- ]to[- ]rank|\bltr\b|ranking (layer|pipeline|model|system|infrastructure)"
    r"|re[- ]rank|\bxgboost\b|\blightgbm\b|relevance (labeling|labelling|tuning|scoring)"
    r"|candidate sourcing|candidate generation|two[- ]tower", re.I)
RECSYS_RX = re.compile(
    r"recommendation system|recommender|collaborative filtering|matrix factorization"
    r"|discovery feed|personalization|personalisation|content[- ]based ranking|item[- ]item", re.I)
RAGEVAL_RX = re.compile(
    r"\brag\b|retrieval[- ]augmented|\bndcg\b|\bmrr\b|\bmap@|a/?b test|offline[- ]to[- ]online"
    r"|eval(uation)? (framework|harness|infrastructure|pipeline)|offline (experiment|benchmark)", re.I)
PROD_RX = re.compile(
    r"\bshipped\b|\bdeployed\b|in production|serving \d|\bqps\b|p95|latency"
    r"|50m\+|10m\+|35m\+|30m\+|million|live a/?b|rolled out|end[- ]to[- ]end|at scale", re.I)

EVIDENCE_FAMILIES = [("retrieval", RETRIEVAL_RX, 0.45), ("ranking", RANKING_RX, 0.40),
                     ("recsys", RECSYS_RX, 0.35), ("rageval", RAGEVAL_RX, 0.30),
                     ("prod", PROD_RX, 0.20)]
# "IR/eval families" used for the breadth bonus (retrieval, ranking, rageval).
IREVAL_FAMILY_KEYS = {"retrieval", "ranking", "rageval"}

# ----------------------------------------------------------------------------
# Skill-corroboration: relevant-skill families (matched on skill NAME, lowercased).
# ----------------------------------------------------------------------------
RELEVANT_SKILL_RX = re.compile(
    r"pinecone|weaviate|qdrant|milvus|pgvector|faiss|elasticsearch|opensearch|vector search"
    r"|vector db|vector database|embedding|sentence[- ]transformer|semantic search|\bbm25\b"
    r"|information retrieval|learning to rank|recommendation|recommender|\brag\b|\bndcg\b|\bmrr\b"
    r"|hugging face|transformers|fine-tun|\blora\b|\bqlora\b|\bpeft\b|\bllm", re.I)

# ----------------------------------------------------------------------------
# Trap framing regexes (matched on SUMMARY / combined narrative).
# ----------------------------------------------------------------------------
SIDEPROJECT_RX = re.compile(
    r"side[- ]project|personal project|hobby project|online course|tutorial|experimenting with"
    r"|experimented with|haven'?t done .{0,30}professional|not .{0,20}professional"
    r"|built a small (rag|llm)|self-?learner|self-?directed|in my spare time|kaggle"
    r"|curious about how ai|exploring how llms|augment my work|emerging ai|ai enthusiast"
    r"|generative ai explorer|played with the openai|i think the space is exciting", re.I)
LANGCHAIN_RECENT_RX = re.compile(r"langchain|openai api|anthropic api|chatgpt", re.I)
RESEARCH_ONLY_RX = re.compile(
    r"\bphd\b|research scientist|research fellow|postdoc|academic|publication|paper|thesis"
    r"|research lab|university research", re.I)
CV_SPEECH_RX = re.compile(
    r"opencv|\byolo\b|object detection|image classification|\bcnn\b|diffusion|\bgan[s]?\b"
    r"|\basr\b|\btts\b|speech recognition|robotics|segmentation", re.I)
NLP_IR_PRESENT_RX = re.compile(
    r"\bnlp\b|natural language|retrieval|ranking|semantic|embedding|recommendation|search"
    r"|information retrieval|\bllm\b", re.I)


# ----------------------------------------------------------------------------
# Small text helpers
# ----------------------------------------------------------------------------
def career_descriptions_text(candidate):
    return " ".join((r.get("description") or "") for r in candidate.get("career_history", []))


def narrative_text(candidate):
    p = candidate.get("profile", {}) or {}
    parts = [p.get("headline", ""), p.get("summary", "")]
    parts += [(r.get("title") or "") + " " + (r.get("description") or "")
              for r in candidate.get("career_history", [])]
    return " ".join(parts)


def skills_text(candidate):
    return " ".join((s.get("name") or "") for s in candidate.get("skills", []))
