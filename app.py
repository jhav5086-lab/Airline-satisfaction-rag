"""
Airline Satisfaction RAG — app.py (v2.3 final)

Bug fixes vs v2.2 (from "NPS / insurance / which domain" transcript):
  #D  Applicability questions ("can this apply to X?") refusing coldly   → new prompt rule + intent hint
  #E  Fragmented follow-ups ("which domain?") refused by gate           → history-aware retrieval
  #F  Gate too strict when history contains a substantive question       → bypass gate if history is relevant
"""
import re
import json
import difflib
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Airline Satisfaction RAG", page_icon="🛫", layout="wide")

GROQ_API_KEY  = st.secrets["GROQ_API_KEY"]
MODEL         = "openai/gpt-oss-20b"
TOP_K         = 6
MAX_TOKENS    = 1500
MIN_RAW_SIM   = 0.20
NUMBER_TOL    = 0.005
HISTORY_TURNS = 2
AMBIGUOUS_MAX_TOKENS = 6   # queries shorter than this use history-expanded retrieval

CHART_KEYWORDS = (
    "chart", "plot", "graph", "visuali", "draw ", "render image",
    "bar plot", "line plot", "pie chart", "histogram", "scatter",
)
GREETING_TOKENS = {"hi", "hello", "hey", "greetings", "yo", "howdy"}

# Applicability / transferability intent — the model should answer these with
# a "not in artifacts, but approach transfers" pattern instead of raw refusal.
APPLICABILITY_KEYWORDS = [
    "apply to", "applied to", "reuse", "reusable", "transfer",
    "other domain", "other industr", "different domain", "different industr",
    "insurance", "healthcare", "banking", "retail", "telecom", "e-commerce",
    "can it work for", "could it work for", "generalize", "generaliz",
    "use case", "use-case", "not airline", "non-airline",
]

STOPWORDS = set("""
a an the of in on at to for with by from as and or but if then else so not no yes
all any some more most very just only too also than about into over under
what when where which who why how this that these those there here
is are was were be been being am
have has had having do does did doing done
will would can could should may might must shall
i you he she it we they me him her us them my your his its our their
okay ok sure fine please thanks thank
hell heck damn
""".split())

# ─────────────────────────────────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_backend():
    embedder    = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings  = np.load("./embeddings/embeddings.npy")
    with open("./embeddings/chunks.json") as f:
        chunks = json.load(f)
    groq_client = Groq(api_key=GROQ_API_KEY)
    return embedder, embeddings, chunks, groq_client

embedder, EMBEDDINGS, CHUNKS, groq_client = load_backend()
EMBEDDINGS_NORM = EMBEDDINGS / (np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True) + 1e-9)

# ─────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior data science consultant for an airline passenger satisfaction model.

## STYLE
- DEFAULT: 2–4 sentences. Direct answer first. No preamble, no greeting.
- Bullet list ONLY when user asks for "list", "prioritize", "all", "compare".
- Markdown table ONLY when user asks for "table" or "compare".
- NEVER exceed 120 words unless the user explicitly asks for "full summary" or "everything".
- NEVER repeat the question.

## GROUNDING
- Use ONLY facts from CONTEXT for claims about THIS model.
- If a factual question about this model is not in CONTEXT: reply exactly "The provided model artifacts do not contain this information."
- Off-topic (math, general knowledge, people, code, news, chit-chat, frustration): ONE sentence redirecting to model topics. Nothing else.

## APPLICABILITY QUESTIONS
- If asked whether this approach can be applied to another domain (insurance, banking, retail, etc.):
  1. State clearly: the model was trained on airline satisfaction data; the artifacts do not cover other industries.
  2. Note which ASPECTS of the approach are domain-agnostic: logistic regression with L2, two-model design (predictive + interpretive), SHAP for feature attribution, segment-level analysis, what-if lever prioritization, and number verification.
  3. Do NOT claim results for other domains. Do NOT invent metrics.
- Keep to 3–5 sentences. Cite [Model Overview] and [Methodology] where relevant.

## NUMBERS
- Quote exact values. Never round, never estimate, never substitute.
- If CONTEXT shows "+5.02", write "+5.02" — not "+5.0", not "about 5".

## CITATIONS — STRICT FORMAT
- After each factual sentence, add ONE citation: [Chunk Title]
- CORRECT: [Model Overview]   [Top 10 SHAP Drivers]   [What-If Business Lever Analysis]
- FORBIDDEN: 【...】 brackets, 【1†L1-L4】, "(Source: X)" — use ONLY [Title]

## CONVERSATION CONTEXT
- The user's previous turn is included above.
- Resolve pronouns and fragments ("which domain?", "how reliable is that?") using prior turns.
- If the current question is a fragment, answer the resolved question, not the literal fragment.

## CAPABILITY BOUNDARIES
- You CANNOT draw, render, or produce charts, plots, images, files.
- If asked for a chart: ONE sentence saying so, then optionally offer the data as a table.
- You CANNOT answer general knowledge, math, code, news.
"""

USER_TEMPLATE = """CONTEXT CHUNKS:
{context}

USER QUESTION: {question}

Grounded answer with [Title] citations:"""

# ─────────────────────────────────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────────────────────────────────
INTENT_KEYWORDS = {
    "caveats":            ["limitation", "caveat", "constraint", "risk", "weakness",
                           "problem", "shortcoming", "drawback", "flaw", "warning",
                           "not causal", "cannot", "can't"],
    "business_levers":    ["lever", "priority", "prioritize", "impact", "what-if",
                           "improve", "action", "investment"],
    "segments":           ["segment", "cohort", "business traveler",
                           "personal", "loyal", "demographic", "cabin"],
    "model_summary":      ["auc", "accuracy", "performance", "score", "brier",
                           "reliable", "test set", "train set", "dataset", "rows",
                           "records", "sample", "training"],
    "methodology":        ["method", "trained", "built", "approach", "feature",
                           "halo", "collinear", "imput", "bootstrap",
                           "cross-validation", "two models"],
    "feature_importance": ["shap", "importance", "driver"],
    "coefficients":       ["coefficient", "odds ratio", "odds"],
    "recommendations":    ["recommend", "suggest", "should we", "what to do"],
}

def _fuzzy_match(token: str, keyword: str) -> bool:
    if token in STOPWORDS or len(token) < 4:
        return False
    if abs(len(token) - len(keyword)) > 3:
        return False
    if token == keyword:
        return True
    if keyword.startswith(token) and len(token) >= len(keyword) - 2:
        return True
    if token.startswith(keyword) and len(keyword) >= len(token) - 2:
        return True
    return difflib.SequenceMatcher(None, token, keyword).ratio() >= 0.72

def detect_intents(query: str) -> set:
    q = query.lower()
    tokens = re.findall(r"[a-z]+", q)
    hints = set()
    for ctype, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if " " in kw or "-" in kw:
                if kw in q:
                    hints.add(ctype)
                    break
                continue
            if kw in q or any(_fuzzy_match(t, kw) for t in tokens):
                hints.add(ctype)
                break
    return hints

def is_applicability_question(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in APPLICABILITY_KEYWORDS)

def is_chart_request(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in CHART_KEYWORDS)

def is_greeting(question: str) -> bool:
    tokens = re.findall(r"[a-z]+", question.lower())
    if not tokens:
        return False
    if set(tokens) <= GREETING_TOKENS:
        return True
    filler = GREETING_TOKENS | {"there", "ya", "you", "how", "are", "doing",
                                 "good", "morning", "evening", "afternoon"}
    return len(tokens) <= 3 and set(tokens) <= filler

# ─────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────
def sanitize_context(text: str) -> str:
    return text.replace("|", "¦")

def normalize_citations(text: str) -> str:
    text = re.sub(r"†L\d+(?:-L\d+)?", "", text)
    text = text.replace("【", "[").replace("】", "]")
    text = re.sub(r"\[\s*(?:ref)?\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", "", text)
    text = re.sub(r"\(Source:\s*([^)]+)\)", r"[\1]", text)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def verify_numbers(answer: str, context: str, tol: float = NUMBER_TOL) -> list:
    clean = re.sub(r"\[[^\]]*\]", "", answer)
    pattern = r"[+-]?\d+\.\d+|[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"

    def parse(nums):
        out = []
        for n in nums:
            try:
                out.append(float(n.replace(",", "")))
            except ValueError:
                pass
        return out

    ctx_nums = parse(re.findall(pattern, context))
    unverified = []
    for n_str in re.findall(pattern, clean):
        try:
            n = float(n_str.replace(",", ""))
        except ValueError:
            continue
        if any(abs(n - c) / max(abs(c), 1e-9) <= tol for c in ctx_nums):
            continue
        unverified.append(n_str)
    return unverified

# ─────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    raw_sims = EMBEDDINGS_NORM @ q_emb
    raw_max  = float(np.max(raw_sims))

    sims  = raw_sims.copy()
    hints = detect_intents(query)

    # Applicability questions benefit from methodology + model_card context
    if is_applicability_question(query):
        hints.update(["methodology", "model_summary"])

    for i, c in enumerate(CHUNKS):
        if c["type"] in hints:
            sims[i] += 0.20
        if c["type"] == "report_page":
            sims[i] -= 0.15

    top_idx = list(np.argsort(sims)[::-1][:top_k])

    for ctype in hints:
        top_types = {CHUNKS[i]["type"] for i in top_idx}
        if ctype in top_types:
            continue
        type_hits = [(i, sims[i]) for i, c in enumerate(CHUNKS)
                     if c["type"] == ctype and i not in top_idx]
        if not type_hits:
            continue
        best_i = int(max(type_hits, key=lambda x: x[1])[0])
        top_idx = sorted(top_idx[:-1] + [best_i], key=lambda i: -sims[i])

    results = [{
        "id":        CHUNKS[i]["id"],
        "type":      CHUNKS[i]["type"],
        "title":     CHUNKS[i]["title"],
        "text":      CHUNKS[i]["text"],
        "score":     float(sims[i]),
        "raw_score": float(raw_sims[i]),
    } for i in top_idx]
    return results, raw_max, hints

def build_search_query(question: str, history: list) -> str:
    """Expand short/ambiguous follow-ups using the last user turn."""
    if not history:
        return question
    tokens = re.findall(r"[a-z]+", question.lower())
    is_short = len(tokens) <= AMBIGUOUS_MAX_TOKENS
    has_hint = bool(detect_intents(question))
    if not (is_short or not has_hint):
        return question
    # Grab last user question from history (roles: user / assistant alternating)
    last_user_q = None
    for m in reversed(history):
        if m["role"] == "user":
            last_user_q = m["content"]
            break
    if not last_user_q:
        return question
    return f"{last_user_q} {question}"

# ─────────────────────────────────────────────────────────────────────────
# Core ask — returns (answer, retrieved, raw_max)
# ─────────────────────────────────────────────────────────────────────────
def ask(question: str, history=None, max_tokens: int = MAX_TOKENS):
    if history is None:
        history = []

    if is_chart_request(question):
        return ("I can't render charts directly, but I can give you the exact numbers "
                "to plot. Ask me for e.g. 'give me the what-if data as a table' or "
                "'give me segment satisfaction as a table'.", [], 0.0)

    if is_greeting(question):
        return ("Ready to answer questions about the satisfaction model.", [], 0.0)

    if not re.search(r"[a-zA-Z]", question):
        return ("The provided model artifacts do not contain this information.", [], 0.0)

    search_q = build_search_query(question, history)
    retrieved, raw_max, hints = retrieve(search_q)

    # Applicability questions are always allowed through (they use prompt rule)
    applicability = is_applicability_question(question) or is_applicability_question(search_q)

    # History-aware gate:
    # - If there's substantive conversation history, don't gate on raw similarity alone.
    # - Applicability questions are never gated.
    has_substantive_history = any(
        m["role"] == "user" and len(m["content"].split()) >= 3 for m in history
    )

    if (not applicability
        and not hints
        and not has_substantive_history
        and raw_max < MIN_RAW_SIM):
        return ("The provided model artifacts do not contain this information.", [], raw_max)

    context = "\n\n".join(
        f"[{c['title']}]\n{sanitize_context(c['text'])}" for c in retrieved
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"][:600]})
    messages.append({
        "role": "user",
        "content": USER_TEMPLATE.format(context=context, question=question),
    })

    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
        )
        answer = resp.choices[0].message.content or ""
    except Exception as e:
        return (f"LLM error: {e}", retrieved, raw_max)

    answer = normalize_citations(answer)

    unverified = verify_numbers(answer, context)
    if unverified:
        answer += ("\n\n_⚠️ Could not verify against retrieved sources: "
                   + ", ".join(sorted(set(unverified))) + "_")

    return answer, retrieved, raw_max

# ─────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────
st.title("🛫 Airline Satisfaction RAG")
st.caption("Ask questions grounded in coefficients, SHAP, what-if analysis, and the executive report.")

with st.sidebar:
    st.header("About")
    st.markdown(
        "**Model**: Logistic Regression (L2, C=1.0)  \n"
        "**CV AUC**: 0.9423  \n"
        "**Public LB**: 0.94191  \n"
        "**Corpus**: 32 chunks  \n"
        "**LLM**: Groq openai/gpt-oss-20b"
    )
    st.divider()
    st.subheader("Suggested questions")
    sample_qs = [
        "What is the AUC of the model?",
        "Which service lever has the highest priority?",
        "What are the limitations of this analysis?",
        "How satisfied are Business class Business travelers?",
        "What is the odds ratio for Customer Type?",
        "What should the airline do to improve satisfaction?",
        "Is Age a driver of satisfaction?",
        "What is the test set size?",
        "Can this approach apply to insurance NPS?",
    ]
    for q in sample_qs:
        if st.button(q, use_container_width=True):
            st.session_state.pending_q = q

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def process(q: str):
    history = st.session_state.messages[-2 * HISTORY_TURNS:] if st.session_state.messages else []

    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Searching corpus..."):
            answer, retrieved, raw_max = ask(q, history=history)
        st.markdown(answer)
        if retrieved:
            with st.expander(f"Sources ({len(retrieved)}) — top raw similarity {raw_max:.3f}"):
                for c in retrieved:
                    st.markdown(
                        f"**{c['title']}** (`{c['type']}`) — "
                        f"score {c['score']:.3f} · raw {c['raw_score']:.3f}"
                    )
        st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.get("pending_q"):
    q = st.session_state.pending_q
    st.session_state.pending_q = None
    process(q)

if user_q := st.chat_input("Ask anything about the model..."):
    process(user_q)
