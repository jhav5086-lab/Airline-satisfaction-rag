"""
Airline Satisfaction RAG — app.py (v2)

Fixes in this version:
  1. Citation normalization  — 【1】, 【1†L1-L4】, (Source: X) → [Title]
  2. Context sanitization    — pipes in context no longer break LLM tables
  3. Conversation memory     — last 2 turns sent to the LLM
  4. Retrieval floor         — refuses before calling LLM when raw similarity is too low
  5. Chart pre-check         — "draw a chart" answers in 1 sentence, saves tokens
  6. Token budget            — 700 → 1500, no more mid-sentence truncation
  7. Concise style rules     — default 2–4 sentences, tables only on request
"""
import re
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Airline Satisfaction RAG", page_icon="🛫", layout="wide")

GROQ_API_KEY   = st.secrets["GROQ_API_KEY"]
MODEL          = "openai/gpt-oss-20b"
TOP_K          = 6
MAX_TOKENS     = 1500          # was 700 — prevents mid-sentence truncation
MIN_RAW_SIM    = 0.20          # refusal threshold (tune on your golden set)
NUMBER_TOL     = 0.005         # 0.5% tolerance when verifying numbers
HISTORY_TURNS  = 2             # conversation turns sent to LLM

CHART_KEYWORDS = (
    "chart", "plot", "graph", "visuali", "draw ", "render image",
    "bar plot", "line plot", "pie chart", "histogram", "scatter",
)

# ----------------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------------
@st.cache_resource
def load_backend():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = np.load("./embeddings/embeddings.npy")
    with open("./embeddings/chunks.json") as f:
        chunks = json.load(f)
    groq_client = Groq(api_key=GROQ_API_KEY)
    return embedder, embeddings, chunks, groq_client

embedder, EMBEDDINGS, CHUNKS, groq_client = load_backend()
EMBEDDINGS_NORM = EMBEDDINGS / (np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True) + 1e-9)

# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior data science consultant for an airline passenger satisfaction model.

## STYLE
- DEFAULT: answer in 2–4 sentences. Direct answer first. No preamble, no greeting.
- Bullet list ONLY when user asks for "list", "prioritize", "all", or "compare".
- Markdown table ONLY when user asks for "table" or "compare".
- NEVER exceed 120 words unless the user explicitly asks for "full summary" or "everything".
- NEVER repeat the question back.

## GROUNDING
- Use ONLY facts from CONTEXT.
- If the answer is not in CONTEXT: reply exactly "The provided model artifacts do not contain this information."
- If the question is off-topic (math, general knowledge, people, code, news, chit-chat): reply in ONE sentence redirecting to model topics. Nothing else.

## NUMBERS
- Quote exact values from CONTEXT. Never round, never estimate, never substitute.
- If CONTEXT shows "+5.02", write "+5.02" — not "+5.0", not "+4.99", not "about 5".

## CITATIONS — STRICT FORMAT
- After each factual sentence, add ONE citation: [Chunk Title]
- CORRECT: [Model Overview]   [Top 10 SHAP Drivers]   [What-If Business Lever Analysis]
- FORBIDDEN: 【...】 brackets, 【1†L1-L4】, "(Source: X)", anything except [Title]
- If multiple sources support one sentence: [Model Overview; Segment-Level Satisfaction]

## CAPABILITY BOUNDARIES
- You CANNOT draw, render, or produce charts, plots, images, or files.
- If asked for a chart: ONE sentence saying so, then offer the numbers as a table if useful.
- You CANNOT answer general knowledge, math, code, or news.

## AMBIGUOUS QUERIES
- If user says a greeting ("hey", "hello", "hi"): reply ONE sentence: "Ready to answer questions about the satisfaction model." Nothing else.
- If user asks "what do you know?" or similar open-ended question: give EXACTLY 5 bullets, each ≤15 words, each with a [source]. Then ask "Which area should I expand?"
"""

USER_TEMPLATE = """CONTEXT CHUNKS:
{context}

USER QUESTION: {question}

Grounded answer with [Title] citations:"""

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def sanitize_context(text: str) -> str:
    """Replace markdown-significant pipes so tables in context don't leak into output."""
    return text.replace("|", "¦")

def normalize_citations(text: str) -> str:
    """Collapse model-invented citation formats into a single [Title] convention."""
    # Kill the †L1-L4 token noise (academic PDF-style artifacts)
    text = re.sub(r"†L\d+(?:-L\d+)?", "", text)
    # CJK brackets → ASCII
    text = text.replace("【", "[").replace("】", "]")
    # Strip bare numeric citations: [1], [1, 2], [ref1]
    text = re.sub(r"\[\s*(?:ref)?\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", "", text)
    # "(Source: X)" → "[X]"
    text = re.sub(r"\(Source:\s*([^)]+)\)", r"[\1]", text)
    # Cleanup leftovers
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_chart_request(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in CHART_KEYWORDS)

def retrieve(query: str, top_k: int = TOP_K):
    """Returns (ranked results, raw_max_similarity)."""
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    raw_sims = EMBEDDINGS_NORM @ q_emb
    raw_max = float(np.max(raw_sims))

    sims = raw_sims.copy()
    q = query.lower()
    hints = set()
    if any(w in q for w in ["limitation","caveat","constraint","risk","weakness","problem"]):
        hints.add("caveats")
    if any(w in q for w in ["lever","priorit","impact","what-if","improve","recommend"]):
        hints.update(["business_levers", "recommendations"])
    if any(w in q for w in ["segment","cohort","class","business traveler","personal","loyal"]):
        hints.add("segments")
    if any(w in q for w in ["auc","accuracy","performance","score","brier","reliable"]):
        hints.add("model_summary")
    if any(w in q for w in ["method","how","trained","built","approach","feature",
                            "halo","collinear","imput","bootstrap","two models"]):
        hints.update(["methodology", "model_summary"])
    if any(w in q for w in ["shap","importance","driver"]):
        hints.add("feature_importance")
    if any(w in q for w in ["coefficient","odds ratio","odds"]):
        hints.update(["coefficients", "coefficient_detail"])
    if any(w in q for w in ["test set","train set","dataset","rows","records","sample"]):
        hints.add("model_summary")

    for i, c in enumerate(CHUNKS):
        if c["type"] in hints:
            sims[i] += 0.20
        if c["type"] == "report_page":
            sims[i] -= 0.15

    top_idx = np.argsort(sims)[::-1][:top_k]
    results = [{
        "id":         CHUNKS[i]["id"],
        "type":       CHUNKS[i]["type"],
        "title":      CHUNKS[i]["title"],
        "text":       CHUNKS[i]["text"],
        "score":      float(sims[i]),
        "raw_score":  float(raw_sims[i]),
    } for i in top_idx]
    return results, raw_max

def verify_numbers(answer: str, context: str, tol: float = NUMBER_TOL):
    """Return decimal/thousands-formatted numbers in answer that don't match context."""
    clean = re.sub(r"\[[^\]]*\]", "", answer)   # ignore citation brackets
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

# ----------------------------------------------------------------------------
# Core ask
# ----------------------------------------------------------------------------
def ask(question: str, history=None, max_tokens: int = MAX_TOKENS):
    """Returns (answer_text, retrieved_chunks)."""
    if history is None:
        history = []

    # Fast path: chart requests
    if is_chart_request(question):
        return (
            "I can't render charts directly, but I can give you the exact numbers "
            "to plot. Ask me for e.g. 'give me the what-if data as a table' or "
            "'give me segment satisfaction as a table'.",
            [],
        )

    retrieved, raw_max = retrieve(question)

    # Retrieval floor — refuse before burning LLM tokens
    if raw_max < MIN_RAW_SIM:
        return (
            "The provided model artifacts do not contain this information.",
            retrieved,
        )

    context = "\n\n".join(
        f"[{c['title']}]\n{sanitize_context(c['text'])}"
        for c in retrieved
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
        return (f"LLM error: {e}", retrieved)

    answer = normalize_citations(answer)

    unverified = verify_numbers(answer, context)
    if unverified:
        answer += (
            "\n\n_⚠️ Could not verify against retrieved sources: "
            + ", ".join(sorted(set(unverified))) + "_"
        )

    return answer, retrieved

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
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
    # History = last N turns BEFORE we append this new user msg
    history = st.session_state.messages[-2 * HISTORY_TURNS:] if st.session_state.messages else []

    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Searching corpus..."):
            answer, retrieved = ask(q, history=history)
        st.markdown(answer)
        if retrieved:
            top_raw = retrieved[0].get("raw_score", retrieved[0]["score"])
            label = f"Sources ({len(retrieved)}) — top raw similarity {top_raw:.3f}"
            with st.expander(label):
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
