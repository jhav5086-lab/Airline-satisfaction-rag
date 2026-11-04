import re
import json
import difflib
import os
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# =============================================================================
# CONFIG & SAFETY CHECKS
# =============================================================================
st.set_page_config(page_title="Airline Satisfaction RAG", page_icon="🛫", layout="wide")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 **Missing API Key:** Please go to Settings -> Secrets and add GROQ_API_KEY.")
    st.stop()

if not os.path.exists("./embeddings/embeddings.npy") or not os.path.exists("./embeddings/chunks.json"):
    st.error("🚨 **Missing Files:** Could not find the `./embeddings/` folder.")
    st.stop()

MODEL = "openai/gpt-oss-20b"
TOP_K = 6
MIN_RAW_SIM = 0.10
HISTORY_TURNS = 4

# =============================================================================
# BACKEND LOAD
# =============================================================================
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

# =============================================================================
# DYNAMIC PROMPTS (Safe from syntax errors)
# =============================================================================
SYSTEM_PROMPT = "\n".join([
    "You are a dynamic senior data science consultant for an airline passenger satisfaction model.",
    "",
    "## STYLE",
    "- Answer in 2-4 sentences. Direct answer first. No preamble.",
    "- If the user asks for a chart/plot/graph, you MUST generate a special Chart JSON block (see CHART FORMAT below).",
    "- Never say 'I cannot render charts' — instead, generate the Chart JSON block so the UI can render it.",
    "- If asked a casual question (e.g., 'why so slow', 'hello'), answer naturally in one sentence and pivot back to the model.",
    "",
    "## GROUNDING",
    "- Use ONLY facts from CONTEXT for claims about THIS model.",
    "- If a factual question is not in CONTEXT: reply exactly 'The provided model artifacts do not contain this information.'",
    "",
    "## CHART FORMAT (CRITICAL)",
    "If the user asks for a chart, plot, graph, or visual, output the data in this exact JSON format inside a ```chart code block. NEVER use LaTeX or Markdown tables for charts.",
    "```chart",
    "{",
    '  "title": "Title of the Chart",',
    '  "type": "bar",',
    '  "x_label": "X Axis Name",',
    '  "y_label": "Y Axis Name",',
    '  "data": [',
    '    {"label": "Category A", "value": 10.5},',
    '    {"label": "Category B", "value": 20.2}',
    "  ]",
    "}",
    "```",
    "",
    "## CITATIONS - STRICT FORMAT",
    "- After each factual sentence (except in chart blocks), add ONE citation: [Chunk Title]",
    "- FORBIDDEN: CJK brackets, dagger-L tokens, '(Source: X)'.",
    "",
    "## MODEL A vs MODEL B",
    "- Model A (predictive): uses ALL 33 raw service ratings.",
    "- Model B (interpretive): EXCLUDES all raw ratings. Uses ONLY aggregates."
])

USER_TEMPLATE = "\n".join([
    "CONTEXT CHUNKS:",
    "{context}",
    "",
    "USER QUESTION: {question}",
    "",
    "Grounded answer with [Title] citations (or Chart JSON if requested):"
])

# =============================================================================
# INTENT & RETRIEVAL
# =============================================================================
INTENT_KEYWORDS = {
    "caveats": ["limitation", "caveat", "risk", "weakness", "problem"],
    "business_levers": ["lever", "priority", "impact", "what-if", "improve"],
    "segments": ["segment", "cohort", "business traveler", "loyal"],
    "model_summary": ["auc", "accuracy", "score", "brier", "dataset", "rows"],
    "methodology": ["method", "trained", "halo", "collinear", "two models"],
    "feature_importance": ["shap", "importance", "driver"],
    "coefficients": ["coefficient", "odds ratio", "odds"],
    "recommendations": ["recommend", "suggest", "what to do"],
}

def detect_intents(query: str) -> set:
    q = query.lower()
    hints = set()
    for ctype, kws in INTENT_KEYWORDS.items():
        if any(kw in q for kw in kws):
            hints.add(ctype)
    return hints

def retrieve(query: str, top_k: int = TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    raw_sims = EMBEDDINGS_NORM @ q_emb
    raw_max = float(np.max(raw_sims))

    sims = raw_sims.copy()
    hints = detect_intents(query)

    for i, c in enumerate(CHUNKS):
        if c["type"] in hints:
            sims[i] += 0.20
        if c["type"] == "report_page":
            sims[i] -= 0.15

    top_idx = list(np.argsort(sims)[::-1][:top_k])
    results = [{
        "title": CHUNKS[i]["title"],
        "type": CHUNKS[i]["type"],
        "text": CHUNKS[i]["text"],
        "score": float(sims[i]),
    } for i in top_idx]
    return results, raw_max

def build_search_query(question: str, history: list) -> str:
    if not history:
        return question
    if len(question.split()) < 4:
        last_q = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        return f"{last_q} {question}"
    return question

def sanitize_context(text: str) -> str:
    return text.replace("|", "\u00a6")

def normalize_citations(text: str) -> str:
    text = re.sub(r"\u2020L\d+(?:-L\d+)?", "", text)
    text = text.replace("\u3010", "[").replace("\u3011", "]")
    text = re.sub(r"\[\s*(?:ref)?\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", "", text)
    text = re.sub(r"\(Source:\s*([^)]+)\)", r"[\1]", text)
    return text.strip()

# =============================================================================
# CORE ASK (Dynamic Generation)
# =============================================================================
def ask(question: str, history=None):
    if history is None:
        history = []

    search_q = build_search_query(question, history)
    retrieved, raw_max = retrieve(search_q)

    # Only block if it's completely out of scope
    if raw_max < MIN_RAW_SIM and len(question.split()) > 3:
        return "The provided model artifacts do not contain this information.", []

    context = "\n\n".join(f"[{c['title']}]\n{sanitize_context(c['text'])}" for c in retrieved)

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
            temperature=0.2,
            max_tokens=1500,
        )
        answer = resp.choices[0].message.content or ""
    except Exception as e:
        return f"LLM error: {e}", []

    return normalize_citations(answer), retrieved

# =============================================================================
# DYNAMIC UI RENDERER (Draws actual charts)
# =============================================================================
st.title("🛫 Airline Satisfaction RAG")
st.caption("Ask questions grounded in coefficients, SHAP, and what-if analysis. Try asking for a chart!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def process(q: str):
    history = st.session_state.messages[-2 * HISTORY_TURNS:]
    
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking dynamically..."):
            answer, retrieved = ask(q, history=history)
        
        # Catch the dynamic chart block
        if "```chart" in answer:
            parts = answer.split("```chart")
            text_part = parts[0].strip()
            
            # Extract JSON safely
            chart_json_str = parts[1].split("```")[0].strip()
            
            if text_part:
                st.markdown(text_part)
            
            try:
                chart_data = json.loads(chart_json_str)
                st.write(f"### {chart_data.get('title', 'Chart')}")
                
                # Create DataFrame for Streamlit
                df = pd.DataFrame(chart_data["data"]).set_index("label")
                
                # Safely render chart based on type
                chart_type = chart_data.get("type", "bar")
                if chart_type == "line":
                    st.line_chart(df, y="value")
                else:
                    st.bar_chart(df, y="value")
                    
            except Exception as e:
                st.error(f"Could not render chart: {e}")
                st.code(chart_json_str)
        else:
            # No chart, just standard text
            st.markdown(answer)
            
        if retrieved:
            with st.expander(f"Sources ({len(retrieved)})"):
                for c in retrieved:
                    st.markdown(f"**{c['title']}** (`{c['type']}`) — score {c['score']:.3f}")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

if user_q := st.chat_input("Ask anything... (e.g., 'generate a bar chart of the what-if levers')"):
    process(user_q)
