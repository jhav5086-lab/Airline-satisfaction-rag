import re
import json
import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
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
# DYNAMIC PROMPTS (Safe from copy-paste syntax errors)
# =============================================================================
SYSTEM_PROMPT = "\n".join([
    "You are a dynamic senior data science consultant for an airline passenger satisfaction model.",
    "",
    "## STYLE",
    "- Answer in 2-4 sentences. Direct answer first. No preamble.",
    "- If the user asks for a chart, plot, graph, PPT, or visual, you MUST generate Chart JSON blocks.",
    "- Never say 'I cannot render charts' — generate the JSON block so the UI can render it.",
    "- If the user asks for 'all plots', output MULTIPLE JSON blocks.",
    "",
    "## GROUNDING",
    "- Use ONLY facts from CONTEXT for claims about THIS model.",
    "- If a factual question is not in CONTEXT: reply exactly 'The provided model artifacts do not contain this information.'",
    "",
    "## CHART FORMAT (CRITICAL)",
    "You MUST wrap the JSON in triple backticks. Do NOT forget the opening `{` and closing `}`.",
    "Supported types: 'bar', 'line', 'pie'.",
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
# CORE ASK
# =============================================================================
def ask(question: str, history=None):
    if history is None:
        history = []

    search_q = build_search_query(question, history)
    is_visual_query = any(kw in question.lower() for kw in ["chart", "plot", "graph", "visual", "ppt", "presentation"])
    retrieved, raw_max = retrieve(search_q)

    if not is_visual_query and raw_max < MIN_RAW_SIM and len(question.split()) > 3:
        return "The provided model artifacts do not contain this information.", [], []

    context = "\n\n".join(f"[{c['title']}]\n{sanitize_context(c['text'])}" for c in retrieved)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"][:600]})
    messages.append({"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)})

    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )
        answer = resp.choices[0].message.content or ""
    except Exception as e:
        return f"LLM error: {e}", [], []

    return normalize_citations(answer), retrieved, is_visual_query

# =============================================================================
# PARSER & RENDERER
# =============================================================================
def parse_response(answer: str):
    """Extracts all JSON chart blocks and removes them from the text."""
    # Regex to find wrapped JSON blocks or raw JSON objects
    pattern = r'```(?:json|chart)?\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*?"title"[\s\S]*?\})'
    matches = re.findall(pattern, answer)
    
    charts = []
    clean_text = answer
    
    for match in matches:
        raw_json = match[0] if match[0] else match[1]
        try:
            # Auto-fix if LLM forgot brackets
            if not raw_json.strip().startswith("{"): raw_json = "{" + raw_json
            if not raw_json.strip().endswith("}"): raw_json = raw_json + "}"
            
            data = json.loads(raw_json)
            if "data" in data and "type" in data:
                charts.append(data)
                # Remove this JSON from the text so we don't print it
                clean_text = clean_text.replace(raw_json, "")
        except:
            pass # Not valid JSON, ignore

    # Strip any leftover empty backticks, empty chart tags, or refusal text if charts exist
    clean_text = re.sub(r'```(?:json|chart)?\s*```', '', clean_text).strip()
    if charts and clean_text == "The provided model artifacts do not contain this information.":
        clean_text = "" # If we have charts, hide the refusal text
        
    return clean_text, charts

def render_chart(chart_data):
    """Renders a beautiful, labeled chart using Altair."""
    df = pd.DataFrame(chart_data["data"])
    chart_type = chart_data.get("type", "bar").lower()
    x_title = chart_data.get("x_label", "")
    y_title = chart_data.get("y_label", "")
    title = chart_data.get("title", "Chart")

    if chart_type == "pie":
        chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="value", type="quantitative"),
            color=alt.Color(field="label", type="nominal", title=x_title),
            tooltip=['label', 'value']
        ).properties(title=title, height=400)
    elif chart_type == "line":
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("label:N", title=x_title, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("value:Q", title=y_title),
            tooltip=['label', 'value']
        ).properties(title=title)
    else: # bar
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("label:N", title=x_title, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("value:Q", title=y_title),
            tooltip=['label', 'value']
        ).properties(title=title)

    st.altair_chart(chart, use_container_width=True)

# =============================================================================
# UI RENDERER
# =============================================================================
st.title("🛫 Airline Satisfaction RAG")
st.caption("Ask questions grounded in coefficients, SHAP, and what-if analysis. Try asking for a pie chart!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("charts"):
            for c in msg["charts"]:
                render_chart(c)
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['title']}** (`{s['type']}`) — score {s['score']:.3f}")

def process(q: str):
    # Get last 4 turns for context
    history = st.session_state.messages[-2 * HISTORY_TURNS:]
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            raw_answer, retrieved, is_visual = ask(q, history=history)
            
            # Parse out text and charts
            clean_text, charts = parse_response(raw_answer)
            
            # Render text if any
            if clean_text:
                st.markdown(clean_text)
                
            # Render charts if any
            if charts:
                for c in charts:
                    render_chart(c)
            
            # Render sources
            if retrieved:
                with st.expander(f"Sources ({len(retrieved)})"):
                    for c in retrieved:
                        st.markdown(f"**{c['title']}** (`{c['type']}`) — score {c['score']:.3f}")
            
            # Save to state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": clean_text,
                "charts": charts,
                "sources": retrieved
            })

if user_q := st.chat_input("Ask anything... (e.g., 'generate a bar chart of the what-if levers')"):
    process(user_q)
