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
MIN_RAW_SIM = 0.20          # FIX: raised from 0.10
HISTORY_TURNS = 4
MAX_TOKENS_DEFAULT = 1500
MAX_TOKENS_VISUAL = 3000    # FIX: avoid truncating multi-chart answers

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
# SYSTEM PROMPT  (FIX 3: ranking + cross-metric rules + memory rules)
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
    "## CONVERSATION MEMORY",
    "- Assistant turns may include chart summaries of the form '[Chart: title — label=value]'.",
    "- When the user says 'previous chart', 'that chart', 'it', or similar, refer to those summaries.",
    "- For 'previous to previous' or older references, scan further back in conversation history.",
    "",
    "## RANKING RULES (CRITICAL)",
    "- Never call a lever 'highest lift' if another lever in CONTEXT has a larger numeric lift.",
    "- When naming a priority, state the ranking criterion in the same sentence: e.g. 'ranked by lift per unit cost' or 'ranked by raw lift'.",
    "- Wi-Fi has higher raw lift (+5.37pp) than Online Boarding (+4.99pp); Online Boarding ranks first on lift-per-cost. Never imply the opposite.",
    "",
    "## CROSS-METRIC COMPARISONS",
    "- Do NOT compare an odds ratio to a mean |SHAP| value — they are on different scales and mean different things.",
    "- Compare like-with-like: OR vs coefficient, or SHAP vs SHAP.",
    "- If the user asks for an OR-vs-SHAP comparison, state the incompatibility and offer the correct comparison instead.",
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

ANAPHORA_TOKENS = ("previous", "that chart", "this chart", "the chart", "above", "last one", "last chart")

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
        "score": float(sims[i]),          # internal boosted score
        "raw_score": float(raw_sims[i]),  # FIX: user-facing raw cosine
    } for i in top_idx]
    return results, raw_max

# FIX 1: anaphora detection + chart hint so follow-ups don't get refused
def build_search_query(question: str, history: list, last_chart: dict = None) -> str:
    if not history:
        return question
    q_lower = question.lower()
    is_anaphora = any(tok in q_lower for tok in ANAPHORA_TOKENS)
    if len(question.split()) < 4 or is_anaphora:
        recent_user = " ".join(m.get("content", "") for m in history[-6:] if m.get("role") == "user").strip()
        chart_hint = ""
        if last_chart:
            labels = ", ".join(str(d.get("label", "")) for d in last_chart.get("data", [])[:8])
            chart_hint = f"chart {last_chart.get('title', '')}: {labels}. "
        return (chart_hint + recent_user + " " + question).strip()
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
# FIX 2: VERIFICATION (numeric + citation)
# =============================================================================
NUM_PATTERN = re.compile(r'\b(\d{1,3}(?:,\d{3})+|\d+\.\d{3,}|\d{5,})\b')

def _extract_nums(text: str) -> set:
    out = set()
    for n in re.findall(r'\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+', text):
        try:
            out.add(round(float(n.replace(',', '')), 4))
        except ValueError:
            pass
    return out

def verify_numbers(answer: str, retrieved: list) -> list:
    ctx_nums = set()
    for c in retrieved:
        ctx_nums |= _extract_nums(c["text"])
    unverified = []
    for m in NUM_PATTERN.finditer(answer):
        raw = m.group(1)
        try:
            v = round(float(raw.replace(',', '')), 4)
        except ValueError:
            continue
        if not any(abs(v - c) <= max(0.001, abs(v) * 0.001) for c in ctx_nums):
            unverified.append(raw)
    return unverified

def verify_citations(answer: str, retrieved: list) -> list:
    titles = {c["title"] for c in retrieved}
    bad = []
    for cite in re.findall(r'\[([^\]]+)\]', answer):
        if cite in titles:
            continue
        if re.fullmatch(r'[\d\s,;]+', cite):  # numeric refs, ignore
            continue
        bad.append(cite)
    return bad

def apply_verification(text: str, retrieved: list) -> str:
    if not text or not retrieved:
        return text
    if text.strip() == "The provided model artifacts do not contain this information.":
        return text
    warnings = []
    unverified = verify_numbers(text, retrieved)
    if unverified:
        warnings.append(f"⚠️ **Unverified figures:** {', '.join(sorted(set(unverified)))} — not found verbatim in retrieved chunks.")
    bad = verify_citations(text, retrieved)
    if bad:
        warnings.append(f"⚠️ **Uncited sources:** {', '.join(f'[{b}]' for b in sorted(set(bad)))} — not among retrieved chunks.")
    if warnings:
        return text + "\n\n" + "  \n".join(f"> {w}" for w in warnings)
    return text

# =============================================================================
# CORE ASK
# =============================================================================
def ask(question: str, history=None, last_chart=None):
    if history is None:
        history = []

    search_q = build_search_query(question, history, last_chart=last_chart)
    is_visual_query = any(kw in question.lower() for kw in ["chart", "plot", "graph", "visual", "ppt", "presentation"])
    retrieved, raw_max = retrieve(search_q)

    if not is_visual_query and raw_max < MIN_RAW_SIM and len(question.split()) > 3:
        return "The provided model artifacts do not contain this information.", [], False

    context = "\n\n".join(f"[{c['title']}]\n{sanitize_context(c['text'])}" for c in retrieved)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        content = m.get("content") or ""
        if m.get("charts"):
            # FIX 1: replay chart summaries so follow-ups can reference them
            summaries = []
            for c in m["charts"]:
                pairs = ", ".join(f"{d.get('label', '')}={d.get('value', '')}" for d in c.get("data", [])[:8])
                summaries.append(f"[Chart: {c.get('title', 'chart')} — {pairs}]")
            content = (content + "\n" + "\n".join(summaries)).strip()
        if content:
            messages.append({"role": m["role"], "content": content[:1000]})

    messages.append({"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)})

    max_tok = MAX_TOKENS_VISUAL if is_visual_query else MAX_TOKENS_DEFAULT
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tok,
        )
        answer = resp.choices[0].message.content or ""
    except Exception as e:
        return f"LLM error: {e}", [], False

    return normalize_citations(answer), retrieved, is_visual_query

# =============================================================================
# PARSER & RENDERER
# =============================================================================
def _try_parse_chart(raw: str):
    raw = raw.strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "data" not in data or "type" not in data:
        return None
    if not isinstance(data["data"], list) or not data["data"]:
        return None
    for row in data["data"]:
        if not isinstance(row, dict) or "label" not in row or "value" not in row:
            return None
        try:
            row["value"] = float(row["value"])
        except (TypeError, ValueError):
            return None
    data["type"] = str(data.get("type", "bar")).lower()
    if data["type"] not in ("bar", "line", "pie"):
        data["type"] = "bar"
    return data

def _find_matching_brace(text: str, start: int) -> int:
    """Return index of matching '}' for text[start] == '{', respecting strings. -1 if not found."""
    if start >= len(text) or text[start] != "{":
        return -1
    depth, i, in_string, escape = 0, start, False, False
    while i < len(text):
        c = text[i]
        if escape:
            escape = False
        elif c == "\\":
            escape = True
        elif c == '"':
            in_string = not in_string
        elif not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1

def parse_response(answer: str):
    """Extract chart JSON (fenced + raw) and strip them from text. Robust to multi-chart."""
    charts = []
    clean_text = answer

    # Pass 1: fenced blocks — remove only those that parse as charts
    fenced_pattern = re.compile(r'```(?:chart|json)?\s*\n?([\s\S]*?)```', re.MULTILINE)
    def _replace(m):
        data = _try_parse_chart(m.group(1))
        if data:
            charts.append(data)
            return ""
        return m.group(0)
    clean_text = fenced_pattern.sub(_replace, clean_text)

    # Pass 2: raw JSON in remaining text
    i = 0
    while i < len(clean_text):
        if clean_text[i] == "{":
            end = _find_matching_brace(clean_text, i)
            if end != -1:
                data = _try_parse_chart(clean_text[i:end + 1])
                if data:
                    charts.append(data)
                    clean_text = clean_text[:i] + clean_text[end + 1:]
                    continue
        i += 1

    clean_text = clean_text.strip()
    if charts and clean_text == "The provided model artifacts do not contain this information.":
        clean_text = ""
    return clean_text, charts

def _pie_is_valid(values) -> bool:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return False
    s = sum(vals)
    if 50 <= s <= 105:                              # percentage composition
        return True
    if 0.9 <= s <= 1.1 and all(v <= 1.05 for v in vals):  # fractional composition
        return True
    return False

def render_chart(chart_data):
    df = pd.DataFrame(chart_data["data"])
    chart_type = chart_data.get("type", "bar").lower()
    x_title = chart_data.get("x_label", "")
    y_title = chart_data.get("y_label", "")
    title = chart_data.get("title", "Chart")

    # FIX 4: coerce misleading pies to bars
    if chart_type == "pie" and not _pie_is_valid(df["value"].tolist()):
        chart_type = "bar"

    if chart_type in ("bar", "line"):
        df = df.sort_values("value", ascending=False).reset_index(drop=True)  # FIX 4: deterministic order

    if chart_type == "pie":
        chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="value", type="quantitative"),
            color=alt.Color(field="label", type="nominal", title=x_title or "Category"),
            tooltip=["label", "value"],
        ).properties(title=title, height=400)
    elif chart_type == "line":
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("label:N", title=x_title, sort=df["label"].tolist(), axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("value:Q", title=y_title),
            tooltip=["label", "value"],
        ).properties(title=title)
    else:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("label:N", title=x_title, sort=df["label"].tolist(), axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("value:Q", title=y_title),
            tooltip=["label", "value"],
        ).properties(title=title)

    st.altair_chart(chart, use_container_width=True)

# =============================================================================
# UI RENDERER
# =============================================================================
st.title("🛫 Airline Satisfaction RAG")
st.caption("Ask questions grounded in coefficients, SHAP, and what-if analysis. Try asking for a pie chart!")

if "messages" not in st.session_state:
    st.session_state.messages = []

def _last_chart_from_history():
    for m in reversed(st.session_state.messages):
        if m.get("charts"):
            return m["charts"][-1]
    return None

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
                    score = s.get("raw_score", s.get("score", 0.0))
                    st.markdown(f"**{s['title']}** (`{s['type']}`) — raw cosine {score:.3f}")

def process(q: str):
    history = st.session_state.messages[-2 * HISTORY_TURNS:]
    last_chart = _last_chart_from_history()

    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            raw_answer, retrieved, _ = ask(q, history=history, last_chart=last_chart)
            clean_text, charts = parse_response(raw_answer)
            clean_text = apply_verification(clean_text, retrieved)  # FIX 2

            if clean_text:
                st.markdown(clean_text)
            for c in charts:
                render_chart(c)
            if retrieved:
                with st.expander(f"Sources ({len(retrieved)})"):
                    for c in retrieved:
                        score = c.get("raw_score", c.get("score", 0.0))
                        st.markdown(f"**{c['title']}** (`{c['type']}`) — raw cosine {score:.3f}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": clean_text,
                "charts": charts,
                "sources": retrieved,
            })

if user_q := st.chat_input("Ask anything... (e.g., 'generate a bar chart of the what-if levers')"):
    process(user_q)
