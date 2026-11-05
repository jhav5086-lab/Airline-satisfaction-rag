import re
import json
import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sentence_transformers import SentenceTransformer
from groq import Groq

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Airline Satisfaction RAG", page_icon="🛫", layout="wide")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 **Missing API Key:** add GROQ_API_KEY to Streamlit secrets.")
    st.stop()

if not os.path.exists("./embeddings/embeddings.npy") or not os.path.exists("./embeddings/chunks.json"):
    st.error("🚨 **Missing Files:** could not find ./embeddings/ folder.")
    st.stop()

MODEL_PREFERENCES = [
    st.secrets.get("LLM_MODEL") if st.secrets.get("LLM_MODEL") else None,
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]
MODEL_PREFERENCES = [m for m in MODEL_PREFERENCES if m]
EMBED_MODEL = st.secrets.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K = 4
MIN_RAW_SIM = 0.10
HISTORY_TURNS = 2
HISTORY_MSG_CHARS = 500
MAX_TOKENS_DEFAULT = 800
MAX_TOKENS_VISUAL = 2000
INTENT_BOOST_MULT = 2.0
INTENT_BOOST_ADD = 0.20
REPORT_PENALTY = 0.15
MAX_RETRIES = 2
MAX_WAIT_SECONDS = 120

SELF_CHECK_TRIGGERS = (
    "confirm", "prove", "proves", "right?", "correct?", "is that true",
    "definitely", "always", "never", "guarantee", "means that", "so we should",
)

META_FOLLOWUPS = ("why", "why?", "explain", "how come", "what do you mean",
                  "what?", "huh", "why not", "why not?", "which", "how")

# =============================================================================
# BACKEND + MODEL RESOLUTION
# =============================================================================
@st.cache_resource
def load_backend():
    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = np.load("./embeddings/embeddings.npy")
    with open("./embeddings/chunks.json") as f:
        chunks = json.load(f)
    groq_client = Groq(api_key=GROQ_API_KEY)
    return embedder, embeddings, chunks, groq_client

@st.cache_resource
def resolve_llm_model(_client, preferences: tuple):
    try:
        available = {m.id for m in _client.models.list().data}
    except Exception as e:
        return None, [], f"Could not list models: {e}"
    for m in preferences:
        if m in available:
            return m, sorted(available), None
    return None, sorted(available), "None of the preferred models are available"

embedder, EMBEDDINGS, CHUNKS, groq_client = load_backend()
EMBEDDINGS_NORM = EMBEDDINGS / (np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True) + 1e-9)

LLM_MODEL, AVAILABLE_MODELS, MODEL_ERROR = resolve_llm_model(groq_client, tuple(MODEL_PREFERENCES))

if LLM_MODEL is None:
    st.error(f"🚨 **No usable LLM model found.** {MODEL_ERROR}")
    if AVAILABLE_MODELS:
        st.info("Available models — set `LLM_MODEL` in secrets:\n\n" +
                "\n".join(f"- `{m}`" for m in AVAILABLE_MODELS))
    st.stop()

# =============================================================================
# SYSTEM PROMPT  (all fixes 3/4/5/6 + 2dp hints)
# =============================================================================
SYSTEM_PROMPT = "\n".join([
    "You are a senior data science consultant for an airline passenger satisfaction model.",
    "",
    "STYLE: 2-4 sentences, direct answer first. Never refuse charts — generate JSON. 'all plots' = multiple JSON blocks.",
    "",
    "GROUNDING: use ONLY facts in CONTEXT. If not in CONTEXT: reply exactly 'The provided model artifacts do not contain this information.'",
    "",
    "PUSHBACK — CRITICAL:",
    "1. Do NOT confirm a false premise because part of it is technically true. State what IS supported and what IS NOT.",
    "2. If the user states a number that contradicts CONTEXT, CORRECT it: 'That figure differs from the artifacts; the correct value is X [Chunk].' Then answer using the correct value. Do NOT refuse unless no relevant artifact exists.",
    "3. If asked to 'confirm' or 'prove', evaluate whether the artifacts support the claim — don't assume true.",
    "",
    "AGGREGATE FEATURES: Service Mean, Service Std, Service Min are AGGREGATES across 33 raw ratings. Their coefficients do NOT describe single-rating effects. If a user claims otherwise, refuse and explain.",
    "",
    "CAUSALITY: SHAP values are model attributions, not causal effects. Do NOT claim a feature 'causes' satisfaction or that 'fixing X will raise satisfaction'. Cite [Limitations & Caveats]. Associations require pilots.",
    "",
    "STATISTICAL SCALE: At n≈700k, all coefficients are statistically significant. Significance ≠ importance. Prioritize by effect size (coefficient, OR, lift), not p-value.",
    "",
    "MODEL A vs B: Model A (predictive) uses all 33 raw ratings. Model B (interpretive) EXCLUDES raw ratings but INCLUDES aggregate statistics (mean/std/min) AND non-rating categorical features (Customer Type, Type of Travel, Class, etc.).",
    "",
    "MEMORY: assistant turns include '(Chart N: title — label=value)' summaries. If user references a chart that doesn't exist, name what exists and explain instead of refusing.",
    "",
    "CHART REFS: in prose use 'Chart 1' — NEVER square brackets. Brackets are citations only.",
    "",
    "RANKING: never call a lever 'highest lift' if another has larger lift. State criterion. Wi-Fi=+5.37pp raw, Online Boarding=+4.99pp raw but #1 on lift-per-cost.",
    "",
    "CROSS-METRIC: do NOT compare odds ratio to mean |SHAP| (different scales). Refuse OR↔SHAP; offer OR↔OR or SHAP↔SHAP.",
    "",
    "SCATTER: list EVERY feature appearing in BOTH retrieved chunks. x=mean|SHAP|, y=odds ratio.",
    "",
    "CHART TYPE: bar for unordered categories (levers, segments, features). line only for ordered axes (time, delay buckets, rank). If user explicitly requests line for unordered categories, provide it + one sentence: 'Note: a bar chart is more appropriate for unordered categories.' Pie coerced to bar if values aren't compositional.",
    "",
    "CAPABILITY QUESTIONS: if user asks 'what can you do', 'what charts can you build', 'what plots are available' — describe in PROSE only. Do NOT emit chart JSON unless they ask for a specific chart.",
    "",
    "REFUSAL FOLLOWUPS: if the previous turn was a refusal and the user asks 'why' / 'explain', explain what specific artifact or data is MISSING — name the exact data type required. Do NOT switch to a different topic from earlier in the conversation.",
    "",
    "FILES: cannot produce files. Say 'I can't generate files, but here is a slide-ready content outline:' then provide outline + chart JSON.",
    "",
    "CHART FORMAT: wrap JSON in ```chart ... ```. Types: bar, line, pie, scatter. The top-level key MUST be \"data\" (not \"rows\"). Bar/line/pie: {\"data\": [{\"label\": str, \"value\": number}]}. Scatter: {\"data\": [{\"label\": str, \"x\": number, \"y\": number}]}.",
    "",
    "CITATIONS: after each factual sentence add [Chunk Title]. No CJK brackets, no [Chart N], no (Source: X)."
])

SELF_CHECK_PROMPT = "\n".join([
    "You are auditing a proposed answer for statistical integrity.",
    "",
    "Check for these FAILURES:",
    "1. CONFIRMING A FALSE PREMISE — did the answer say 'yes' / 'confirmed' / 'exactly' to a claim CONTEXT doesn't support?",
    "2. CAUSAL OVERCLAIM — did it say a feature 'causes', 'proves', 'guarantees', or 'will definitely raise' satisfaction?",
    "3. AGGREGATE CONFLATION — did it treat Service Mean/Std/Min as if single-rating effects?",
    "4. ACCEPTED FALSE NUMBERS — did it use a user-provided number that isn't in CONTEXT? Correct it instead.",
    "5. MISSING CITATIONS — does any factual claim lack [Chunk Title]?",
    "",
    "If any failure: REWRITE the answer to fix it. If no failures: return UNCHANGED. Return ONLY the answer text."
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
    "caveats": ["limitation", "caveat", "risk", "weakness", "problem", "significant", "causal", "synthetic"],
    "business_levers": ["lever", "priority", "impact", "what-if", "improve"],
    "segments": ["segment", "cohort", "business traveler", "loyal"],
    "model_summary": ["auc", "accuracy", "score", "brier", "dataset", "rows", "test"],
    "methodology": ["method", "trained", "halo", "collinear", "two models", "regulariz"],
    "feature_importance": ["shap", "importance", "driver"],
    "coefficients": ["coefficient", "odds ratio", "odds", "p-value", "p value", "significant"],
    "recommendations": ["recommend", "suggest", "what to do"],
}

ANAPHORA_TOKENS = ("previous", "that chart", "this chart", "the chart", "above", "last one", "last chart")

def detect_intents(query: str) -> set:
    q = query.lower()
    return {t for t, kws in INTENT_KEYWORDS.items() if any(kw in q for kw in kws)}

def retrieve(query: str, top_k: int = TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    raw_sims = EMBEDDINGS_NORM @ q_emb
    raw_max = float(np.max(raw_sims))

    hints = detect_intents(query)
    sims = raw_sims.copy()
    for i, c in enumerate(CHUNKS):
        s = float(raw_sims[i])
        if c["type"] in hints:
            s = s * INTENT_BOOST_MULT + INTENT_BOOST_ADD
        elif c["type"] == "report_page":
            s = s - REPORT_PENALTY
        sims[i] = s

    top_idx = list(np.argsort(sims)[::-1][:top_k])
    results = [{
        "title": CHUNKS[i]["title"], "type": CHUNKS[i]["type"], "text": CHUNKS[i]["text"],
        "score": float(sims[i]), "raw_score": float(raw_sims[i]),
    } for i in top_idx]
    return results, raw_max

def _get_chart_back(question: str) -> int:
    q = question.lower()
    back = 0
    for phrase, n in [
        ("previous to previous", 2), ("the one before that", 2), ("the one before", 2),
        ("one before that", 2), ("one before", 2), ("previous", 1), ("prior", 1),
        ("that chart", 1), ("this chart", 1), ("the chart", 1),
        ("above", 1), ("last one", 1), ("last chart", 1),
    ]:
        if phrase in q and n > back:
            back = n
    return back

def _chart_hint(history: list, question: str) -> str:
    back = _get_chart_back(question)
    charts = [c for m in history for c in (m.get("charts") or [])]
    if not charts:
        return ""
    if back <= 0 or back > len(charts):
        target = charts[-1]
        prefix = f"Only {len(charts)} chart(s) exist. Most recent: "
    else:
        target = charts[-back]
        prefix = f"Referring to chart {len(charts) - back + 1} of {len(charts)}: "
    labels = ", ".join(f"{d.get('label','')}={d.get('value', d.get('x',''))}" for d in target.get("data", [])[:8])
    return f"{prefix}'{target.get('title','chart')}' with data: {labels}."

# FIX 3: refusal follow-up detection
def _last_was_refusal(history: list) -> bool:
    for m in reversed(history):
        if m.get("role") == "assistant":
            content = (m.get("content") or "").lower()
            return ("not contain this information" in content
                    or "do not contain" in content
                    or "don't contain" in content)
    return False

def build_search_query(question: str, history: list) -> str:
    if not history:
        return question
    q_lower = question.lower().strip()
    # FIX 3: meta-followup after refusal → return "" to trigger META mode
    if _last_was_refusal(history) and q_lower in META_FOLLOWUPS:
        return ""
    if len(question.split()) < 4 or any(t in q_lower for t in ANAPHORA_TOKENS):
        recent = " ".join(m.get("content", "") for m in history[-6:] if m.get("role") == "user").strip()
        return (_chart_hint(history, question) + " " + recent + " " + question).strip()
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
# VERIFICATION  (FIX 2: broader pattern + rounding tolerance)
# =============================================================================
NUM_PATTERN = re.compile(r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d{2,}\b|\b\d{5,}\b')

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
        raw = m.group(0)
        try:
            v = round(float(raw.replace(',', '')), 4)
        except ValueError:
            continue
        # FIX 2: rounding-aware — 5.37 in answer matches 5.374 in context
        decimals = len(raw.split('.')[1]) if '.' in raw else 0
        if not any(round(c, decimals) == round(v, decimals) for c in ctx_nums):
            unverified.append(raw)
    return unverified

def _norm_title(t: str) -> str:
    t = re.sub(r"\s*\([^)]*\)", "", t)
    t = re.sub(r"[\s—–\-_]+", " ", t)
    return t.strip().lower()

def verify_citations(answer: str, retrieved: list) -> list:
    titles = {_norm_title(c["title"]) for c in retrieved}
    bad = []
    for cite in re.findall(r"\[([^\]\[]+?)\]", answer):
        if "{" in cite or "}" in cite or '"' in cite:
            continue
        if re.fullmatch(r"[\d\s,;]+", cite):
            continue
        if cite.lower().startswith("chart "):
            continue
        if _norm_title(cite) in titles:
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
# GROQ CALL
# =============================================================================
def _track_usage(resp):
    if hasattr(resp, "usage") and resp.usage:
        u = resp.usage
        st.session_state.setdefault("usage", {"prompt": 0, "completion": 0, "total": 0, "calls": 0})
        st.session_state.usage["prompt"] += getattr(u, "prompt_tokens", 0) or 0
        st.session_state.usage["completion"] += getattr(u, "completion_tokens", 0) or 0
        st.session_state.usage["total"] += getattr(u, "total_tokens", 0) or 0
        st.session_state.usage["calls"] += 1

def _groq_call(messages, max_tok):
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = groq_client.chat.completions.create(
                model=LLM_MODEL, messages=messages, temperature=0.2, max_tokens=max_tok,
            )
            _track_usage(resp)
            return resp
        except Exception as e:
            last_err = e
            err = str(e)
            is_rate = ("429" in err) or ("rate_limit" in err.lower()) or ("tokens per day" in err.lower())
            if is_rate and attempt < MAX_RETRIES:
                m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", err)
                wait = 30 * (attempt + 1)
                if m:
                    wait = int(m.group(1) or 0) * 60 + float(m.group(2) or 0)
                if wait > MAX_WAIT_SECONDS:
                    raise RuntimeError(f"RATE_LIMIT_LONG:{int(wait)}") from e
                time.sleep(wait)
                continue
            raise
    raise last_err

# =============================================================================
# CORE ASK  (FIX 3: meta-followup handling)
# =============================================================================
def _needs_self_check(question: str) -> bool:
    q = question.lower()
    return any(tok in q for tok in SELF_CHECK_TRIGGERS)

def _ask_impl(question: str, history: list):
    search_q = build_search_query(question, history)
    is_visual_query = any(kw in question.lower() for kw in ["chart", "plot", "graph", "visual", "ppt", "presentation"])

    retrieved = []
    meta_mode = (search_q == "")

    if meta_mode:
        # FIX 3: meta-followup after refusal — skip retrieval, explain the refusal
        last_user_q = next((m.get("content", "") for m in reversed(history) if m.get("role") == "user"), "")
        context = (
            f"META-INSTRUCTION: You just refused the user's previous question: '{last_user_q}'. "
            f"The user is asking 'why'. Explain concisely (2-3 sentences) what SPECIFIC artifact or data would be needed to answer their question. "
            f"Name the exact data type required (e.g., 'ROC curve data (TPR/FPR at each threshold)', 'per-country satisfaction aggregates'). "
            f"Do NOT switch to a different topic from earlier in the conversation. Do NOT re-explain an unrelated answer."
        )
    else:
        retrieved, raw_max = retrieve(search_q)
        if not is_visual_query and not history and raw_max < MIN_RAW_SIM and len(question.split()) > 3:
            return "The provided model artifacts do not contain this information.", [], False
        context = "\n\n".join(f"[{c['title']}]\n{sanitize_context(c['text'])}" for c in retrieved)
        hint = _chart_hint(history, question)
        if hint:
            context = f"CHART CONTEXT: {hint}\n\n{context}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        content = m.get("content") or ""
        if m.get("charts"):
            summaries = [
                f"(Chart {idx}: {c.get('title','chart')} — " +
                ", ".join(f"{d.get('label','')}={d.get('value', d.get('x',''))}" for d in c.get("data", [])[:8]) + ")"
                for idx, c in enumerate(m["charts"], 1)
            ]
            content = (content + "\n" + "\n".join(summaries)).strip()
        if content:
            messages.append({"role": m["role"], "content": content[:HISTORY_MSG_CHARS]})

    messages.append({"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)})
    max_tok = MAX_TOKENS_VISUAL if is_visual_query else MAX_TOKENS_DEFAULT

    try:
        resp = _groq_call(messages, max_tok)
        answer = resp.choices[0].message.content or ""
    except RuntimeError as e:
        if str(e).startswith("RATE_LIMIT_LONG:"):
            wait_s = int(str(e).split(":")[1])
            mins = max(1, wait_s // 60)
            return (f"⏳ **Groq daily token limit reached.** Retry in ~{mins} min. "
                    f"This is **account-level** (200k/day), not session."), [], False
        return f"LLM error: {e}", [], False
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            return "⏳ **Groq rate limit hit.** Check https://console.groq.com/settings/billing", [], False
        return f"LLM error: {e}", [], False

    if _needs_self_check(question) and not is_visual_query and not meta_mode:
        try:
            check_msgs = [
                {"role": "system", "content": SELF_CHECK_PROMPT},
                {"role": "user", "content": (
                    f"ORIGINAL QUESTION:\n{question}\n\n"
                    f"PROPOSED ANSWER:\n{answer}\n\n"
                    f"CONTEXT:\n{context[:4000]}\n\n"
                    f"Rewrite or return unchanged per instructions."
                )},
            ]
            check_resp = _groq_call(check_msgs, max_tok)
            revised = check_resp.choices[0].message.content or answer
            if revised.strip():
                answer = revised
        except Exception:
            pass

    return normalize_citations(answer), retrieved, is_visual_query

@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def _cached_ask(question: str):
    return _ask_impl(question, [])

def ask(question: str, history=None):
    if history is None:
        history = []
    if not history and not st.session_state.get("bypass_cache", False):
        return _cached_ask(question)
    return _ask_impl(question, history)

# =============================================================================
# PARSER  (FIX 1: rows/items/names aliases)
# =============================================================================
def _try_parse_chart(raw: str):
    raw = raw.strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    # FIX 1a: accept alternate top-level keys for the data array
    if "data" not in data:
        for alias in ("rows", "items", "values", "entries"):
            if alias in data and isinstance(data[alias], list):
                data["data"] = data.pop(alias)
                break

    if "data" not in data or "type" not in data:
        return None
    if not isinstance(data["data"], list) or not data["data"]:
        return None

    # FIX 1b: accept name→label aliasing inside rows
    for row in data["data"]:
        if isinstance(row, dict) and "label" not in row:
            if "name" in row:
                row["label"] = row.pop("name")
            elif "category" in row:
                row["label"] = row.pop("category")

    data["type"] = str(data.get("type", "bar")).lower()
    if data["type"] not in ("bar", "line", "pie", "scatter"):
        data["type"] = "bar"

    if data["type"] == "scatter":
        for row in data["data"]:
            if not isinstance(row, dict) or "label" not in row or "x" not in row or "y" not in row:
                return None
            try:
                row["x"] = float(row["x"]); row["y"] = float(row["y"])
            except (TypeError, ValueError):
                return None
    else:
        for row in data["data"]:
            if not isinstance(row, dict) or "label" not in row:
                return None
            if "value" not in row:
                for v_alias in ("count", "amount", "v"):
                    if v_alias in row:
                        row["value"] = row.pop(v_alias)
                        break
            if "value" not in row:
                return None
            try:
                row["value"] = float(row["value"])
            except (TypeError, ValueError):
                return None
    return data

def _find_matching_brace(text: str, start: int) -> int:
    if start >= len(text) or text[start] != "{":
        return -1
    depth, i, in_string, escape = 0, start, False, False
    while i < len(text):
        c = text[i]
        if escape: escape = False
        elif c == "\\": escape = True
        elif c == '"': in_string = not in_string
        elif not in_string:
            if c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0: return i
        i += 1
    return -1

def parse_response(answer: str):
    charts = []
    clean_text = answer
    fenced = re.compile(r'```(?:chart|json)?\s*\n?([\s\S]*?)```', re.MULTILINE)
    def _replace(m):
        d = _try_parse_chart(m.group(1))
        if d:
            charts.append(d); return ""
        return m.group(0)
    clean_text = fenced.sub(_replace, clean_text)
    i = 0
    while i < len(clean_text):
        if clean_text[i] == "{":
            end = _find_matching_brace(clean_text, i)
            if end != -1:
                d = _try_parse_chart(clean_text[i:end + 1])
                if d:
                    charts.append(d)
                    clean_text = clean_text[:i] + clean_text[end + 1:]
                    continue
        i += 1
    clean_text = clean_text.strip()
    if charts and clean_text == "The provided model artifacts do not contain this information.":
        clean_text = ""
    return clean_text, charts

# =============================================================================
# RENDERER
# =============================================================================
def _pie_is_valid(values) -> bool:
    vals = [float(v) for v in values if v is not None]
    if not vals: return False
    s = sum(vals)
    if 50 <= s <= 105: return True
    if 0.9 <= s <= 1.1 and all(v <= 1.05 for v in vals): return True
    return False

def render_chart(chart_data):
    df = pd.DataFrame(chart_data["data"])
    chart_type = chart_data.get("type", "bar").lower()
    x_title = chart_data.get("x_label", "")
    y_title = chart_data.get("y_label", "")
    title = chart_data.get("title", "Chart")

    if chart_type == "pie" and not _pie_is_valid(df["value"].tolist()):
        chart_type = "bar"
    if chart_type in ("bar", "line"):
        df = df.sort_values("value", ascending=False).reset_index(drop=True)

    if chart_type == "pie":
        chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="value", type="quantitative"),
            color=alt.Color(field="label", type="nominal", title=x_title or "Category"),
            tooltip=["label", "value"],
        ).properties(title=title, height=400)
    elif chart_type == "line":
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("label:N", title=x_title, sort=df["label"].tolist(), axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("value:Q", title=y_title), tooltip=["label", "value"],
        ).properties(title=title)
    elif chart_type == "scatter":
        base = alt.Chart(df).encode(x=alt.X("x:Q", title=x_title), y=alt.Y("y:Q", title=y_title))
        points = base.mark_circle(size=120).encode(tooltip=["label", "x", "y"])
        labels = base.mark_text(align="left", dx=8, dy=-8).encode(text="label:N")
        chart = (points + labels).properties(title=title)
    else:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("label:N", title=x_title, sort=df["label"].tolist(), axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("value:Q", title=y_title), tooltip=["label", "value"],
        ).properties(title=title)
    st.altair_chart(chart, use_container_width=True)

# =============================================================================
# UI
# =============================================================================
st.title("🛫 Airline Satisfaction RAG")
st.caption("Ask questions grounded in coefficients, SHAP, and what-if analysis.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "usage" not in st.session_state:
    st.session_state.usage = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.session_state.bypass_cache = st.checkbox("Force fresh response (bypass cache)", value=False)
    if st.button("Clear chat + reset usage", use_container_width=True):
        st.session_state.messages = []
        st.session_state.usage = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
        st.cache_data.clear()
        st.rerun()

    st.markdown("### 🔧 Model config")
    st.caption(f"**LLM:** `{LLM_MODEL}`")
    st.caption(f"Embedding: `{EMBED_MODEL.split('/')[-1]}`")
    st.caption(f"TOP_K: {TOP_K} · Self-check: auto")
    with st.expander("Available models"):
        for m in AVAILABLE_MODELS:
            mark = "✅ " if m == LLM_MODEL else "   "
            st.caption(f"{mark}`{m}`")

    st.markdown("### 📊 Session usage")
    st.metric("Total tokens", f"{st.session_state.usage['total']:,}")
    st.caption(f"Prompt: {st.session_state.usage['prompt']:,} · Completion: {st.session_state.usage['completion']:,} · Calls: {st.session_state.usage['calls']}")
    if st.session_state.usage['calls'] > 0:
        avg = st.session_state.usage['total'] / st.session_state.usage['calls']
        st.caption(f"Avg cost/call: **{avg:,.0f}** tokens")
    st.caption(f"Est. remaining (200k cap): **{max(0, 200_000 - st.session_state.usage['total']):,}**")

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
                    sc = s.get("raw_score", s.get("score", 0.0))
                    st.markdown(f"**{s['title']}** (`{s['type']}`) — raw cosine {sc:.3f}")

def process(q: str):
    history = st.session_state.messages[-2 * HISTORY_TURNS:]
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            raw_answer, retrieved, _ = ask(q, history=history)
            clean_text, charts = parse_response(raw_answer)
            clean_text = apply_verification(clean_text, retrieved)
            if clean_text:
                st.markdown(clean_text)
            for c in charts:
                render_chart(c)
            if retrieved:
                with st.expander(f"Sources ({len(retrieved)})"):
                    for c in retrieved:
                        sc = c.get("raw_score", c.get("score", 0.0))
                        st.markdown(f"**{c['title']}** (`{c['type']}`) — raw cosine {sc:.3f}")
            st.session_state.messages.append({
                "role": "assistant", "content": clean_text,
                "charts": charts, "sources": retrieved,
            })

if user_q := st.chat_input("Ask anything... (e.g., 'generate a bar chart of the what-if levers')"):
    process(user_q)
