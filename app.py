"""
Airline Satisfaction RAG — app.py (v2.4 final)

Bug fixes vs v2.3 (from "grapsh / Chharts / equation" transcript):
  #G  Chart pre-check misses typos ("grapsh", "Chharts")       → fuzzy keyword match
  #H  LLM markdown tables break on |SHAP| column               → prompt: use ¦ not | inside cells
  #I  verify_numbers false-positives on SHAP values            → check against cited chunks too
  #I' verify_numbers false-positives on descriptive thresholds → skip 1-decimal < 1.0
  #J  LaTeX equations render as garbage                        → prompt: ASCII math in code block
  #K  Equation mixes Model A and Model B features              → prompt: A vs B disambiguation
  #L  "how to make it causal" misread as "top levers"          → causal keywords → caveats intent
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
AMBIGUOUS_MAX_TOKENS = 6

CHART_KEYWORDS = (
    "chart", "plot", "graph", "visuali", "draw ", "render image",
    "bar plot", "line plot", "pie chart", "histogram", "scatter",
)
GREETING_TOKENS = {"hi", "hello", "hey", "greetings", "yo", "howdy"}

APPLICABILITY_KEYWORDS = [
    "apply to", "applied to", "reuse", "reusable", "transfer",
    "other domain", "other industr", "different domain", "different industr",
    "insurance", "healthcare", "banking", "retail", "telecom", "e-commerce",
    "can it work for", "could it work for", "generalize", "generaliz",
    "use case", "use-case", "not airline", "non-airline", "pharma", "finance",
]

CAUSAL_KEYWORDS = [
    "causal", "causality", "causal inference", "make it causal",
    "randomiz", "ab test", "a/b test", "controlled experiment",
    "pilot study", "instrumental variable", "counterfactual",
    "do-calculus", "confound", "identification strategy",
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
- If asked whether this approach can be applied to another domain (insurance, banking, retail, pharma, etc.):
  1. State clearly: the model was trained on airline satisfaction data; the artifacts do not cover other industries.
  2. Note which ASPECTS are domain-agnostic: logistic regression with L2, two-model design (predictive + interpretive), SHAP attribution, segment analysis, what-if lever prioritization, number verification.
  3. Do NOT claim results for other domains. Do NOT invent metrics.
- 3–5 sentences. Cite [Model Overview] and [Methodology] where relevant.

## CAUSALITY QUESTIONS
- If asked "how to make it causal", "how to validate causality", "how to prove", etc.:
  1. Clarify the model is observational — provides statistical associations, not causal effects.
  2. Recommend concrete causal-inference methods: randomized controlled trial / A/B test, difference-in-differences, instrumental variables, propensity-score matching, or do-calculus / DAG analysis.
  3. Note that a controlled pilot is the cheapest first step.
  4. Do NOT answer with "top 3 levers to improve satisfaction" — that's a different question.
- 3–5 sentences. Cite [Limitations & Caveats].

## NUMBERS
- Quote exact values. Never round, never estimate, never substitute.
- If CONTEXT shows "+5.02", write "+5.02" — not "+5.0", not "about 5".
- Descriptive ranges or thresholds you construct yourself (e.g., "below +0.5 pp") are acceptable prose — state them as your own framing, not as quoted data.

## CITATIONS — STRICT FORMAT
- After each factual sentence, add ONE citation: [Chunk Title]
- CORRECT: [Model Overview]   [Top 10 SHAP Drivers]   [What-If Business Lever Analysis]
- FORBIDDEN: 【...】 brackets, 【1†L1-L4】, "(Source: X)" — use ONLY [Title]

## TABLES
- Inside any markdown table cell, replace `|` with `¦` so the table does not break.
- Example: write `Mean ¦SHAP¦` — NOT `Mean |SHAP|`.

## EQUATIONS
- Write equations as ASCII inside a fenced code block:
