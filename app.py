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
# DYNAMIC PROMPTS
# =============================================================================
SYSTEM_PROMPT = """You are a dynamic senior data science consultant for an airline passenger satisfaction model.

## STYLE
- Answer in 2-4 sentences. Direct answer first. No preamble.
- If the user asks for a chart, plot, graph, PPT, or visual, you MUST generate Chart JSON blocks.
- Never say 'I cannot render charts' — generate the JSON block so the UI can render it.
- If the user asks for 'all plots', output MULTIPLE JSON blocks.

## GROUNDING
- Use ONLY facts from CONTEXT for claims about THIS model.
- If a factual question is not in CONTEXT: reply exactly 'The provided model artifacts do not contain this information.'

## CHART FORMAT (CRITICAL)
You MUST wrap the JSON in triple backticks. Do NOT forget the opening `{` and closing `}`.
Supported types: 'bar', 'line', 'pie'.
```chart
{
  "title": "Title of the Chart",
  "type": "bar",
  "x_label": "X Axis Name",
  "y_label": "Y Axis Name",
  "data": [
    {"label": "Category A", "value": 10.5},
    {"label": "Category B", "value": 20.2}
  ]
}
