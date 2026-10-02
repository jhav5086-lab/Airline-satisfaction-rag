import re
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Airline Satisfaction RAG", page_icon="🛫", layout="wide")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
MODEL = "openai/gpt-oss-20b"

@st.cache_resource
def load_backend():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = np.load("./embeddings/embeddings.npy")
    with open("./embeddings/chunks.json") as f:
        chunks = json.load(f)
    groq_client = Groq(api_key=GROQ_API_KEY)
    return embedder, embeddings, chunks, groq_client

embedder, embeddings, chunks, groq_client = load_backend()

SYSTEM_PROMPT = """You are a senior data science consultant answering questions about an airline passenger satisfaction model.

Rules:
1. Answer ONLY using facts from the provided CONTEXT.
2. If the context does not contain the answer, say: "The provided model artifacts do not contain this information."
3. Be concise. Business audience. No filler.
4. Always cite which source chunk(s) you used.
5. For numerical facts, quote the exact number from the context.
6. If the user asks about causality or ROI, explicitly state the model provides associations, not causal effects.
7. Never fabricate numbers, features, or claims not present in the context.
8. When citing numerical values, copy them EXACTLY as they appear. Do not round or approximate.
9. When the context shows a number like "+5.02", write it exactly as "+5.02". Do NOT write "+4.99", "+5.0", or "approximately 5"."""

USER_TEMPLATE = """CONTEXT CHUNKS:
{context}

USER QUESTION: {question}

Provide a grounded answer with citations."""

def retrieve(query, top_k=6):
    q_lower = query.lower()
    type_hints = set()
    if any(w in q_lower for w in ["limitation","caveat","constraint","risk","weakness","problem"]):
        type_hints.add("caveats")
    if any(w in q_lower for w in ["lever","priorit","impact","what-if","improve"]):
        type_hints.add("business_levers")
    if any(w in q_lower for w in ["segment","who","cohort","class","demographic"]):
        type_hints.add("segments")
    if any(w in q_lower for w in ["auc","accuracy","performance","score","brier","reliable"]):
        type_hints.add("model_summary")
    if any(w in q_lower for w in ["methodology","method","how","built","trained","approach",
                                   "data","features","engineered","why two models",
                                   "halo","collinear","imputation","bootstrap","cross-validation"]):
        type_hints.update(["methodology", "model_summary"])
    if any(w in q_lower for w in ["test","train","dataset","data size","rows","records","sample"]):
        type_hints.add("model_summary")

    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    q_norm = np.linalg.norm(q_emb) + 1e-9
    sims = (embeddings / emb_norms) @ (q_emb / q_norm)

    for i, c in enumerate(chunks):
        if c["type"] in type_hints:
            sims[i] += 0.20
        if c["type"] == "report_page":
            sims[i] -= 0.15

    top_idx = np.argsort(sims)[::-1][:top_k]
    return [{"id": chunks[i]["id"], "type": chunks[i]["type"],
             "title": chunks[i]["title"], "text": chunks[i]["text"],
             "score": float(sims[i])} for i in top_idx]

def verify_numbers(answer, context):
    """Return data-like numbers in answer not in context (with 5% rounding tolerance)."""
    pattern = r'\b\d+\.\d+\b|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b'
    answer_nums = re.findall(pattern, answer)
    ctx_raw = re.findall(pattern, context)
    ctx_nums = []
    for n in ctx_raw:
        try:
            ctx_nums.append(float(n.replace(',', '')))
        except ValueError:
            continue
    unverified = []
    for n_str in answer_nums:
        try:
            n = float(n_str.replace(',', ''))
        except ValueError:
            continue
        if n < 1 and '.' not in n_str:
            continue
        if any(abs(n - c) / max(abs(c), 1e-9) < 0.05 for c in ctx_nums):
            continue
        unverified.append(n_str)
    return unverified

def ask(question):
    retrieved = retrieve(question)
    context = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in retrieved)
    resp = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
        ],
        temperature=0.1,
        max_tokens=700,
    )
    answer = resp.choices[0].message.content
    unverified = verify_numbers(answer, context)
    if unverified:
        answer += f"\n\n_⚠️ Note: the following numbers could not be verified against retrieved sources: {', '.join(set(unverified))}_"
    return answer, retrieved

st.title("🛫 Airline Satisfaction RAG")
st.caption("Ask questions grounded in coefficients, SHAP, what-if analysis, and the executive report.")

with st.sidebar:
    st.header("About")
    st.markdown("**Model**: Logistic Regression (L2, C=1.0)  \n**CV AUC**: 0.9423  \n**Public LB**: 0.94191  \n**Corpus**: 32 chunks  \n**LLM**: Groq openai/gpt-oss-20b")
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

def process(q):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching corpus..."):
            ans, retrieved = ask(q)
        st.markdown(ans)
        with st.expander("Sources"):
            for c in retrieved:
                st.markdown(f"**{c['title']}** ({c['type']}) — score {c['score']:.3f}")
        st.session_state.messages.append({"role": "assistant", "content": ans})

if "pending_q" in st.session_state and st.session_state.pending_q:
    q = st.session_state.pending_q
    st.session_state.pending_q = None
    process(q)

if user_q := st.chat_input("Ask anything about the model..."):
    process(user_q)
