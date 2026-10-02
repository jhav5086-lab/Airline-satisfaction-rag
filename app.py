import re
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Airline Satisfaction RAG", page_icon="🛫", layout="wide")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
MODEL = "openai/gpt-oss-20b"

@st.cache_resource
def load_backend():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma")
    collection = client.get_collection("airline_satisfaction")
    groq_client = Groq(api_key=GROQ_API_KEY)
    return embedder, collection, groq_client

embedder, collection, groq_client = load_backend()

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
    type_hints = []
    if any(w in q_lower for w in ["limitation","caveat","constraint","risk","weakness","problem"]):
        type_hints.append("caveats")
    if any(w in q_lower for w in ["lever","priorit","impact","what-if","improve"]):
        type_hints.append("business_levers")
    if any(w in q_lower for w in ["segment","who","cohort","class","demographic"]):
        type_hints.append("segments")
    if any(w in q_lower for w in ["auc","accuracy","performance","score","brier","reliable"]):
        type_hints.append("model_summary")
    if any(w in q_lower for w in ["methodology","method","how","built","trained","approach",
                                   "data","features","engineered","why two models",
                                   "halo","collinear","imputation","bootstrap","cross-validation"]):
        type_hints.extend(["methodology", "model_summary"])
    q_emb = embedder.encode([query], convert_to_numpy=True)
    n = top_k * 4 if type_hints else top_k
    results = collection.query(query_embeddings=q_emb.tolist(), n_results=n)
    ranked = []
    for cid, meta, dist, doc in zip(
        results["ids"][0], results["metadatas"][0],
        results["distances"][0], results["documents"][0]
    ):
        sim = 1 - dist
        if meta["type"] in type_hints:
            sim += 0.20
        ranked.append({"id": cid, "type": meta["type"], "title": meta["title"],
                       "text": doc, "score": sim})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]

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
        # Skip tiny numbers — they're usually references, not data
        if n < 1 and '.' not in n_str:
            continue
        # Consider verified if within 5% of any context number
        if any(abs(n - c) / max(abs(c), 1e-9) < 0.05 for c in ctx_nums):
            continue
        unverified.append(n_str)
    return unverified


def ask(question):
    chunks = retrieve(question)
    context = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in chunks)
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
    return answer, chunks

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
            ans, chunks = ask(q)
        st.markdown(ans)
        with st.expander("Sources"):
            for c in chunks:
                st.markdown(f"**{c['title']}** ({c['type']}) — score {c['score']:.3f}")
        st.session_state.messages.append({"role": "assistant", "content": ans})

if "pending_q" in st.session_state and st.session_state.pending_q:
    q = st.session_state.pending_q
    st.session_state.pending_q = None
    process(q)

if user_q := st.chat_input("Ask anything about the model..."):
    process(user_q)
