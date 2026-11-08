# 🛫 Airline Satisfaction RAG

A retrieval-augmented chatbot grounded in a trained machine learning model for airline passenger satisfaction prediction — with a post-processing verification layer that checks every number and citation against retrieved artifacts.

**Live App** · **Python** · **Streamlit** · **Groq**

---

## 📌 Overview

This project combines a statistically interpretable logistic regression model with a retrieval-augmented generation (RAG) system to create an interactive business intelligence tool. Stakeholders can ask natural-language questions about the satisfaction model and receive grounded, cited answers.

The RAG layer includes a **verification pipeline** that:
- Checks every number in the answer against retrieved chunks
- Verifies every `[Chunk Title]` citation matches a real retrieved chunk
- Runs a self-check audit on adversarial queries (`confirm`, `prove`, `right?`) to catch false premises
- Appends visible `⚠️` warnings when verification fails

**Live Demo:** [Airline Satisfaction RAG](https://airline-satisfaction-rag.streamlit.app)

---

## 🎯 Business Context

An airline wants to understand which service levers most influence passenger satisfaction — and by how much. Rather than a raw model, stakeholders need an interactive tool that answers:

- "Which lever should we prioritize?"
- "How does satisfaction vary by segment?"
- "What does the model say about delays?"
- "What are the limitations of this analysis?"

This RAG system delivers exactly that — grounded in SHAP values, coefficients, what-if simulations, and an executive report.

---

## ✨ Features

### Model
- **Two-model design** — Model A (predictive, all 33 raw ratings) and Model B (interpretive, aggregate + categorical features)
- **Full-rank design** — uniquely identified coefficients with confidence intervals
- **SHAP analysis** — model-agnostic feature attribution
- **Constraint-aware what-if** — lever lift with cost × time weighting
- **Kaggle submission** — Public LB: 0.94191

### RAG
- **Grounded answers** — every factual claim cites a retrieved chunk
- **Number verification** — rounding-aware check against retrieved text; flags any figure not found
- **Citation verification** — normalized title matching; flags hallucinated chunk names
- **Self-check audit** — second LLM pass on adversarial queries catches false premises, causal overclaims, aggregate conflation, and unit-translation errors
- **Type-aware retrieval** — boosts relevant chunk types based on query intent (multiplicative)
- **Refusal behavior** — says "not in artifacts" rather than hallucinating
- **Pushback on false premises** — corrects users who misstate numbers or claim causality from SHAP
- **Meta refusal handling** — "why?" after a refusal explains what specific data is missing, rather than switching topics

### Application
- **Live public URL** — deployed on Streamlit Community Cloud
- **Chat interface** — conversational, history preserved in session
- **Chart generation** — bar / line / pie / scatter rendered via Altair
- **Chart memory** — supports ordinal references (`previous chart`, `previous to previous`)
- **Source transparency** — expandable panel showing retrieved chunks with **raw cosine similarity** (not boosted scores)
- **Session cost tracker** — live token usage and remaining daily budget in the sidebar
- **Model fallback chain** — resolves available Groq models at startup

---

## 🚀 Quick Start

### Use the Live App

Just visit: [Airline Satisfaction RAG](https://airline-satisfaction-rag.streamlit.app)

Try these questions:

- "What is the AUC of Model A?"
- "Which service lever should we prioritize first, and why?"
- "What are the limitations of this analysis?"
- "Why not XGBoost?"
- "How satisfied are Business class Business travelers?"
- "Compare the odds ratio of Customer Type with the mean |SHAP| value of Type of Travel"
- "Generate a bar chart of the top 5 SHAP drivers"

### Run Locally

```bash
git clone https://github.com/jhav5086-lab/Airline-satisfaction-rag.git
cd Airline-satisfaction-rag
pip install -r requirements.txt

mkdir -p .streamlit
echo 'GROQ_API_KEY = "your_key_here"' > .streamlit/secrets.toml

streamlit run app.py
