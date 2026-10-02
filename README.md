# 🛫 Airline Satisfaction RAG

> **A retrieval-augmented chatbot grounded in a trained machine learning model for airline passenger satisfaction prediction.**

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-blue?style=for-the-badge)](https://airline-satisfaction-rag-jmeasrvakbevpbhwjhvjvk.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-gpt--oss--20b-FF6B35?style=flat-square)](https://groq.com/)

---

## 📌 Overview

This project combines a **statistically rigorous logistic regression model** with a **retrieval-augmented generation (RAG) system** to create an interactive business intelligence tool. Stakeholders can ask natural-language questions about the satisfaction model and receive grounded, cited answers — no hallucination, no fabrication.

**Live Demo:** [Airline Satisfaction RAG](https://airline-satisfaction-rag-jmeasrvakbevpbhwjhvjvk.streamlit.app)

---

## 🎯 Business Context

An airline wants to understand **which service levers most influence passenger satisfaction** — and by how much. Rather than a raw model, stakeholders need an **interactive tool** that answers:

- *"Which lever should we prioritize?"*
- *"How does satisfaction vary by segment?"*
- *"What does the model say about delays?"*
- *"What are the limitations of this analysis?"*

This RAG system delivers exactly that — grounded in SHAP values, coefficients, what-if simulations, and an executive report.


---

## ✨ Features

### Model
- **Two-model design** — Model A for prediction, Model B for interpretation
- **Statistically valid coefficients** — full-rank design, real p-values, confidence intervals
- **SHAP analysis** — model-agnostic feature attribution
- **Constraint-aware what-if** — lever lift with cost/time weighting
- **Kaggle submission** — Public LB: **0.94191**

### RAG
- **Grounded answers** — every fact comes from retrieved chunks
- **Verbatim citations** — source chunks shown for every answer
- **Number verification** — post-processing flags any unverified figures
- **Type-aware retrieval** — boosts relevant chunk types based on query intent
- **Refusal behavior** — says "not in artifacts" rather than hallucinating
- **Pushback on false premises** — corrects users who misread the data

### Application
- **Live public URL** — deployed on Streamlit Community Cloud
- **Chat interface** — conversational, history preserved in session
- **Suggested questions** — quick-start buttons for common queries
- **Source transparency** — expandable panel showing retrieved chunks with scores


---

## 🚀 Quick Start

### Use the Live App

Just visit: **[Airline Satisfaction RAG](https://airline-satisfaction-rag-jmeasrvakbevpbhwjhvjvk.streamlit.app)**

Try these questions:
- *"What is the AUC of the model?"*
- *"Which service lever has the highest priority?"*
- *"What are the limitations of this analysis?"*
- *"Why not XGBoost?"*
- *"How satisfied are Business class Business travelers?"*
- *"What is the test set size?"*

### Run Locally

```bash
git clone https://github.com/jhav5086-lab/Airline-satisfaction-rag.git
cd Airline-satisfaction-rag
pip install -r requirements.txt
mkdir -p .streamlit
echo 'GROQ_API_KEY = "your_key_here"' > .streamlit/secrets.toml
streamlit run app.py
```
