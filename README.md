<div align="center">

# 🛫 Airline Satisfaction RAG

**A retrieval-augmented chatbot grounded in a trained ML model — with a verification layer that catches hallucinations before they reach the user.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Try_it_now-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://airline-satisfaction-rag-qxbdxjugyx9cdkaw9g5tkj.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=for-the-badge)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[**Overview**](#-overview) · [**Features**](#-features) · [**Quick Start**](#-quick-start) · [**Examples**](#-example-behaviors) · [**Limitations**](#️-known-limitations) · [**Roadmap**](#️-roadmap)

</div>

---

## 📌 Overview

This project combines a statistically interpretable **logistic regression model** with a **retrieval-augmented generation (RAG)** system to create an interactive business intelligence tool. Stakeholders can ask natural-language questions about the airline satisfaction model and receive grounded, cited answers.

What makes this RAG different from a standard LLM wrapper:

> **Every factual claim is verified post-generation.** Numbers are checked against retrieved chunks. Citations are validated against real chunk titles. Adversarial queries get a second audit pass. When verification fails, `⚠️` warnings appear in the answer — never silent hallucination.

<table>
<tr>
<td width="50%">

### 🎯 What it does

- Answers grounded in SHAP, coefficients, and what-if simulations
- Generates charts (bar, line, pie, scatter) from verified data
- Pushes back on false premises and misstated numbers
- Refuses to answer from general knowledge
- Explains *why* it refused when asked

</td>
<td width="50%">

### 🛡️ What makes it defensible

- **Numeric verification** — rounding-aware check against retrieved text
- **Citation verification** — normalized title matching
- **Self-check audit** — second LLM pass on adversarial queries
- **Meta-refusal** — "why?" gets specific missing-data explanations
- **Cost-aware** — live token budget in the sidebar

</td>
</tr>
</table>

---

## 🎯 Business Context

An airline wants to understand **which service levers most influence passenger satisfaction** — and by how much. Rather than a static report or a raw model, stakeholders need an interactive tool that answers:

| Stakeholder question | How the RAG answers it |
|---|---|
| *"Which lever should we prioritize?"* | Grounded in what-if lift + cost × time weighting |
| *"How does satisfaction vary by segment?"* | Cited from segment-level satisfaction artifacts |
| *"What does the model say about delays?"* | Coefficient + odds ratio with significance caveat |
| *"What are the limitations?"* | Explicit caveats chunk on causality and halo effect |

---

## ✨ Features

### 🧠 Model

| Feature | Detail |
|---|---|
| **Two-model design** | Model A (predictive, 33 raw ratings) · Model B (interpretive, aggregates) |
| **Full-rank design** | Uniquely identified coefficients with confidence intervals |
| **SHAP analysis** | Model-agnostic feature attribution |
| **Constraint-aware what-if** | Lever lift weighted by cost × time |
| **Kaggle submission** | Public LB: **0.94191** |

### 🔍 RAG Pipeline

| Feature | Detail |
|---|---|
| **Grounded answers** | Every factual claim cites a retrieved chunk |
| **Number verification** | Rounding-aware check against retrieved text |
| **Citation verification** | Normalized title matching against retrieved chunks |
| **Self-check audit** | Second LLM pass on adversarial queries |
| **Type-aware retrieval** | Multiplicative boost for intent-matched chunk types |
| **Refusal behavior** | Says *"not in artifacts"* rather than hallucinating |
| **Pushback** | Corrects false premises, misstated numbers, causal claims from SHAP |
| **Meta-refusal** | Explains *what specific data is missing* when asked "why" |

### 💻 Application

| Feature | Detail |
|---|---|
| **Live deployment** | Streamlit Community Cloud |
| **Chat interface** | Session-preserved history |
| **Charts** | Bar · Line · Pie · Scatter via Altair |
| **Chart memory** | Ordinal references (`previous chart`, `previous to previous`) |
| **Source transparency** | Expandable panel with raw cosine similarity |
| **Session tracker** | Live token usage in the sidebar |
| **Model fallback** | Resolves available Groq models at startup |

---

## 🚀 Quick Start

### Try the Live App

<div align="center">

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airline-satisfaction-rag-qxbdxjugyx9cdkaw9g5tkj.streamlit.app/)
</div>

**Sample questions to try:**
