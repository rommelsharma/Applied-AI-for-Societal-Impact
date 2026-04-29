# Bias-Aware Decision Making RAG Copilot

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) based AI assistant** designed to support better decision-making by identifying cognitive biases and reducing inconsistency (“noise”) in human judgment.

The system is grounded in behavioral science research, referring to focused work on the subject, for example:

* *Noise: A Flaw in Human Judgment* — Daniel Kahneman, Olivier Sibony, Cass Sunstein
* *Thinking, Fast and Slow* — Daniel Kahneman

---

## Objective

To build a practical AI system that:

* Helps users analyze complex situations involving multiple stakeholders
* Identifies potential cognitive biases and judgment noise
* Provides structured, research-backed perspectives to improve decisions

---

## Intended Use

This tool is designed for individuals involved in decision-making, including:

* Leaders and managers
* HR professionals
* Legal and compliance contexts
* Teams handling complex interpersonal situations
* Individuals making high-impact personal or professional decisions

---

## Disclaimer

This system is **not a source of legal, financial, or medical advice**.

It provides:

* Research-informed perspectives
* Structured thinking guidance

It does **not replace professional judgment** or domain expertise.

---

## How It Works

1. User provides a scenario or decision context
2. System retrieves relevant insights from curated literature
3. LLM generates a structured response highlighting:

   * Potential biases
   * Noise in judgment
   * Risks and considerations
   * Suggested approaches

---

## Core Components

* Document ingestion and preprocessing
* Embeddings and vector search
* Retrieval pipeline (RAG)
* LLM-based response generation
* API layer (FastAPI)
* Optional UI (Streamlit)

---

## Project Structure

```id="5n9kq2"
bias-aware-decision-making-rag-copilot/
│
├── app/                # Backend API
├── rag/                # RAG pipeline components
├── data/               # Source and processed content
├── prompts/            # Prompt templates
├── evaluation/         # Test scenarios and metrics
├── frontend/           # UI (optional)
├── notebooks/          # Experiments
├── config/             # Configuration
├── tests/              # Tests
└── README.md
```

---

## Expected Output (Example Structure)

* Situation Summary
* Identified Biases
* Noise Considerations
* Risks
* Suggested Actions
* Supporting Insights

---

## Status

Initial setup and design phase.
Focus areas:

* Bias taxonomy
* Data structuring
* RAG pipeline implementation

---

## Next Steps

* Build ingestion and chunking pipeline
* Implement vector search and retrieval
* Design prompt templates for bias detection
* Add evaluation framework

---

## Author

Rommel Sharma
Enterprise Technology Leader | Applied AI Practitioner
LinkedIn: https://www.linkedin.com/in/rommelsharma/
