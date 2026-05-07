# Project Context

## Repository
`Applied-AI-for-Societal-Impact`

## Flagship project
`decision_intelligence/bias-aware-decision-making-rag-copilot`

## Vision
Create a polished enterprise-style AI portfolio project centered on decision intelligence. The project should help users reason about biased or noisy judgments by combining:
- a curated knowledge base from books and scholarly sources
- a bias taxonomy
- a retrieval-augmented generation pipeline
- structured outputs that explain reasoning, tradeoffs, and limits

## Target users
- managers
- leaders
- HR professionals
- legal and policy-adjacent reviewers
- individual contributors making difficult judgment calls
- recruiters reviewing candidates

## Main user experience
A user submits a complex scenario. The system responds with:
- likely biases and noise sources
- supporting evidence from the knowledge base
- suggested mitigation steps
- a recommendation framed as decision support, not authoritative advice
- a baseline vs RAG comparison when requested

## Principles
- explainability first
- source traceability
- enterprise-grade documentation
- reproducibility
- avoid overstating model certainty
- do not present outputs as legal or professional advice

## Success criteria
The project is successful if it can:
- ingest multiple PDFs reliably
- preserve source metadata
- produce quality chunks and concept tags
- support retrieval over the curated corpus
- generate credible, grounded decision support responses
- clearly demonstrate improvement from RAG over baseline prompting
