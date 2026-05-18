# Applied-AI-for-Societal-Impact
A curated portfolio of end-to-end AI systems focused on real-world decision intelligence and impact, spanning LLMs, computer vision, audio, and data-driven applications across domains such as behavioral science, environment, and financial analysis.

# Applied AI for Societal Impact

## Overview

This repository represents a curated portfolio of **applied Artificial Intelligence systems designed to address real-world problems across multiple domains**.

With over two decades of experience in enterprise technology—spanning system architecture, large-scale delivery, and business transformation—this work reflects a transition into **AI as an enabling layer for better decision-making, operational efficiency, and measurable impact**.

The focus is not on isolated models, but on **end-to-end AI systems** that integrate:

* Data pipelines
* Intelligent models
* APIs and user interfaces
* Evaluation and impact measurement

---

## Purpose

The purpose of this repository is to:

* Demonstrate how AI can be applied to **complex, real-world decision problems**
* Bridge the gap between **theoretical AI capabilities and production-ready systems**
* Build solutions that are:

  * Practical
  * explainable
  * measurable in impact

---

## Core Themes

This portfolio spans multiple applied domains, unified by a common goal:
**augmenting human and system-level decision-making using AI.**

---

### 1. Decision Intelligence & Behavioral AI

AI systems designed to improve decision quality by incorporating behavioral science and reducing cognitive biases.

**Example:**

* Bias-Aware AI Decision Copilot (LLM + RAG)

---

### 2. Financial & Market Intelligence

Data-driven systems for analyzing trends, identifying signals, and supporting investment or financial decisions.

**Example areas:**

* Stock market analysis and recommendation systems
* Signal detection and forecasting models

---

### 3. Environmental & Wildlife Intelligence

Application of AI to monitor and understand natural ecosystems.

**Example areas:**

* Wildlife detection (Computer Vision)
* Acoustic monitoring (Audio AI)

---

### 4. Audio ML — Human Voice Synthesis & Acoustic Intelligence

AI systems that generate, transform, and analyse audio for practical applications.

**Human voice synthesis (Text-to-Speech):**

Producing natural, broadcast-quality speech from text at a local level — without cloud dependency or commercial licensing restrictions. The focus is on narration-grade output suitable for documentaries, training materials, and professional media workflows, including voice cloning from short reference clips.

**Wildlife identification through sound:**

Acoustic monitoring of natural environments to detect and classify species from their calls and vocalisations. Audio signatures are used to identify presence, behaviour, and population patterns — complementing camera-based wildlife detection with a non-invasive, always-on sensing layer.

---

### 5. Responsible & Explainable AI

Focus on building AI systems that are transparent, fair, and interpretable.

**Key aspects:**

* Bias detection in AI systems
* Explainability techniques
* Responsible AI design practices

---

### 6. Applied Enterprise AI Systems

Design of scalable AI solutions that operate within real-world constraints:

* Latency
* Cost
* Integration
* Security

---

## Engineering Principles

All projects are built with a consistent set of principles:

* **End-to-End System Thinking**
  From data ingestion to user-facing applications

* **Problem-First Approach**
  Clear articulation of real-world use cases and constraints

* **Pragmatic Technology Choices**
  Favoring simplicity, reliability, and maintainability

* **Explainability by Design**
  Ensuring outputs are interpretable and grounded

* **Evaluation & Metrics Driven**
  Measuring performance, cost, latency, and impact

---

## Repository Structure

```id="f3k29d"
Applied-AI-for-Societal-Impact/
│
├── decision-intelligence/
│   └── bias-aware-decision-making-rag-copilot/
│
├── financial-intelligence/
│   └── market-analysis/
│
├── computer-vision/
│   └── jaguar-reid/
│
├── audio-ML/
│   └── Custom_TTS_Model/          # Kokoro-82M + F5-TTS narration engine
│
├── audio-ai/
│   └── acoustic-monitoring/       # (planned)
│
├── shared-components/
│   ├── data-pipelines/
│   ├── evaluation/
│   └── utilities/
│
└── README.md
```

---

## Featured Projects

### Bias-Aware AI Decision Copilot

A local-first Retrieval-Augmented Generation (RAG) system grounded in peer-reviewed decision science and behavioural economics literature, designed to detect cognitive biases and improve decision quality.

| | |
|---|---|
| **Purpose** | Decision-support copilot that grounds LLM responses in evidence from behavioural science — detecting biases, surfacing interventions, and improving reasoning consistency |
| **Retrieval** | Hybrid BM25 + FAISS dense retrieval with RRF fusion (+15–30% recall lift over dense-only); MMR (λ=0.7) for result diversity |
| **Knowledge base** | 8-layer decision-intelligence ontology: cognitive biases, failure modes, interventions, evidence hierarchy, group dynamics, forecasting, uncertainty, and decision frameworks |
| **Generation** | Claude Sonnet on Amazon Bedrock; Amazon Titan Embeddings v2 for semantic indexing |
| **Query routing** | Rule-based intent classifier (recall / comparison / synthesis / mitigation) with per-class MMR λ override |
| **Traceability** | 20+ metadata fields per chunk including all 8 ontology dimensions, source, and similarity score |
| **Synthesis** | Book-level and cross-book synthesis artefacts for epistemic grounding across the literature corpus |
| **Privacy** | Fully local FAISS in-process vector store — no managed vector DB, no data leaves the machine |

Located in `decision-intelligence/bias-aware-decision-making-rag-copilot/`.

---

### Market Intelligence & Analysis (Planned)

AI-driven system for analyzing financial data, identifying patterns, and supporting investment decision-making.

---

### Wildlife Detection (Computer Vision) — Jaguar Re-Identification

A computer vision system for identifying individual jaguars from camera-trap imagery, moving away from manual photo-ID to support conservation monitoring of this near-threatened species.

| | |
|---|---|
| **Task** | Fine-grained re-identification — matching individual jaguars across images by coat pattern, pose, and markings |
| **Backbone** | MegaDescriptor-L / DINOv2 ViT-B-14 transformer, fine-tuned with layer-wise learning rate decay (LLRD) |
| **Metric learning** | ArcFace angular margin loss + batch-hard triplet loss; embeddings projected onto a unit hypersphere |
| **Pooling** | Learnable Generalised Mean (GeM) pooling — outperforms fixed average/max pooling on fine-grained retrieval |
| **Validation** | Stratified 5-fold cross-validation; identity-balanced mean Average Precision (macro mAP) |
| **Performance** | 0.726 mAP on validation set (1,895 training images across 31 jaguar identities) |
| **Inference** | 4-transform test-time augmentation (TTA) × 5-fold ensemble; k-reciprocal re-ranking for score refinement |
| **Dataset** | Highly imbalanced (1–169 images per individual); 371 test images forming 137,270 query–gallery pairs |

Located in `computer-vision/jaguar-reid/`.

---

### Custom TTS Model (Audio ML)

A production-grade, locally deployed Text-to-Speech system built for documentary narration,
training material production, and professional media workflows.

| | |
|---|---|
| **Engines** | Kokoro-82M (Apache 2.0) — 13 built-in voices (EN-US, EN-GB, Hindi); F5-TTS (MIT) — zero-shot voice cloning |
| **Audio quality** | 44.1 kHz stereo WAV · EBU R128 loudness normalisation · noise reduction · parametric EQ |
| **Narration control** | SSML markup: `<pause>`, `<break>`, `<emphasis>`, `<say-as>` |
| **Batch processing** | Up to 500 segments per job with optional concatenation and gap control |
| **Privacy** | Fully local — no cloud calls, no telemetry |
| **Platform** | macOS (MPS/CPU) and Windows WSL2 (CUDA) |

Located in `audio-ML/Custom_TTS_Model/`.

---

### Acoustic Monitoring (Audio AI) — Planned

An audio-based system for detecting environmental signals and biodiversity patterns.

---

## Approach to Problem Solving

Each project follows a structured approach:

1. **Problem Definition**
   Context, stakeholders, and constraints

2. **Solution Design**
   Architecture, tradeoffs, and technology selection

3. **Implementation**
   Data pipelines, models, APIs, and interfaces

4. **Evaluation**
   Accuracy, latency, cost, and usability metrics

5. **Impact Assessment**
   Business or societal value

---

## Perspective

The underlying philosophy of this work is:

> AI delivers the most value when it functions as a **decision augmentation layer**, embedded within real systems and workflows.

The emphasis is on:

* Improving consistency
* Reducing bias and noise
* Enabling better outcomes at scale

---

## Roadmap

* Expansion of decision intelligence systems (multi-agent architectures)
* Development of financial analytics and forecasting models
* Real-time environmental monitoring systems
* Exploration of AI governance and evaluation frameworks

---

## Collaboration

This repository is intended as both a portfolio and a foundation for collaboration.
Discussions, feedback, and contributions are welcome.

---

## Author

Rommel Sharma

Enterprise Technology Leader | Applied AI Practitioner

LinkedIn: https://www.linkedin.com/in/rommelsharma/
