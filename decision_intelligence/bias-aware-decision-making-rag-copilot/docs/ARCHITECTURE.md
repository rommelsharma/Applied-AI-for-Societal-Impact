# Architecture

## Current pipeline (corpus-aware)
```text
PDF in data/corpora/<corpus>/raw/                  # corpus ∈ {public, private, ...}
  -> pdf_parser.py
  -> data/corpora/<corpus>/parsed_text/{book}.txt
  -> data/corpora/<corpus>/parsed_text/{book}.meta.json
  -> chunker.py                                    # profile: dossier | book | auto
  -> data/corpora/<corpus>/chunks/chunks.json
  -> concept_extractor.py
  -> data/corpora/<corpus>/chunks/chunks_with_concepts.json
  -> enrich_chunks.py                              # adds passage_type, decision_phase, chapter_title
  -> data/corpora/<corpus>/knowledge/knowledge_base.json
  -> build_vector_index.py
  -> data/corpora/<corpus>/vector_store/
```

## Planned RAG pipeline
```text
knowledge_base.json
  -> Bedrock embeddings
  -> FAISS / numpy retrieval index
  -> retriever
  -> prompt assembly
  -> Bedrock Claude response
```

## Planned directories
```text
bias-aware-decision-making-rag-copilot/
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── data_pipeline/
├── shared_components/
├── rag/
├── prompts/
├── evaluation/
├── notebooks/
└── README.md
```

## Important design choices
### Metadata sidecar files
Each parsed book should produce both:
- a text file with extracted content
- a JSON sidecar with metadata such as author, title, publication year, and parser notes

### Ingestion cleanup
Chunking should trim front matter and back matter so retrieval is grounded in actual book content rather than copyright pages, tables of contents, notes, or author bios.

### Traceability in chunks
Every chunk should carry:
- unique id
- source file or book slug
- author
- title
- publication year
- chunk index
- word count
- extracted concepts
- cleaning notes
- confidence/importance fields when available

### Taxonomy-driven enrichment
Concept extraction should be driven by external metadata files rather than a small hardcoded keyword map:
- `data/metadata/bias-taxonomy.json`
- `data/metadata/retrieval-concepts.json`

This keeps retrieval concepts and prompt taxonomy aligned.

### Centralized path management
Use shared utilities for:
- project root resolution
- raw data directory
- processed data directory
- ensuring directories exist

### Direct execution support
Scripts should work when run from:
- PyCharm
- terminal
- module invocation like `python -m data_pipeline.build_knowledge_base`

## RAG roadmap
1. validate Bedrock embedding generation for knowledge chunks
2. create and persist FAISS vector index
3. implement retrieval with top-k context selection
4. add optional metadata filters for concept tags and domains
5. wire retrieval into the bias detector / decision copilot
6. add response comparison: baseline vs RAG
7. add evaluation metrics for grounding, precision, and usefulness
