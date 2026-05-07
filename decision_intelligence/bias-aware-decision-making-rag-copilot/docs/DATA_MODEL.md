# Data Model

## Parsed book metadata
Each book should have a metadata sidecar JSON with fields like:
- file_name
- source
- author
- title
- publication_year
- page_count
- character_count
- parser_notes if needed

## Chunk schema
Each chunk should include:
- id
- source
- file_name
- author
- title
- publication_year
- chunk_index
- word_count
- cleaning_notes
- text

## Concept-tagged chunk schema
Add:
- concepts
- concept_confidence

## Enriched knowledge schema
Add:
- summary
- importance
- decision_domains
- keywords
- parser_notes
- cleaning_notes
- source
- author
- traceability fields

## Why this matters
The system should always be able to explain:
- which book the information came from
- which author wrote it
- which chunk supported the answer
- which concept tags led to retrieval
