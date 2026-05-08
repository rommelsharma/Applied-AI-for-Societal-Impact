# Next Steps

## Phase 1 complete or nearly complete
- PDF parsing
- front/back matter cleanup during chunking
- taxonomy-driven concept extraction
- knowledge enrichment
- author/title/year metadata extraction
- Bedrock provider layer

## Next implementation steps
### 1. Validate Bedrock runtime access
- confirm Claude chat calls succeed from the local environment
- confirm Titan embedding calls succeed from the local environment
- finalize local environment instructions in `.env.example`

### 2. Build retrieval artifacts
- generate vectors for each knowledge chunk
- persist vectors and metadata together
- write FAISS index when available and keep numpy fallback for local dev

### 3. Tune retriever quality
- calibrate top-k selection
- refine concept/domain filtering
- inspect false positives and empty-result cases

### 4. Wire RAG into the copilot
- inject retrieved context into the system prompt
- keep the expanded taxonomy aligned with prompt usage
- preserve traceability of supporting chunks

### 5. Add comparison mode scoring
- baseline LLM response without retrieval
- RAG-enhanced response with retrieval
- save scenario-by-scenario comparison results

### 6. Expand evaluation
- retrieval relevance checks
- response grounding checks
- simple scorecards for usefulness, clarity, and evidence alignment

### 7. Add a demo interface
- minimal Streamlit UI or FastAPI endpoint
- enough for HR, Legal, and Manager users to test the solution interactively
