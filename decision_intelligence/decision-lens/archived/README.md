# Archived material

This folder holds **legacy inputs, captures, and superseded documentation** that are not part of the slim **live** tree. Runtime code should use canonical paths under `data/` and `evaluation.scenario_catalog`; operators should read **`docs/DESIGN.md`** first.

## Layout

| Path | Contents |
|------|-----------|
| `response_captures/` | Pre–v4 `response/` artefacts and other **captured run outputs** (`*_rag_eval.json`, `connectivity_log.txt`, `sample_results_comparison.md`, `sample_results.json`). |
| `docs/` | **Retired Markdown** — former `docs/*.md` files (`ARCHITECTURE.md`, `code-flow.md`, `fundamental-concepts.md`, `PROJECT_CONTEXT.md`, `PRIVATE_CORPUS_GUIDE.md`, `ml-ops.plan.md`, `phased-rollout-plan.md`, `CODEX_HANDBOOK.md`). |

## Rebuilding the scenario catalog

From the project root:

```bash
python scripts/build_scenarios_catalog.py
```

The script reads from ``data/eval/gold/`` first, then (if present) ``archived/evaluation_jsonsources/`` for legacy scenario JSON only, then ``evaluation/``.

## Documentation policy

- **Live contract:** [`docs/DESIGN.md`](../docs/DESIGN.md) in the repo root `docs/` folder (alongside Word artefacts).
- **Historical detail:** edit only when intentionally preserving a snapshot; otherwise update `DESIGN.md` and code.
