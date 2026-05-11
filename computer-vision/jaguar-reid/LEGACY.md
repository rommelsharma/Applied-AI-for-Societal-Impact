# Legacy source location

The canonical, unmodified competition workspace (original filenames and paths) lives next to this project under **`computer-vision/`**. From inside **`jaguar-reid/`**, use:

```text
../jaguar-id/submitted-on-kaggle-tested-on-gcp/submission4/successful_solution_notebook/
```

This **`jaguar-reid/`** directory duplicates selected files into a **src / docs / notebooks** layout for portfolio and collaboration. If the two trees diverge, treat **legacy +** `../jaguar-id/.../submission4/README.md` as the historical record unless you intentionally promote this repo to the single source of truth.

## Entrypoint names

| Legacy | `jaguar-reid/` |
|--------|----------------|
| `successful_v7-jaguar-re-identify-megadescriptor.py` | `src/jaguar_reid/train.py` (run as `python -m jaguar_reid.train` with `PYTHONPATH=src`) |
| `0-817-jaguar-id-megadescriptor-l-with-llrd.ipynb` | `notebooks/jaguar_id_megadescriptor_llrd.ipynb` |
