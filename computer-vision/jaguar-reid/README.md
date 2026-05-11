# Jaguar re-identification

Professional layout for a **Kaggle-style fine-grained re-ID** solution: pairwise similarity for **371** test images (**137,270** pairs), evaluated with **identity-balanced mean Average Precision**. Training uses **MegaDescriptor-L**, **GeM pooling**, **ArcFace**, **LLRD**, **5-fold CV**, and inference uses **TTA**, **query expansion**, and **k-reciprocal re-ranking**.

## Repository layout

```
jaguar-reid/
├── README.md                 # This file
├── LEGACY.md               # Where the frozen original tree lives
├── pyproject.toml            # Package metadata (src layout)
├── requirements.txt          # Pip-style deps (pin in your own lockfile)
├── .gitignore
├── docs/
│   ├── README.md             # Index of Word docs
│   ├── architecture/         # Technical design report (.docx)
│   └── competition/          # Problem & dataset description (.docx)
├── notebooks/
│   └── jaguar_id_megadescriptor_llrd.ipynb
├── src/
│   └── jaguar_reid/
│       ├── __init__.py
│       └── train.py          # Full pipeline (EDA → train → infer)
├── scripts/
│   └── run_train.sh          # Sets PYTHONPATH and runs python -m jaguar_reid.train
├── configs/                  # Place YAML/JSON hyperparameters here over time
├── data/                     # Local competition data (not committed; .gitkeep)
├── outputs/                  # Checkpoints & submissions (not committed; .gitkeep)
└── tests/
    └── test_package_import.py
```

## File mapping (legacy → this repo)

| Legacy location (unchanged originals) | This repository |
|----------------------------------------|-----------------|
| `successful_solution_notebook/successful_v7-jaguar-re-identify-megadescriptor.py` | `src/jaguar_reid/train.py` |
| `successful_solution_notebook/0-817-jaguar-id-megadescriptor-l-with-llrd.ipynb` | `notebooks/jaguar_id_megadescriptor_llrd.ipynb` |
| `successful_solution_notebook/Jaguar Re-ID Architecture Report.docx` | `docs/architecture/Jaguar_Re-ID_Architecture_Report.docx` |
| `successful_solution_notebook/source-docs/overview.docx` | `docs/competition/overview.docx` |
| `successful_solution_notebook/source-docs/dataset_description.docx` | `docs/competition/dataset_description.docx` |

Paths in the first column are relative to  
`jaguar-id/submitted-on-kaggle-tested-on-gcp/submission4/`  
(see [LEGACY.md](./LEGACY.md) for the full relative path from `computer-vision/`).

## Quick start

1. **Environment**

   ```bash
   cd jaguar-reid
   python -m venv .venv && source .venv/bin/activate
   pip install -U pip && pip install -r requirements.txt
   ```

2. **Data** — Place Kaggle competition files under `data/` (or edit `DATA_DIR` / `OUTPUT_DIR` inside `src/jaguar_reid/train.py`).

3. **Run**

   ```bash
   ./scripts/run_train.sh
   ```

   Or manually:

   ```bash
   export PYTHONPATH="$(pwd)/src"
   python -m jaguar_reid.train
   ```

4. **Notebook** — Open `notebooks/jaguar_id_megadescriptor_llrd.ipynb` for the documented Kaggle-oriented workflow.

## Documentation

- **Competition context:** `docs/competition/`
- **Architecture & ablations:** `docs/architecture/` (the architecture report’s code pointer now matches **`src/jaguar_reid/train.py`** in this layout)

See [docs/README.md](./docs/README.md) for an index.

## Provenance

This tree is a **copy** of the solution artifacts from the historical workspace (unchanged originals remain in the legacy folder). Use **`jaguar-reid/`** for **GitHub presentation**; keep the legacy tree for exact historical paths and Kaggle notebooks that still reference them.

## License

Respect **Kaggle competition rules** and third-party model licenses (**MegaDescriptor**, **timm**, **PyTorch**). Add a root `LICENSE` file that matches your intent before publishing.
