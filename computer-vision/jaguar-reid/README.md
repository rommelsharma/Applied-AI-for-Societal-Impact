# Jaguar re-identification

Jaguars are the apex predators of the Americas, but tracking their populations across vast landscapes like the Pantanal is a monumental task. Researchers have relied on manual identification by comparing unique spot patterns, which act like a biological fingerprint, to distinguish individuals. With the rise of eco-tourism and citizen science, thousands of photographs are captured each year. However, the volume of data makes manual identification a bottleneck for conservation efforts. The Jaguar Identification Project aims to automate this process. 

This solution is my approach to developing a computer vision model capable of identifying individual jaguars (such as "Medrosa," "Patricia," or "Ousado") from wildlife photographs. 

I loved this project due to its practical environmental and societal impact for wildlife identification and preservation by being part of wildlife demographics research. 
Collectively this research and contributions of code help support the survival of this near-threatened species.

Training uses **MegaDescriptor-L**, **GeM pooling**, **ArcFace**, **LLRD**, **5-fold CV**, and inference uses **TTA**, **query expansion**, and **k-reciprocal re-ranking**.

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


## Quick start

1. **Environment**

   ```bash
   cd jaguar-reid
   python -m venv .venv && source .venv/bin/activate
   pip install -U pip && pip install -r requirements.txt
   ```

2. **Data** — Place training data files under `data/` (or edit `DATA_DIR` / `OUTPUT_DIR` inside `src/jaguar_reid/train.py`).

3. **Run**

   ```bash
   ./scripts/run_train.sh
   ```

   Or manually:

   ```bash
   export PYTHONPATH="$(pwd)/src"
   python -m jaguar_reid.train
   ```

4. **Notebook** — Open `notebooks/jaguar_id_megadescriptor_llrd.ipynb` for the documented workflow.

## Documentation

- **Competition context:** `docs/competition/`
- **Architecture & ablations:** `docs/architecture/` (the architecture report’s code pointer now matches **`src/jaguar_reid/train.py`** in this layout)

See [docs/README.md](./docs/README.md) for an index.

## Provenance

This work is a summary of my submission to a Kaggle competition to identify Jaguars. 
This was an excellent learning on computer vision moving beyond generic computer vision models and needed specialised model application. 
My best code was after the competition submission deadline as I took some time to reach a better approach starting from computer vision fundamentals to more specialised application.

## License

**Kaggle competition rules** and third-party model licenses (**MegaDescriptor**, **timm**, **PyTorch**). 
