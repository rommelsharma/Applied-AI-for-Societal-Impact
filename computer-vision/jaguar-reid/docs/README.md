# Documentation

All files below live under **`jaguar-reid/docs/`** in the portfolio repository.

| Path | Contents |
|------|----------|
| [competition/overview.docx](./competition/overview.docx) | Competition overview, task, evaluation, and recommended approaches |
| [competition/dataset_description.docx](./competition/dataset_description.docx) | File formats, splits, and submission schema |
| [architecture/Jaguar_Re-ID_Architecture_Report.docx](./architecture/Jaguar_Re-ID_Architecture_Report.docx) | Technical architecture: model, training, inference, caching, CV results. **Code reference** in the report points to **`src/jaguar_reid/train.py`** (GitHub layout); legacy filename **`successful_v7-jaguar-re-identify-megadescriptor.py`** is noted there for the original tree. |

For GitHub-friendly reading, export these to PDF or Markdown in a follow-up PR if desired.

## Related paths in this repo

- **Runnable pipeline:** `../src/jaguar_reid/train.py`
- **Notebook:** `../notebooks/jaguar_id_megadescriptor_llrd.ipynb`
- **Run helper:** `../scripts/run_train.sh`
