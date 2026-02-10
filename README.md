# UNFCCC Climate QA App

Streamlit application and preprocessing pipeline for exploring UNFCCC climate-policy documents with clustering and retrieval-augmented Q&A.

## Repository Layout

```text
unfccc/
├── apps/streamlit/        # UI and app entrypoints
├── core/config/           # Shared settings and environment loading
├── core/pipeline/         # Data/embedding/document management helpers
├── core/rag/              # Retrieval and generation engines
├── scripts/               # Batch pipeline and index builders
├── data/                  # Input PDFs and generated artifacts
└── *.py                   # Backward-compatible root wrappers
```

## Setup

```bash
git clone <repo-url>
cd unfccc
conda env create -f environment.yml
conda activate unfccc_env
```

Configure `OPENAI_API_KEY` via environment variable or `.streamlit/secrets.toml`.

Optional check:

```bash
python scripts/test_api_setup.py
```

## Full Workflow (From Scratch)

Run in this order:

```bash
python scripts/extract_texts.py
python scripts/build_embeddings.py
python scripts/prepare_plot_df.py
python scripts/prepare_index.py
python scripts/prepare_enhanced_index.py
```

Then launch the app:

```bash
streamlit run cluster_qa_app.py
```

## One-Command Pipeline

For end-to-end processing (including optional app launch):

```bash
python scripts/automated_pipeline.py
```

Useful flags:

```bash
python scripts/automated_pipeline.py --offline
python scripts/automated_pipeline.py --skip-app
python scripts/automated_pipeline.py --enhanced-only
python scripts/automated_pipeline.py --force
```

## Notes

- `scripts/prepare_enhanced_index.py` builds ultra-fast retrieval artifacts used by enhanced RAG mode.
- Root-level modules (`cluster_qa_app.py`, `rag_engine.py`, etc.) are compatibility wrappers; new code should import from `apps.*`, `core.*`, and `scripts.*`.
- Generated artifacts are expected under `data/`, `indexes/`, `embeddings/`, `indexes_enhanced/`, and `embeddings_enhanced/`.
