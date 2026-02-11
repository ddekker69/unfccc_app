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

## Setup (Linux/macOS/Windows)

```bash
git clone <repo-url>
cd unfccc
conda env create -f environment.yml
conda activate unfccc_env
```

Configure `OPENAI_API_KEY` via environment variable or `.streamlit/secrets.toml`.

Compatibility notes:

- `environment.yml` and `requirements.txt` are now the default cross-platform setup.
- `requirements.txt` is UTF-8 encoded (pip-safe on all OSes).
- Optional GPU extras are in `requirements-gpu.txt`.
- `pdfkit` requires `wkhtmltopdf` installed on the host if you want PDF export features.

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

Recommended direct app entrypoint:

```bash
streamlit run apps/streamlit/cluster_qa_app.py
```

## Current App Behavior

- The app uses the **Ultra-Fast RAG** flow as the primary path.
- The sidebar no longer exposes legacy debug controls and legacy inline pipeline-management actions.
- Azure upload/download warnings are expected in local mode when Azure credentials are not configured.

### Model Menu (Local + API)

The UI currently exposes:

- `DeepSeek-R1-Distill-Qwen-14B (Recommended)`
- `DeepSeek-R1-Distill-Qwen-7B (Faster)`
- `DeepSeek-R1-Distill-Llama-8B (Alternative)`
- `Qwen3-4B-Instruct (Fast Recommended)`
- `Phi-4-mini-instruct (Efficient)`
- `Qwen2.5-3B-Instruct (Light)`
- `SmolLM2-1.7B-Instruct (Ultra Light)`
- `TinyLlama-1B (8GB RAM)`
- `gpt-4o`

Notes:

- First use of a model downloads weights from Hugging Face and can be large (especially 7B/14B models).
- Legacy small-model fallbacks may still exist internally for resilience, even if not shown in the main menu.

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
