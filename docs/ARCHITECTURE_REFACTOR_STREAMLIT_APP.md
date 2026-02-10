# Streamlit App Architecture Refactor

## Goal

Separate Streamlit/UI concerns from reusable runtime services and pipeline code while preserving legacy entrypoints.

## Active Module Boundaries

```text
unfccc/
├── apps/streamlit/
│   ├── cluster_qa_app.py
│   └── streamlit_headless_processor.py
├── core/config/
│   └── settings.py
├── core/rag/
│   ├── rag_engine.py
│   ├── rag_pipeline.py
│   └── ultra_fast_rag.py
├── core/pipeline/
│   ├── embedding_store.py
│   ├── pipeline_bootstrap.py
│   └── pdf_file_management.py
├── scripts/
│   ├── automated_pipeline.py
│   ├── build_embeddings.py
│   ├── extract_texts.py
│   ├── prepare_plot_df.py
│   ├── prepare_index.py
│   └── prepare_enhanced_index.py
└── root wrappers (backward compatibility)
```

## Compatibility Strategy

Root-level files (`cluster_qa_app.py`, `rag_engine.py`, `rag_pipeline.py`, `ultra_fast_rag.py`, `config.py`) remain thin wrappers so existing commands still work.

## Import Guidance

Prefer these imports for new code:

- UI: `apps.streamlit.*`
- Runtime retrieval/generation: `core.rag.*`
- Shared config: `core.config.settings`
- Pipeline services: `core.pipeline.*`
- Operational entrypoints: `scripts.*`
