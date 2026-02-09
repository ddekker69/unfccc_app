# Streamlit App Architecture Refactor

## Goal

Decouple user-interface concerns (Streamlit) from runtime services (RAG/config), while preserving existing CLI and import entrypoints.

## New Module Boundaries

```text
unfccc/
├── apps/streamlit/
│   ├── cluster_qa_app.py
│   ├── streamlit_headless_processor.py
│   ├── working_headless_processor.py
│   └── rag_headless_processor.py
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
│   ├── prepare_enhanced_index.py
│   ├── prepare_plot_df.py
│   └── ... (other operational scripts)
└── legacy wrapper modules at root (same original filenames)
```

## Compatibility Strategy

Each original top-level module now acts as a thin wrapper that re-exports from the new location.

Example:

- `unfccc/rag_engine.py` -> `from core.rag.rag_engine import *`

For script-style modules, wrappers also call `main()` under `if __name__ == "__main__":`.

This keeps existing commands functional:

- `streamlit run cluster_qa_app.py`

## Configuration Hardening

`core/config/settings.py` was updated to:

- treat Streamlit as optional (no hard dependency for non-UI execution),
- resolve project root paths correctly from the new location,
- load `.env` from the repository module root instead of `core/config/`.

## Validation

Static syntax validation was run with:

```bash
python -m compileall unfccc/apps unfccc/core
```

## Migration Guidance

New code should import from the new package paths:

- App/UI: `apps.streamlit.*`
- RAG services: `core.rag.*`
- Pipeline helpers: `core.pipeline.*`
- Operational scripts: `scripts.*`
- Shared settings: `core.config.settings`

Legacy top-level imports remain supported for backward compatibility, but should be treated as transitional.
