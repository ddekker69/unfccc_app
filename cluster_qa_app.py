"""Backward-compatible wrapper for the Streamlit UI app.

When this file is executed by Streamlit, we run the real app module via
`runpy` so reruns (e.g. Cmd+Enter) don't hit import-cache blank screens.
"""

if __name__ != "__main__":
    # Preserve backwards-compatible imports for non-entrypoint usage.
    from apps.streamlit.cluster_qa_app import *  # noqa: F401,F403
else:
    import runpy

    runpy.run_module("apps.streamlit.cluster_qa_app", run_name="__main__")
