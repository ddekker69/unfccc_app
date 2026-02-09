"""Backward-compatible wrapper for the Streamlit UI app."""

from apps.streamlit.cluster_qa_app import *  # noqa: F401,F403
from apps.streamlit.cluster_qa_app import main as _main


if __name__ == "__main__":
    _main()
