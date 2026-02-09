"""Backward-compatible wrapper for the Streamlit headless processor."""

from apps.streamlit.streamlit_headless_processor import *  # noqa: F401,F403
from apps.streamlit.streamlit_headless_processor import main as _main


if __name__ == "__main__":
    _main()
