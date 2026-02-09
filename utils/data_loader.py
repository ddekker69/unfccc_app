# utils/data_loader.py

import pandas as pd
from pathlib import Path
import streamlit as st
from utils.azure_blob_utils import download_blob
from config import IS_STREAMLIT_CLOUD, AZURE_CONTAINER_NAME, TMP_DIR, get_tmp_path
import numpy as np

# Detect Streamlit Cloud mode
IS_STREAMLIT_CLOUD = False
try:
    IS_STREAMLIT_CLOUD = st.secrets.get("running_on_streamlit", False)
except Exception:
    pass


def is_running_on_streamlit_cloud() -> bool:
    try:
        return st.secrets.get("running_on_streamlit", False)
    except Exception:
        return False

@st.cache_data(show_spinner="Downloading file from Azure...", ttl=3600)
def resolve_path(file_path: str) -> str:
    """Get the path for the file, downloading from Azure if needed."""
    if IS_STREAMLIT_CLOUD:
        local_path = get_tmp_path(file_path)
        if not Path(local_path).exists():
            success = download_blob(
                container_name=AZURE_CONTAINER_NAME,
                blob_name=file_path,
                download_path=local_path
            )
            if not success:
                st.error(f"❌ Could not retrieve {file_path} from Azure Blob Storage.")
                st.stop()
        return local_path
    else:
        return file_path


def validate_dataset(df: pd.DataFrame, clustering_mode: str) -> pd.DataFrame:
    """Ensure dataset contains the required columns, normalize if needed."""
    if clustering_mode == "Country-level":
        # Check for either cluster or country_cluster
        cluster_col_present = 'cluster' in df.columns or 'country_cluster' in df.columns
        text_col_present = 'combined_text' in df.columns or 'text' in df.columns
        
        required_basic = ['country', 'x', 'y']
        missing_basic = [col for col in required_basic if col not in df.columns]
        
        if missing_basic:
            raise ValueError(f"Dataset missing required columns: {missing_basic}")
        if not cluster_col_present:
            raise ValueError("Dataset missing cluster information (need 'cluster' or 'country_cluster' column)")
        if not text_col_present:
            raise ValueError("Dataset missing text information (need 'text' or 'combined_text' column)")
    else:
        required = ['document_id', 'text', 'title', 'cluster', 'status']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

    # Normalize for country-level
    if clustering_mode == "Country-level":
        # Only rename if column doesn't already exist
        if 'cluster' not in df.columns and 'country_cluster' in df.columns:
            df = df.rename(columns={"country_cluster": "cluster"})
        if 'text' not in df.columns and 'combined_text' in df.columns:
            df = df.rename(columns={"combined_text": "text"})
        
        # Add placeholder columns if they don't exist
        if 'document_id' not in df.columns:
            df['document_id'] = df['country']
        if 'title' not in df.columns:
            df['title'] = df['country']
        if 'status' not in df.columns:
            df['status'] = 'ok'

    return df

@st.cache_data(show_spinner="Loading dataset...", ttl=3600)
def load_dataset(path: str, clustering_mode: str) -> pd.DataFrame:
    """Load and validate dataset."""
    resolved_path = resolve_path(path)
    
    # Check if file exists locally
    if not Path(resolved_path).exists():
        st.warning(f"📝 Data file `{path}` not found. Running data preparation scripts...")
        
        # Try to run data preparation
        success = prepare_missing_data(path, clustering_mode)
        if not success:
            st.error("❌ Failed to prepare data. Please run data preparation scripts manually.")
            st.info("""
            **To prepare the data:**
            1. Run `python extract_texts.py` to extract text from documents
            2. Run `python prepare_plot_df.py` to create clustering data
            3. Or check Azure Blob Storage configuration
            """)
            st.stop()
        
        # Try loading again after preparation
        if not Path(resolved_path).exists():
            st.error(f"❌ Data file still not found after preparation: {path}")
            st.stop()
    
    if resolved_path.endswith(".pkl"):
        df = pd.read_pickle(resolved_path)
    elif resolved_path.endswith(".csv"):
        df = pd.read_csv(resolved_path)
    else:
        raise ValueError("Unsupported file format.")
    return validate_dataset(df, clustering_mode)

def prepare_missing_data(path: str, clustering_mode: str) -> bool:
    """Run data preparation scripts to generate missing data files."""
    import subprocess
    import sys
    import os
    
    try:
        # Check if we have the prerequisite data
        extracted_texts_path = "data/extracted_texts.pkl"
        if not os.path.exists(extracted_texts_path):
            st.info("🔄 Running text extraction...")
            result = subprocess.run(
                [sys.executable, "extract_texts.py"],
                capture_output=True,
                text=True,
                cwd=".",
                encoding="utf-8",
            )
            if result.returncode != 0:
                st.error(f"Text extraction failed: {result.stderr}")
                return False
        
        # Run the plot data preparation
        st.info("🔄 Running plot data preparation...")
        result = subprocess.run(
            [sys.executable, "prepare_plot_df.py"],
            capture_output=True,
            text=True,
            cwd=".",
            encoding="utf-8",
        )
        if result.returncode != 0:
            st.error(f"Plot data preparation failed: {result.stderr}")
            return False
        
        st.success("✅ Data preparation completed successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error during data preparation: {e}")
        return False

@st.cache_data(show_spinner="Loading dataset...", ttl=3600)
def load_cluster_data(clustering_mode: str) -> pd.DataFrame:
    """Main entry point for loading cluster data (document/country level)."""
    if clustering_mode == "Country-level":
        path = "data/country_plot_df.pkl"
    else:
        path = "data/plot_df.pkl"
    return load_dataset(path, clustering_mode)
