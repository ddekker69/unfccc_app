import os
from pathlib import Path
import torch
import logging
from typing import Any, Optional
import tempfile

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional dependency for non-UI flows
    st = None

# === Debug Configuration ===
# Global debug control - can be overridden by environment variable
DEBUG_ENABLED = os.getenv('DEBUG_MODE', 'false').lower() in ['true', '1', 'on', 'yes']

# Fine-grained debug controls
DEBUG_RAG = os.getenv('DEBUG_RAG', str(DEBUG_ENABLED)).lower() in ['true', '1', 'on', 'yes']
DEBUG_EMBEDDING = os.getenv('DEBUG_EMBEDDING', str(DEBUG_ENABLED)).lower() in ['true', '1', 'on', 'yes']
DEBUG_GENERATION = os.getenv('DEBUG_GENERATION', str(DEBUG_ENABLED)).lower() in ['true', '1', 'on', 'yes']
DEBUG_SEARCH = os.getenv('DEBUG_SEARCH', str(DEBUG_ENABLED)).lower() in ['true', '1', 'on', 'yes']
DEBUG_PERFORMANCE = os.getenv('DEBUG_PERFORMANCE', str(DEBUG_ENABLED)).lower() in ['true', '1', 'on', 'yes']

def debug_print(*args, debug_type: str = "general", **kwargs) -> None:
    """
    Global debug printing function that respects debug settings.
    
    Args:
        *args: Arguments to print
        debug_type: Type of debug message ("rag", "embedding", "generation", "search", "performance")
        **kwargs: Additional keyword arguments for print
    """
    debug_mapping = {
        "rag": DEBUG_RAG,
        "embedding": DEBUG_EMBEDDING, 
        "generation": DEBUG_GENERATION,
        "search": DEBUG_SEARCH,
        "performance": DEBUG_PERFORMANCE,
        "general": DEBUG_ENABLED
    }
    
    if debug_mapping.get(debug_type.lower(), DEBUG_ENABLED):
        print(f"[DEBUG-{debug_type.upper()}]", *args, **kwargs)

def debug_streamlit(message: str, level: str = "info", debug_type: str = "general") -> None:
    """
    Global Streamlit debug display function that respects debug settings.
    
    Args:
        message: Debug message to display
        level: Streamlit display level ("info", "success", "warning", "error")
        debug_type: Type of debug message ("rag", "embedding", "generation", "search", "performance")
    """
    debug_mapping = {
        "rag": DEBUG_RAG,
        "embedding": DEBUG_EMBEDDING,
        "generation": DEBUG_GENERATION, 
        "search": DEBUG_SEARCH,
        "performance": DEBUG_PERFORMANCE,
        "general": DEBUG_ENABLED
    }
    
    if not debug_mapping.get(debug_type.lower(), DEBUG_ENABLED):
        return
    
    # Only display if streamlit is available and running in app context.
    if st is None:
        debug_print(message, debug_type=debug_type)
        return

    try:
        if level == "info":
            st.info(f"🔍 **DEBUG-{debug_type.upper()}**: {message}")
        elif level == "success":
            st.success(f"✅ **DEBUG-{debug_type.upper()}**: {message}")
        elif level == "warning":
            st.warning(f"⚠️ **DEBUG-{debug_type.upper()}**: {message}")
        elif level == "error":
            st.error(f"❌ **DEBUG-{debug_type.upper()}**: {message}")
        else:
            st.info(f"🔍 **DEBUG-{debug_type.upper()}**: {message}")
    except Exception:
        # Fallback to print if not in Streamlit context
        debug_print(message, debug_type=debug_type)

def set_debug_mode(enabled: bool = True, debug_types: Optional[list] = None) -> None:
    """
    Runtime function to enable/disable debugging.
    
    Args:
        enabled: Whether to enable debugging
        debug_types: Specific debug types to enable/disable (None for all)
    """
    global DEBUG_ENABLED, DEBUG_RAG, DEBUG_EMBEDDING, DEBUG_GENERATION, DEBUG_SEARCH, DEBUG_PERFORMANCE
    
    if debug_types is None:
        DEBUG_ENABLED = enabled
        DEBUG_RAG = enabled
        DEBUG_EMBEDDING = enabled
        DEBUG_GENERATION = enabled
        DEBUG_SEARCH = enabled
        DEBUG_PERFORMANCE = enabled
    else:
        for debug_type in debug_types:
            if debug_type.lower() == "rag":
                DEBUG_RAG = enabled
            elif debug_type.lower() == "embedding":
                DEBUG_EMBEDDING = enabled
            elif debug_type.lower() == "generation":
                DEBUG_GENERATION = enabled
            elif debug_type.lower() == "search":
                DEBUG_SEARCH = enabled
            elif debug_type.lower() == "performance":
                DEBUG_PERFORMANCE = enabled
            elif debug_type.lower() == "general":
                DEBUG_ENABLED = enabled

def get_debug_status() -> dict:
    """Get current debug status for all categories."""
    return {
        "general": DEBUG_ENABLED,
        "rag": DEBUG_RAG,
        "embedding": DEBUG_EMBEDDING,
        "generation": DEBUG_GENERATION,
        "search": DEBUG_SEARCH,
        "performance": DEBUG_PERFORMANCE
    }

# Configure Python logging based on debug settings
if DEBUG_ENABLED:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
else:
    logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)

# === Model Configuration ===
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # Stable and reliable model

def get_optimal_device():
    """
    Automatically select the best device for sentence transformers.
    Handles MPS conflicts on Apple Silicon while using GPU when available.
    
    Can be overridden with EMBEDDING_DEVICE environment variable:
    - EMBEDDING_DEVICE=cpu (force CPU)
    - EMBEDDING_DEVICE=cuda (force CUDA)
    - EMBEDDING_DEVICE=mps (force MPS)
    """
    # Check for manual override
    override_device = os.getenv('EMBEDDING_DEVICE', '').lower()
    if override_device in ['cpu', 'cuda', 'mps']:
        debug_print(f"Using manual device override: {override_device}", debug_type="embedding")
        return override_device
    
    try:
        # Check for CUDA first (most reliable)
        if torch.cuda.is_available():
            device = 'cuda'
            debug_print(f"Using CUDA GPU: {torch.cuda.get_device_name()}", debug_type="embedding")
            return device
        
        # Check for MPS on Apple Silicon
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Test MPS with a small operation to check stability
            try:
                test_tensor = torch.randn(10, 10, device='mps')
                _ = test_tensor @ test_tensor  # Simple matrix multiplication test
                device = 'mps'
                debug_print("Using Apple Silicon MPS (Metal Performance Shaders)", debug_type="embedding")
                debug_print("If you encounter issues, set EMBEDDING_DEVICE=cpu environment variable", debug_type="embedding")
                return device
            except Exception as e:
                debug_print(f"MPS test failed ({e}), falling back to CPU", debug_type="embedding")
                device = 'cpu'
                return device
        
        # Default to CPU
        else:
            device = 'cpu'
            debug_print("Using CPU (no GPU acceleration available)", debug_type="embedding")
            return device
            
    except Exception as e:
        debug_print(f"Device detection failed ({e}), using CPU", debug_type="embedding")
        return 'cpu'

# Get the optimal device for this system
OPTIMAL_DEVICE = get_optimal_device()

# === Project paths ===
# settings.py lives in core/config/, so repository module root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# === Safely detect if running on Streamlit Cloud ===
if st is None:
    IS_STREAMLIT_CLOUD = False
else:
    try:
        IS_STREAMLIT_CLOUD = st.secrets.get("running_on_streamlit", False)
    except Exception:
        IS_STREAMLIT_CLOUD = False

# === Secure API Key Loading ===
def load_api_keys():
    """
    Securely load API keys from multiple sources:
    1. Streamlit secrets (for cloud deployment)
    2. Environment variables (for local development)
    3. .env file (for local development)
    
    Returns a dictionary with available API keys.
    """
    keys = {}
    
    # Try loading from .env file first (local development)
    env_file_path = PROJECT_ROOT / '.env'
    if env_file_path.exists():
        try:
            with open(env_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Only set in os.environ if not already set
                        if key.strip() not in os.environ:
                            os.environ[key.strip()] = value.strip().strip('"\'')
            debug_print("Loaded environment variables from .env file", debug_type="general")
        except Exception as e:
            debug_print(f"Warning: Could not load .env file: {e}", debug_type="general")
    
    # Load OpenAI API Key
    if IS_STREAMLIT_CLOUD and st is not None:
        try:
            keys['openai'] = st.secrets["OPENAI_API_KEY"]
            debug_print("Loaded OpenAI API key from Streamlit secrets", debug_type="general")
        except KeyError:
            keys['openai'] = None
            debug_print("OpenAI API key not found in Streamlit secrets", debug_type="general")
    else:
        keys['openai'] = os.getenv("OPENAI_API_KEY")
        if keys['openai']:
            debug_print("Loaded OpenAI API key from environment variables", debug_type="general")
        else:
            debug_print("OpenAI API key not found in environment variables", debug_type="general")
    
    # Load Azure Storage Account Name and Key
    if IS_STREAMLIT_CLOUD and st is not None:
        try:
            keys['azure_account_name'] = st.secrets["AZURE_STORAGE_ACCOUNT_NAME"]
            keys['azure_account_key'] = st.secrets["AZURE_STORAGE_ACCOUNT_KEY"]
            debug_print("Loaded Azure credentials from Streamlit secrets", debug_type="general")
        except KeyError:
            keys['azure_account_name'] = None
            keys['azure_account_key'] = None
            debug_print("Azure credentials not found in Streamlit secrets", debug_type="general")
    else:
        keys['azure_account_name'] = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        keys['azure_account_key'] = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        if keys['azure_account_name'] and keys['azure_account_key']:
            debug_print("Loaded Azure credentials from environment variables", debug_type="general")
        else:
            debug_print("Azure credentials not found in environment variables", debug_type="general")
    
    # Build Azure connection string if we have the components
    if keys['azure_account_name'] and keys['azure_account_key']:
        keys['azure_connection_string'] = f"DefaultEndpointsProtocol=https;AccountName={keys['azure_account_name']};AccountKey={keys['azure_account_key']};EndpointSuffix=core.windows.net"
        debug_print("Built Azure connection string from account credentials", debug_type="general")
    else:
        # Try to get connection string directly (fallback)
        if IS_STREAMLIT_CLOUD and st is not None:
            try:
                keys['azure_connection_string'] = st.secrets["AZURE_STORAGE_CONNECTION_STRING"]
            except KeyError:
                keys['azure_connection_string'] = None
        else:
            keys['azure_connection_string'] = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    return keys

# Load all API keys
API_KEYS = load_api_keys()

# Export individual keys for backwards compatibility
OPENAI_API_KEY = API_KEYS.get('openai')
AZURE_STORAGE_ACCOUNT_NAME = API_KEYS.get('azure_account_name') 
AZURE_STORAGE_ACCOUNT_KEY = API_KEYS.get('azure_account_key')
AZURE_CONNECTION_STRING = API_KEYS.get('azure_connection_string')

# === Azure Configuration ===
AZURE_CONTAINER_NAME = "unfccccontainer"
AZURE_MODEL_BLOB_NAME = "models/all-MiniLM-L6-v2.zip"

# === Paths (relative) ===
BASE_DIR = PROJECT_ROOT

PDF_DIR = BASE_DIR / "data" / "unfccc_documents_topic_mitigation"
CSV_PATH = BASE_DIR / "data" / "unfccc_documents_topic_mitigation_paginated.csv"
OUTPUT = BASE_DIR / "data" / "extracted_texts.pkl"
OUTPUT_PATH = "data/extracted_texts.pkl"
OUTPUT_PATH_LEGACY = "data/extracted_texts_legacy.pkl"
UPLOAD_PDF_DIR = "data/uploaded_documents"
CHECKPOINTS_DIR = Path("data/checkpoints")
CACHE_DIR = Path("embeddings_cache")

# === Models and Indexes ===
LOCAL_MODEL_PATH = BASE_DIR / "models" / "all-MiniLM-L6-v2"

# Cross-platform temporary directory
TMP_DIR = tempfile.gettempdir()  # Works on Windows, Mac, and Linux

def get_tmp_path(file_path):
    """Get cross-platform temporary file path."""
    return str(Path(TMP_DIR) / Path(file_path).name)

# === Required Columns ===
REQUIRED_COLUMNS = [
    "document_id", "text", "title", "country", "cluster",
    "x", "y", "status",  # for UMAP & diagnostics
]

# === Other Config ===
OPENAI_MODEL = "gpt-4o"
MAX_CONTEXT_TOKENS = 8000

USE_GPT = False        # If False, fallback to ClimateBERT or local model
CLIMATEBERT_NAME = "climatebert/distilroberta-base-climate-policy"

OUTPUT_PATH = "data/extracted_texts.pkl"

PLOT_DATA_BLOB = "data/plot_df.pkl"
COUNTRY_DATA_BLOB = "data/country_plot_df.pkl"

# === Configuration Status Display ===
def display_config_status():
    """Display current configuration status for debugging."""
    debug_print("=== Configuration Status ===", debug_type="general")
    debug_print(f"Streamlit Cloud: {IS_STREAMLIT_CLOUD}", debug_type="general")
    debug_print(f"OpenAI API Key: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}", debug_type="general")
    debug_print(f"Azure Account Name: {'✅ Configured' if AZURE_STORAGE_ACCOUNT_NAME else '❌ Missing'}", debug_type="general")
    debug_print(f"Azure Account Key: {'✅ Configured' if AZURE_STORAGE_ACCOUNT_KEY else '❌ Missing'}", debug_type="general")
    debug_print(f"Azure Connection String: {'✅ Configured' if AZURE_CONNECTION_STRING else '❌ Missing'}", debug_type="general")
    debug_print(f"Optimal Device: {OPTIMAL_DEVICE}", debug_type="general")
    debug_print("=== End Configuration Status ===", debug_type="general")

# Display configuration status if debug is enabled
if DEBUG_ENABLED:
    display_config_status()
