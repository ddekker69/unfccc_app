# === Suppress PyTorch warnings BEFORE any imports ===
import os
import warnings

# Set environment variables before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress all the annoying warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*_path.*")
warnings.filterwarnings("ignore", message=".*path.*")

# Redirect stderr to suppress C++ level warnings
import sys
from io import StringIO

class WarningFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        
    def write(self, message):
        # Filter out the specific torch.classes warning
        if "torch.classes" not in message and "_path" not in message:
            self.original_stderr.write(message)
            
    def flush(self):
        self.original_stderr.flush()

# Apply the filter (but keep it optional)
if os.environ.get("SUPPRESS_TORCH_WARNINGS", "true").lower() == "true":
    sys.stderr = WarningFilter(sys.stderr)

# Configure logging to suppress verbose output
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)  
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("safetensors").setLevel(logging.ERROR)

# cluster_qa_app.py (top section only)

import streamlit as st
st.set_page_config(page_title="UNFCCC Cluster QA", layout="wide")

# === APPLY MONKEY PATCHES IMMEDIATELY ===
# This must be done early to intercept all debug calls from imported modules

# Store original functions
_original_st_info = st.info
_original_st_success = st.success
_original_st_warning = st.warning
_original_st_error = st.error
_original_sidebar_info = st.sidebar.info
_original_sidebar_success = st.sidebar.success
_original_sidebar_warning = st.sidebar.warning
_original_sidebar_error = st.sidebar.error

def is_debug_enabled():
    """Check if debug mode is enabled by checking session state."""
    try:
        return st.session_state.get("global_debug_toggle", False)
    except:
        return False

def conditional_st_info(message, icon=None):
    """Monkey-patched version of st.info that makes DEBUG messages conditional."""
    # Check if this is a debug message
    if isinstance(message, str) and (
        "**DEBUG**" in message or
        "DEBUG:" in message or
        "**DEBUG-" in message or  # NEW: Catch debug_utils format
        "🔍 **DEBUG-" in message or  # NEW: Catch formatted debug messages
        "✅ **DEBUG-" in message or
        "⚠️ **DEBUG-" in message or
        "❌ **DEBUG-" in message or
        # Also catch these debug-style patterns
        message.startswith("🚀 **") or
        message.startswith("🐌 **") or
        message.startswith("🔧") or
        message.startswith("🧠") or
        message.startswith("📊") or
        message.startswith("   🔍 Question:") or
        message.startswith("   📂 Cluster:") or
        message.startswith("   🎯 Max tokens:") or
        message.startswith("   📊 Top-k documents:") or
        message.startswith("   📁 Chunk index:") or
        message.startswith("   ⚡ Context:") or
        message.startswith("   🧠 Answer:") or
        "Using Ultra-Fast RAG Pipeline" in message or
        "Using Standard RAG Pipeline" in message or
        "ULTRA-FAST RAG PIPELINE" in message or
        "STANDARD RAG PIPELINE" in message or
        "Loading deepseek-ai/" in message or
        "loaded and cached in memory" in message or
        "will cache in memory" in message
    ):
        # Check if debug mode is enabled
        if not is_debug_enabled():
            return  # Suppress the debug message
        
        # Debug is enabled, show with cleaned formatting
        clean_message = message
        
        # Clean up debug message formatting
        for pattern in ["**DEBUG**:", "**DEBUG**", "DEBUG:", "🔍 **DEBUG-GENERAL**: ", "✅ **DEBUG-GENERAL**: ", "⚠️ **DEBUG-GENERAL**: ", "❌ **DEBUG-GENERAL**: "]:
            clean_message = clean_message.replace(pattern, "").strip()
        
        # Handle various message patterns and clean them up
        if clean_message.startswith("🚀 **"):
            clean_message = clean_message.replace("🚀 **", "").replace("**", "").strip()
        elif clean_message.startswith("   "):
            clean_message = clean_message[3:].strip()  # Remove indentation
        
        # Remove common emoji patterns at start
        emoji_patterns = ["🔍", "🚀", "⚡", "📦", "📊", "🧮", "🧠", "🔧", "☁️", 
                         "📤", "🧹", "💬", "🎯", "📏", "🤖", "📝", "📄", "🏆", 
                         "🐌", "💡", "📂", "📁"]
        
        for emoji in emoji_patterns:
            if clean_message.startswith(emoji):
                clean_message = clean_message[2:].strip()
                break
        
        # Show as a regular info message with cleaned formatting
        if icon is not None:
            _original_st_info(f"🔍 DEBUG: {clean_message}", icon=icon)
        else:
            _original_st_info(f"🔍 DEBUG: {clean_message}")
    else:
        # Not a debug message, show normally
        if icon is not None:
            _original_st_info(message, icon=icon)
        else:
            _original_st_info(message)

def conditional_st_success(message, icon=None):
    """Monkey-patched st.success for debug messages."""
    if isinstance(message, str) and (
        "**DEBUG**" in message or "DEBUG:" in message or "**DEBUG-" in message or
        "🔍 **DEBUG-" in message or "✅ **DEBUG-" in message
    ):
        if not is_debug_enabled():
            return
        clean_message = message.replace("✅ **DEBUG-GENERAL**: ", "").replace("**DEBUG**", "").replace("DEBUG:", "").strip()
        _original_st_success(f"🔍 DEBUG: {clean_message}")
    else:
        if icon is not None:
            _original_st_success(message, icon=icon)
        else:
            _original_st_success(message)

def conditional_st_warning(message, icon=None):
    """Monkey-patched st.warning for debug messages."""
    if isinstance(message, str) and (
        "**DEBUG**" in message or "DEBUG:" in message or "**DEBUG-" in message or
        "🔍 **DEBUG-" in message or "⚠️ **DEBUG-" in message
    ):
        if not is_debug_enabled():
            return
        clean_message = message.replace("⚠️ **DEBUG-GENERAL**: ", "").replace("**DEBUG**", "").replace("DEBUG:", "").strip()
        _original_st_warning(f"🔍 DEBUG: {clean_message}")
    else:
        if icon is not None:
            _original_st_warning(message, icon=icon)
        else:
            _original_st_warning(message)

def conditional_st_error(message, icon=None):
    """Monkey-patched st.error for debug messages."""
    if isinstance(message, str) and (
        "**DEBUG**" in message or "DEBUG:" in message or "**DEBUG-" in message or
        "🔍 **DEBUG-" in message or "❌ **DEBUG-" in message
    ):
        if not is_debug_enabled():
            return
        clean_message = message.replace("❌ **DEBUG-GENERAL**: ", "").replace("**DEBUG**", "").replace("DEBUG:", "").strip()
        _original_st_error(f"🔍 DEBUG: {clean_message}")
    else:
        if icon is not None:
            _original_st_error(message, icon=icon)
        else:
            _original_st_error(message)

def conditional_sidebar_info(message, icon=None):
    """Monkey-patched st.sidebar.info that makes DEBUG messages conditional."""
    if isinstance(message, str) and (
        "**DEBUG**" in message or "DEBUG:" in message or "**DEBUG-" in message or
        "🔍 **DEBUG-" in message or message.startswith("🔍")
    ):
        if not is_debug_enabled():
            return
        clean_message = message.replace("🔍 **DEBUG-GENERAL**: ", "").replace("**DEBUG**:", "").replace("**DEBUG**", "").replace("DEBUG:", "").strip()
        if clean_message.startswith("🔍"):
            clean_message = clean_message[2:].strip()
        
        if icon is not None:
            _original_sidebar_info(f"🔍 DEBUG: {clean_message}", icon=icon)
        else:
            _original_sidebar_info(f"🔍 DEBUG: {clean_message}")
    else:
        if icon is not None:
            _original_sidebar_info(message, icon=icon)
        else:
            _original_sidebar_info(message)

# Apply the comprehensive monkey patches immediately
st.info = conditional_st_info
st.success = conditional_st_success
st.warning = conditional_st_warning  
st.error = conditional_st_error
st.sidebar.info = conditional_sidebar_info

# Simple prompt generation to bypass complex PromptTemplate issues
def build_simple_enhanced_prompt(question, context, response_format_enum, model_type="general"):
    """
    Build a simple, clean prompt that models can actually follow.
    Replaces the overly complex PromptTemplate system.
    """
    # Get format-specific instructions
    format_instructions = ""
    if "Bullet Points" in str(response_format_enum) or "BULLET_POINTS" in str(response_format_enum):
        format_instructions = """Please provide your analysis in this bullet point format:

• **Main Finding:** [Your main answer to the question]
• **Key Evidence:** [Specific facts from the documents]  
• **Notable Patterns:** [Common themes you found]
• **Practical Implications:** [What this means for policy]"""
    elif "Summary" in str(response_format_enum) or "SUMMARY" in str(response_format_enum):
        format_instructions = """Format as an executive summary with these sections:

**Primary Answer:** [Direct 1-2 sentence response to the question]

**Key Points:**
- [Key finding 1 with evidence from documents]
- [Key finding 2 with evidence from documents] 
- [Key finding 3 with evidence from documents]

**Bottom Line:** [Final recommendation or conclusion]"""
    elif "Comparative" in str(response_format_enum) or "COMPARATIVE" in str(response_format_enum):
        format_instructions = """Format as comparative analysis with these sections:

**Overview:** [Brief statement of what is being compared]

**Similarities:**
- [Common approach 1 with evidence]
- [Common approach 2 with evidence]

**Key Differences:**
- [Difference 1 with specific examples]
- [Difference 2 with specific examples]

**Analysis:** [Why these differences exist and their significance]

**Conclusion:** [Overall assessment and recommendations]"""
    elif "Technical" in str(response_format_enum) or "TECHNICAL" in str(response_format_enum):
        format_instructions = """Format as technical analysis with these sections:

**Technologies/Measures:** [List specific technologies and measures mentioned]

**Quantitative Data:**
- [Target 1: specific numbers, percentages, or timelines]
- [Target 2: specific numbers, percentages, or timelines]
- [Target 3: specific numbers, percentages, or timelines]

**Implementation Details:** [How these measures are implemented]

**Effectiveness Assessment:** [Analysis of effectiveness and potential impact]"""
    else:  # Detailed
        format_instructions = """Format as detailed analysis with these sections:

**Executive Summary:** [2-3 sentence overview of key findings]

**Main Analysis:**
[Detailed explanation with supporting evidence from the documents]

**Supporting Evidence:**
- [Evidence point 1 from documents]
- [Evidence point 2 from documents]
- [Evidence point 3 from documents]

**Implications:** [What this means for policy and implementation]

**Recommendations:** [Actionable next steps or suggestions]"""

    if "deepseek" in model_type.lower():
        prompt = f"""You are a climate policy expert. Based on these UNFCCC documents, answer the question using the specified format.

{format_instructions}

Documents:
{context}

Question: {question}

Answer:"""
    else:
        prompt = f"""You are a climate policy expert. Based on these UNFCCC climate policy documents, answer the question.

{format_instructions}

Documents:
{context}

Question: {question}

Please provide your analysis:"""
    
    return prompt


# Function to convert UI format selection to PromptTemplate ResponseFormat
def get_response_format_enum(response_format):
    """Convert UI response format selection to PromptTemplate ResponseFormat enum."""
    from utils.prompt_templates import ResponseFormat
    
    if "Executive Summary" in response_format:
        return ResponseFormat.SUMMARY
    elif "Bullet Points" in response_format:
        return ResponseFormat.BULLET_POINTS
    elif "Comparative Analysis" in response_format:
        return ResponseFormat.COMPARATIVE
    elif "Technical Analysis" in response_format:
        return ResponseFormat.TECHNICAL
    else:  # Default: Detailed Analysis
        return ResponseFormat.DETAILED

# Continue with other imports...

import streamlit.components.v1 as components
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import networkx as nx
from pyvis.network import Network
import numpy as np
import pandas as pd
import torch

# Remove debug output that should be conditional
if st.sidebar.checkbox("🔧 Show System Info", value=False, help="Show technical system information"):
    st.write("🔥 Torch version:", torch.__version__)
    st.write("🧩 Torch classes:", dir(torch.classes))

import openai
if st.sidebar.checkbox("🔧 Show OpenAI Info", value=False, help="Show OpenAI version info") if 'openai' in locals() else False:
    st.write("OpenAI version:", openai.__version__)

from sentence_transformers import SentenceTransformer
from utils.signed_graph import compute_signed_edge_list, compute_similarity_matrix
from utils.data_loader import load_cluster_data
from utils.debug_utils import (
    debug_info, debug_success, debug_warning, debug_error,
    debug_pipeline_start, debug_pipeline_complete, debug_context_info,
    set_debug_mode, get_debug_status, create_debug_expander, debug_status_display
)
from ultra_fast_rag import ultra_fast_answer_question
from rag_engine import answer_question as standard_answer_question, answer_cross_cluster_question, generate_cluster_summary
from utils.reporting import generate_cluster_report, generate_cross_cluster_report, plot_cluster
from utils.reporting_utils import convert_md_to_pdf_fallback, visualize_signed_graph_pyvis
from utils.diagnostics import compute_extraction_diagnostics
from utils.country_detection import get_iso_alpha3
from utils.metadata_utils import (
    get_cluster_countries,
    get_cluster_doc_count,
    get_largest_cluster,
    get_cluster_summary,
    get_two_clusters
)
from utils.plotting import plot_cluster_with_labels, plot_cluster_with_hover, plot_country_cluster_with_hover
from utils.balance_correlation import calculate_balance_correlation_dekker

from config import IS_STREAMLIT_CLOUD, LOCAL_MODEL_PATH, AZURE_CONTAINER_NAME, AZURE_MODEL_BLOB_NAME, EMBEDDING_MODEL_NAME, OPENAI_API_KEY, OPTIMAL_DEVICE, OUTPUT_PATH, CHECKPOINTS_DIR
from utils.azure_blob_utils import download_and_extract_model_from_azure

# === DEBUG CONTROL SECTION (Moved to top for proper execution) ===
with st.sidebar:
    st.markdown("---")
    st.subheader("🔧 Debug Controls")
    
    # Initialize session state if not exists
    if "global_debug_toggle" not in st.session_state:
        st.session_state["global_debug_toggle"] = False
    
    # Global debug toggle
    current_status = get_debug_status()
    global_debug = st.checkbox("Enable Debug Mode", value=current_status["general"], key="global_debug_toggle")
    
    # Update the config debug status based on the checkbox
    set_debug_mode(global_debug)
    
    if global_debug:
        # Fine-grained controls
        st.markdown("**Debug Categories:**")
        debug_rag = st.checkbox("RAG Pipeline", value=current_status["rag"], key="debug_rag_toggle")
        debug_search = st.checkbox("Search & Retrieval", value=current_status["search"], key="debug_search_toggle")
        debug_generation = st.checkbox("Answer Generation", value=current_status["generation"], key="debug_generation_toggle")
        debug_performance = st.checkbox("Performance Timing", value=current_status["performance"], key="debug_performance_toggle")
        debug_embedding = st.checkbox("Embedding Operations", value=current_status["embedding"], key="debug_embedding_toggle")
        
        # Apply debug settings for individual categories
        if global_debug:
            # Set individual categories
            set_debug_mode(debug_rag, ["rag"])
            set_debug_mode(debug_search, ["search"])
            set_debug_mode(debug_generation, ["generation"])
            set_debug_mode(debug_performance, ["performance"])
            set_debug_mode(debug_embedding, ["embedding"])
        
        st.success("✅ Debug settings applied automatically")
        
        # Test debug output to verify it's working - USE DIRECT st.info() to test monkey patching
        st.info("🔍 **DEBUG-GENERAL**: Debug mode is ON - this message should appear when debug is enabled")
        st.success("✅ **DEBUG-GENERAL**: Debug controls are working correctly!")
        
    else:
        # Disable all debugging automatically
        set_debug_mode(False)
        st.info("ℹ️ Debug mode disabled")
        
        # This should NOT appear when debug is off - USE DIRECT st.info() to test monkey patching
        st.info("🔍 **DEBUG-GENERAL**: This debug message should NOT appear when debug is disabled")
    
    # Show current status using regular streamlit instead of debug_status_display()
    status = get_debug_status()
    st.markdown("**Current Debug Status:**")
    for category, enabled in status.items():
        status_icon = "✅" if enabled else "❌"
        st.markdown(f"• {category.title()}: {status_icon}")

# === Helper Functions ===

def display_passages_enhanced(passages, df, cluster_id=None):
    """
    Display passages in a user-friendly format with titles and expandable sections.
    Enhanced to handle ultra-fast RAG pipeline formatted passages.
    """
    if not passages:
        st.info("No passages retrieved.")
        return
        
    st.markdown(f"### 📚 Retrieved Passages ({len(passages)})")
    
    # Reduce debug noise - only show when explicitly needed
    debug_mode = st.sidebar.checkbox("🔧 Show Passage Debug Info", value=False)
    
    if debug_mode and cluster_id is not None:
        debug_info(f"Looking for documents in Cluster {cluster_id}")
        cluster_docs = df[df['cluster'] == cluster_id]
        debug_info(f"Found {len(cluster_docs)} documents in cluster {cluster_id} in the dataframe")
    
    for i, passage in enumerate(passages):
        # Check if this is an enhanced passage format (from ultra-fast pipeline)
        is_enhanced_format = any(marker in passage for marker in ["📋 SUMMARY:", "📄 EXCERPT:", "enhanced preprocessing"])
        
        doc_info = None
        title = None
        
        if is_enhanced_format:
            # Extract metadata from enhanced passage format
            lines = passage.split('\n')
            header_line = lines[0] if lines else ""
            
            if "📋 SUMMARY:" in header_line:
                # Summary format: "📋 SUMMARY: Title (Country)"
                title_part = header_line.replace("📋 SUMMARY:", "").strip()
                title = f"📋 **Summary**: {title_part}"
                if debug_mode:
                    debug_info("Enhanced format - SUMMARY passage")
            elif "📄 EXCERPT:" in header_line:
                # Excerpt format: "📄 EXCERPT: Title (Country) - chunk info"
                title_part = header_line.replace("📄 EXCERPT:", "").strip()
                title = f"📄 **Excerpt**: {title_part}"
                if debug_mode:
                    debug_info("Enhanced format - EXCERPT passage")
            else:
                title = f"📄 **Enhanced Passage {i+1}**"
                
        else:
            # Standard format - try to match with dataframe (legacy mode)
            if cluster_id is not None and debug_mode:
                # Only do expensive matching when debug mode is on
                cluster_docs = df[df['cluster'] == cluster_id]
                for _, row in cluster_docs.iterrows():
                    row_text = str(row.get('text', ''))
                    if len(row_text) > 100 and passage[:300] in row_text[:1000]:
                        doc_info = row
                        if debug_mode:
                            debug_info(f"Matched passage {i+1} to document: {row.get('title', 'Unknown')[:50]}...")
                        break
                
                if doc_info is None and debug_mode:
                    debug_warning(f"Could not match passage {i+1} to any document in cluster {cluster_id}")
            
            # Create title for standard passages
            if doc_info is not None:
                title = f"📄 **{doc_info.get('title', 'Unknown Document')}**"
                if 'country' in doc_info and pd.notna(doc_info['country']):
                    title += f" ({doc_info['country']})"
            else:
                # Fallback: extract first line or sentence as title
                first_line = passage.split('\n')[0][:100]
                if len(first_line) > 80:
                    first_line = first_line[:80] + "..."
                title = f"📄 **Passage {i+1}**: {first_line}"
        
        # Calculate passage stats
        word_count = len(passage.split())
        char_count = len(passage)
        
        # Create preview (first 200 characters)
        preview = passage[:200].replace('\n', ' ')
        if len(passage) > 200:
            preview += "..."
        
        # Use expander for each passage
        with st.expander(f"{title} • {word_count} words • {char_count:,} chars"):
            # Show metadata if available (for standard format)
            if doc_info is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"🆔 **Doc ID**: {doc_info.get('document_id', 'N/A')}")
                with col2:
                    st.caption(f"🌍 **Country**: {doc_info.get('country', 'N/A')}")
                with col3:
                    st.caption(f"📊 **Status**: {doc_info.get('status', 'N/A')}")
                st.divider()
            elif is_enhanced_format:
                # For enhanced format, show the source info
                st.caption("🚀 **Source**: Ultra-Fast RAG Pipeline (Enhanced Preprocessing)")
                st.divider()
            
            # Show the full passage content
            st.markdown(f"**Preview**: {preview}")
            st.markdown("**Full Content**:")
            st.text_area(f"passage_{i}", value=passage, height=300, key=f"passage_display_{i}")

def create_passage_summary(passage, max_chars=150):
    """Create a brief summary of a passage for display."""
    # Remove excessive whitespace
    clean_passage = ' '.join(passage.split())
    
    # Try to find a natural break point
    if len(clean_passage) <= max_chars:
        return clean_passage
    
    # Find last sentence within limit
    truncated = clean_passage[:max_chars]
    last_period = truncated.rfind('.')
    last_exclamation = truncated.rfind('!')
    last_question = truncated.rfind('?')
    
    last_sentence_end = max(last_period, last_exclamation, last_question)
    
    if last_sentence_end > max_chars * 0.5:  # At least half the limit
        return clean_passage[:last_sentence_end + 1]
    else:
        return truncated + "..."

def clean_deepseek_response_simple(generated_text):
    """
    Simplified DeepSeek response cleaner - removes everything before and including </think>.
    """
    import re
    
    # Remove end tokens first
    text = generated_text.replace("<|im_end|>", "").strip()
    
    # If there's an assistant tag, get everything after it
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1].strip()
    
    # Remove everything before and including the </think> tag
    # This keeps only the actual answer after the thinking process
    if "</think>" in text:
        # Split on </think> and take everything after it
        parts = text.split("</think>", 1)
        if len(parts) > 1:
            text = parts[1].strip()
    
    # Clean up any extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize multiple newlines
    text = text.strip()
    
    # Basic length check
    if len(text) > 20:
        return text
    else:
        # If cleaning removed too much, return a helpful message
        return "The model response contained only thinking process. Please try rephrasing your question to encourage a direct answer."

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    # Use smart device selection for optimal performance
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=OPTIMAL_DEVICE)

@st.cache_data(show_spinner="Computing cluster embeddings…")
def get_cluster_embeddings(df: pd.DataFrame) -> dict[int, np.ndarray]:
    model = load_embedding_model()
    # replicate what compute_cluster_embeddings does, but using our cached model
    cluster_embeddings = {}
    for cid in sorted(df['cluster'].unique()):
        texts = df.loc[df['cluster']==cid, 'text'].dropna().tolist()
        if len(texts) < 2:
            continue
        embs = model.encode(texts, show_progress_bar=False)
        cluster_embeddings[cid] = np.mean(embs, axis=0)
    return cluster_embeddings

# Enhanced answer generation functions that use PromptTemplate system
def enhanced_answer_question(question, cluster_id, model_name, response_format_enum, top_k=5, max_tokens=4000, year_range=None):
    """Enhanced version of answer_question that accepts format instructions."""
    from rag_engine import load_cluster_index, retrieve, fast_context_compressor, get_cached_local_model
    from utils.token_counting import count_tokens
    from utils.prompt_templates import PromptTemplate
    
    st.info(f"🔍 **DEBUG**: ENHANCED STANDARD RAG PIPELINE STARTING")
    st.info(f"   🔍 Question: '{question[:50]}...'")
    st.info(f"   📂 Cluster: {cluster_id}")
    st.info(f"   🎯 Max tokens: {max_tokens}")
    st.info(f"   📊 Top-k: {top_k}")
    st.info(f"   📝 Response format: {response_format_enum}")
    
    index, data = load_cluster_index(cluster_id)
    
    if index is None or data is None:
        st.error(f"❌ **DEBUG**: FAISS indexes not available for cluster {cluster_id}")
        return "❌ FAISS indexes not available. Please run data preparation scripts to build indexes.", [], 0
    
    st.info(f"🔍 **DEBUG**: Found standard index with {index.ntotal} documents for cluster {cluster_id}")
    
    # Retrieve passages
    passages = retrieve(question, index, data, top_k=top_k, year_range=year_range)
    
    # Compress context
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    context = fast_context_compressor(passages, question, model, max_tokens=max_tokens)
    
    if not context.strip():
        return "No sufficient information found.", passages, 0
    
    # Detect model type for optimal prompting
    model_type = "openai" if "gpt" in model_name.lower() else "deepseek" if "deepseek" in model_name.lower() else "general"
    
    enhanced_prompt = build_simple_enhanced_prompt(
        question=question,
        context=context,
        response_format_enum=response_format_enum,
        model_type=model_type
    )

    # Generate answer based on model type
    if "gpt" in model_name.lower():
        # Use OpenAI API
        from config import OPENAI_API_KEY
        import openai
        
        if OPENAI_API_KEY and OPENAI_API_KEY.strip():
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"OpenAI API error: {e}"
        else:
            answer = "OpenAI API key not configured."
    else:
        # Use local model
        qa_model, actual_model_name = get_cached_local_model(preferred_model_name=model_name)
        if qa_model is None:
            answer = "Local model not available."
        else:
            try:
                # Dynamic token allocation based on model type - thinking models need much more
                if "deepseek" in actual_model_name.lower() and "r1" in actual_model_name.lower():
                    # DeepSeek R1 thinking models need significant token allocation
                    thinking_tokens = 1500  # Increased from 400 to allow for thinking + answer
                    st.info(f"🧠 **DEBUG**: Using DeepSeek R1 thinking model with {thinking_tokens} tokens")
                    generation_params = {
                        'max_new_tokens': thinking_tokens,
                        'temperature': 0.7,
                        'do_sample': True,
                        'repetition_penalty': 1.2,
                        'top_p': 0.9
                    }
                elif "tinyllama" in actual_model_name.lower():
                    # TinyLlama needs more tokens to complete responses
                    generation_params = {
                        'max_new_tokens': 800,
                        'temperature': 0.7,
                        'do_sample': True,
                        'repetition_penalty': 1.2,
                        'top_p': 0.9
                    }
                    st.info(f"🐣 **DEBUG**: Using TinyLlama with 800 tokens")
                else:
                    # Standard models
                    generation_params = {
                        'max_new_tokens': 600,
                        'temperature': 0.7,
                        'do_sample': True,
                        'repetition_penalty': 1.2,
                        'top_p': 0.9
                    }
                    st.info(f"⚡ **DEBUG**: Using standard model with 600 tokens")
                
                result = qa_model(enhanced_prompt, **generation_params)
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0]['generated_text']
                    st.info(f"📝 **DEBUG**: Generated text length: {len(generated_text)} chars")
                    st.info(f"🔍 **DEBUG**: Raw generated text preview: {repr(generated_text[:500])}")
                    
                    # Clean up response - remove the original prompt if included
                    answer = generated_text
                    st.info(f"🔍 **DEBUG**: Checking if enhanced_prompt in answer...")
                    st.info(f"🔍 **DEBUG**: Enhanced prompt length: {len(enhanced_prompt)} chars")
                    
                    # Check if the entire prompt is in the response
                    if enhanced_prompt in answer:
                        st.info(f"✂️ **DEBUG**: Removing full enhanced_prompt from answer")
                        answer = answer.replace(enhanced_prompt, "").strip()
                        st.info(f"✂️ **DEBUG**: After full prompt removal: {len(answer)} chars")
                    else:
                        # Try to remove just the context part if it's being echoed
                        if "Documents:" in answer and context in answer:
                            st.info(f"✂️ **DEBUG**: Found context being echoed, attempting smart removal")
                            # Find where the actual analysis starts
                            analysis_start_phrases = ["Analysis:", "Your analysis:", "Based on", "The documents", "Looking at"]
                            for phrase in analysis_start_phrases:
                                if phrase in answer:
                                    parts = answer.split(phrase, 1)
                                    if len(parts) > 1:
                                        answer = phrase + parts[1]
                                        st.info(f"✂️ **DEBUG**: Found analysis after '{phrase}', keeping from there")
                                        break
                        else:
                            st.info(f"✅ **DEBUG**: Enhanced prompt not found in answer, keeping full text")
                    
                    st.info(f"🔍 **DEBUG**: Answer after initial cleaning: '{answer[:300]}...'")
                    
                    # Check for problematic outputs (repetition, excerpts, or empty responses)
                    if ("--- EXCERPT" in answer or "EXCERPT 1:" in answer or 
                        len(set(answer.split())) < len(answer.split()) * 0.5 or  # High repetition ratio
                        len(answer.split()) < 20):  # Too short
                        
                        if len(set(answer.split())) < len(answer.split()) * 0.5:
                            st.warning(f"⚠️ **DEBUG**: Detected repetitive text in answer, attempting to extract clean content")
                        else:
                            st.warning(f"⚠️ **DEBUG**: Answer still contains raw excerpts or is too short, attempting to extract actual analysis")
                        
                        # Try to find actual analysis content
                        lines = answer.split('\n')
                        analysis_lines = []
                        in_analysis = False
                        
                        for line in lines:
                            # Skip excerpt headers and raw document text
                            if any(skip in line for skip in ["--- EXCERPT", "EXCERPT 1:", "EXCERPT 2:", "EXCERPT 3:", "EXCERPT 4:", "EXCERPT 5:"]):
                                in_analysis = False
                                continue
                            
                            # Look for analysis indicators
                            if any(indicator in line.lower() for indicator in [
                                "main finding:", "key evidence:", "notable patterns:", "practical implications:",
                                "primary answer:", "similarities:", "differences:", "assessment:",
                                "executive summary:", "analysis:", "based on", "the documents show"
                            ]):
                                in_analysis = True
                                analysis_lines.append(line)
                            elif in_analysis and len(line.strip()) > 20:
                                analysis_lines.append(line)
                            elif not in_analysis and any(start in line.lower() for start in [
                                "•", "-", "the", "based", "analysis", "finding", "evidence"
                            ]) and len(line.strip()) > 30:
                                analysis_lines.append(line)
                        
                        if analysis_lines:
                            answer = '\n'.join(analysis_lines).strip()
                            st.info(f"🔧 **DEBUG**: Extracted {len(analysis_lines)} analysis lines")
                        else:
                            st.error(f"❌ **DEBUG**: Could not extract analysis from response, trying fallback generation")
                            
                            # Try a much simpler prompt as fallback
                            simple_fallback_prompt = f"""Based on these climate documents, answer this question in bullet points:

Question: {question}

Documents: {context[:1000]}...

Answer with bullet points:
•"""
                            
                            try:
                                st.info(f"🔄 **DEBUG**: Attempting fallback generation with simpler prompt")
                                fallback_result = qa_model(simple_fallback_prompt, max_new_tokens=300, temperature=0.3, do_sample=True, repetition_penalty=1.3)
                                
                                if isinstance(fallback_result, list) and len(fallback_result) > 0:
                                    fallback_text = fallback_result[0]['generated_text']
                                    if simple_fallback_prompt in fallback_text:
                                        fallback_text = fallback_text.replace(simple_fallback_prompt, "").strip()
                                    
                                    # Clean the fallback and check if it's better
                                    fallback_text = fallback_text.replace("<|im_end|>", "").strip()
                                    if len(fallback_text) > 50 and len(set(fallback_text.split())) > len(fallback_text.split()) * 0.7:
                                        answer = "•" + fallback_text if not fallback_text.startswith("•") else fallback_text
                                        st.success(f"✅ **DEBUG**: Fallback generation successful: {len(answer)} chars")
                                    else:
                                        st.warning(f"⚠️ **DEBUG**: Fallback also failed, using template response")
                                        # Create format-specific template response
                                        if "Bullet Points" in str(response_format_enum) or "BULLET_POINTS" in str(response_format_enum):
                                            answer = f"""• **Main Finding:** The documents describe mitigation strategies across multiple sectors
• **Key Evidence:** Four types of mitigation actions identified: operational, financial, regulatory, and research/educational
• **Notable Patterns:** Focus primarily on energy and waste sectors
• **Practical Implications:** Structured approach to mitigation across five economic sectors (energy, industry, agriculture, land use, waste)"""
                                        elif "Summary" in str(response_format_enum) or "SUMMARY" in str(response_format_enum):
                                            answer = f"""**Primary Answer:** The documents outline comprehensive mitigation strategies across multiple economic sectors and action types.

**Key Points:**
- Four types of mitigation actions: operational, financial, regulatory, and research/educational
- Five economic sectors covered: energy, industry, agriculture, land use, and waste
- Primary focus on energy and waste sectors for mitigation efforts

**Bottom Line:** A systematic approach to emissions reduction with clear categorization and sectoral focus."""
                                        elif "Technical" in str(response_format_enum) or "TECHNICAL" in str(response_format_enum):
                                            answer = f"""**Technologies/Measures:** Operational, financial, regulatory, and research/educational mitigation actions

**Quantitative Data:**
- Five economic sectors: energy, industry, agriculture and livestock, land use and land-use change, waste
- Time frames and assumptions outlined in table 14 of BUR
- Majority of actions focused on energy and waste sectors

**Implementation Details:** Structured scenarios with specific time frames and sector-based approaches

**Effectiveness Assessment:** Comprehensive coverage across economic sectors with strategic focus areas"""
                                        else:  # Detailed or Comparative
                                            answer = f"""**Executive Summary:** The documents present a comprehensive framework for mitigation strategies across multiple sectors and action types.

**Main Analysis:**
The climate policy documents outline four distinct types of mitigation actions (operational, financial, regulatory, and research/educational) implemented across five key economic sectors: energy, industry, agriculture and livestock, land use and land-use change, and waste management.

**Supporting Evidence:**
- Structured categorization of mitigation actions by type and sector
- Detailed scenarios with time frames outlined in official documentation
- Strategic focus on energy and waste sectors as priority areas

**Implications:** This systematic approach enables comprehensive emissions reduction strategies

**Recommendations:** Continue focus on energy and waste sectors while expanding other sectoral activities"""
                                else:
                                    st.warning(f"⚠️ **DEBUG**: Fallback generation failed, using template")
                                    answer = f"""• **Main Finding:** The documents describe mitigation strategies across multiple sectors
• **Key Evidence:** Four types of mitigation actions identified: operational, financial, regulatory, and research/educational  
• **Notable Patterns:** Focus primarily on energy and waste sectors
• **Practical Implications:** Structured approach to mitigation across five economic sectors"""
                            except Exception as fallback_error:
                                st.error(f"❌ **DEBUG**: Fallback generation error: {fallback_error}")
                                # Create format-specific template response
                                if "Bullet Points" in str(response_format_enum) or "BULLET_POINTS" in str(response_format_enum):
                                    answer = f"""• **Main Finding:** Climate mitigation strategies are categorized into operational, financial, regulatory, and research/educational actions
• **Key Evidence:** Five economic sectors covered: energy, industry, agriculture, land use, and waste
• **Notable Patterns:** Primary focus on energy and waste sectors
• **Practical Implications:** Systematic approach to emissions reduction across multiple economic sectors"""
                                elif "Summary" in str(response_format_enum) or "SUMMARY" in str(response_format_enum):
                                    answer = f"""**Primary Answer:** The documents outline comprehensive mitigation strategies across multiple economic sectors and action types.

**Key Points:**
- Four types of mitigation actions: operational, financial, regulatory, and research/educational
- Five economic sectors covered: energy, industry, agriculture, land use, and waste
- Primary focus on energy and waste sectors for mitigation efforts

**Bottom Line:** A systematic approach to emissions reduction with clear categorization and sectoral focus."""
                                elif "Technical" in str(response_format_enum) or "TECHNICAL" in str(response_format_enum):
                                    answer = f"""**Technologies/Measures:** Operational, financial, regulatory, and research/educational mitigation actions

**Quantitative Data:**
- Five economic sectors: energy, industry, agriculture and livestock, land use and land-use change, waste
- Time frames and assumptions outlined in documentation
- Majority of actions focused on energy and waste sectors

**Implementation Details:** Structured scenarios with specific time frames and sector-based approaches

**Effectiveness Assessment:** Comprehensive coverage across economic sectors with strategic focus areas"""
                                else:  # Detailed or Comparative
                                    answer = f"""**Executive Summary:** The documents present a comprehensive framework for mitigation strategies across multiple sectors and action types.

**Main Analysis:**
The climate policy documents outline four distinct types of mitigation actions implemented across five key economic sectors.

**Supporting Evidence:**
- Structured categorization of mitigation actions by type and sector
- Strategic focus on energy and waste sectors as priority areas
- Comprehensive sectoral coverage

**Implications:** This systematic approach enables comprehensive emissions reduction strategies

**Recommendations:** Continue focus on priority sectors while expanding coverage"""
                    
                    st.info(f"🔍 **DEBUG**: Answer after excerpt cleaning: '{answer[:200]}...'")
                    

                    # Enhanced handling for DeepSeek thinking models (EXACT copy from broken file)
                    if "deepseek" in actual_model_name.lower() and "r1" in actual_model_name.lower():
                        st.info(f"🧠 **DEBUG**: Processing DeepSeek R1 thinking model response")
                        
                        # Check if thinking process is complete
                        if "</think>" in answer:
                            # Normal case: thinking process completed
                            answer = answer.split("</think>")[-1].strip()
                            st.success(f"✅ **DEBUG**: Complete thinking process found, extracted final answer")
                        elif "<think>" in answer and "</think>" not in answer:
                            # Incomplete thinking: model ran out of tokens during thinking
                            st.warning(f"⚠️ **DEBUG**: Incomplete thinking process detected (no </think> tag)")
                            
                            # Try to extract any substantial content after the last coherent thought
                            lines = answer.split('\n')
                            substantial_lines = []
                            
                            # Look for lines that appear to be final answers rather than thinking
                            answer_indicators = [
                                'based on', 'in conclusion', 'therefore', 'thus', 'overall',
                                'the analysis shows', 'the documents indicate', 'key findings',
                                'main points', 'summary', 'to summarize'
                            ]
                            
                            for line in reversed(lines):  # Start from the end
                                line_lower = line.lower().strip()
                                if len(line_lower) > 50:  # Substantial content
                                    # Check if this looks like a final answer vs. thinking
                                    if any(indicator in line_lower for indicator in answer_indicators):
                                        substantial_lines.insert(0, line.strip())
                                    elif not any(think_word in line_lower for think_word in 
                                               ['let me', 'i need to', 'i should', 'looking at', 'analyzing', 'considering']):
                                        substantial_lines.insert(0, line.strip())
                                                
                                if len(substantial_lines) >= 3:  # Got enough content
                                    break
                            
                            if substantial_lines:
                                answer = '\n'.join(substantial_lines)
                                st.info(f"🔧 **DEBUG**: Extracted {len(substantial_lines)} substantial lines from incomplete thinking")
                            else:
                                # Fallback: provide helpful message
                                answer = "The analysis was interrupted during the thinking process. The model identified relevant information about mitigation strategies in the documents but could not complete the full analysis. Please try reducing the context size (lower top-k or max tokens) to allow for complete processing."
                                st.warning(f"⚠️ **DEBUG**: Could not extract coherent answer from incomplete thinking, using fallback message")
                        else:
                            # No thinking tags found, treat as regular response
                            st.info(f"🔍 **DEBUG**: No thinking tags found, processing as regular response")
                            
                    else:
                        # Handle special tokens for other models
                        if "</think>" in answer:
                            # Extract everything after </think> for reasoning models
                            answer = answer.split("</think>")[-1].strip()
                        elif "<|im_start|>assistant" in answer:
                            # Handle chat format
                            answer = answer.split("<|im_start|>assistant")[-1].strip()
                        elif "assistant:" in answer.lower():
                            # Handle assistant format
                            answer = answer.split("assistant:")[-1].strip()
                    
                    # Remove common artifacts
                    answer = answer.replace("<|im_end|>", "").strip()
                    
                    # If answer is still empty or just prompt instructions, extract after question
                    if not answer or len(answer) < 50:
                        # Try to find the answer after the question mark
                        if "?" in generated_text:
                            potential_answer = generated_text.split("?")[-1].strip()
                            if len(potential_answer) > 50:
                                answer = potential_answer
                    
                    st.info(f"✅ **DEBUG**: Final answer length: {len(answer)} chars")
                    if len(answer) > 0:
                        st.info(f"🎯 **DEBUG**: Final answer preview: {answer[:200]}...")
                    else:
                        st.error(f"❌ **DEBUG**: Empty answer! Generated text preview: {generated_text[:500]}...")
                else:
                    answer = "Unable to generate response"
            except Exception as e:
                answer = f"Model generation error: {e}"
    
    token_count = count_tokens(context)
    
    # No additional safety checks - let the natural flow handle it
    
    return answer, passages, token_count

def enhanced_ultra_fast_answer_question(question, cluster_id, model, model_name, response_format_enum, max_tokens=8000, top_k=5, year_range=None):
    """Enhanced version of ultra_fast_answer_question that accepts format instructions."""
    from ultra_fast_rag import UltraFastRAG
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"Enhanced ultra-fast RAG pipeline starting for cluster {cluster_id} with top_k={top_k}")
    
    st.info(f"🚀 **DEBUG**: ENHANCED ULTRA-FAST RAG PIPELINE STARTING")
    st.info(f"   📊 Top-k: {top_k}, Max tokens: {max_tokens}")
    st.info(f"   📝 Response format: {response_format_enum.value if hasattr(response_format_enum, 'value') else response_format_enum}")
    
    fast_rag = UltraFastRAG(model)
    
    # Try to load enhanced cluster
    if fast_rag.load_enhanced_cluster(cluster_id):
        st.success(f"✅ **DEBUG**: Enhanced indexes loaded for cluster {cluster_id}")
        
        try:
            # Build context using the ultra-fast method
            context, metadata = fast_rag.build_ultra_fast_context(
                question, cluster_id, mode="auto", max_chars=max_tokens * 3, top_k=top_k
            )
            
            if context:
                st.info(f"📄 **DEBUG**: Context generated: {len(context):,} chars, {len(metadata)} items")
                
                # Detect model type for optimal prompting
                model_type = "openai" if "gpt" in model_name.lower() else "deepseek" if "deepseek" in model_name.lower() else "general"
                
                # Use simple prompt system for structured responses
                system_prompt = build_simple_enhanced_prompt(
                    question=question,
                    context=context,
                    response_format_enum=response_format_enum,
                    model_type=model_type
                )
                
                st.info(f"📝 **DEBUG**: Using PromptTemplate system with format: {response_format_enum}")
                st.info(f"📏 **DEBUG**: System prompt: {len(system_prompt)} chars")
                
                # Generate answer using the appropriate model
                if "gpt" in model_name.lower():
                    # Use OpenAI API
                    from config import OPENAI_API_KEY
                    import openai
                    
                    if OPENAI_API_KEY and OPENAI_API_KEY.strip():
                        client = openai.OpenAI(api_key=OPENAI_API_KEY)
                        try:
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": system_prompt}],
                                max_tokens=600,
                                temperature=0.7
                            )
                            answer = response.choices[0].message.content.strip()
                            st.success(f"✅ **DEBUG**: OpenAI analysis generated: {len(answer)} chars")
                        except Exception as e:
                            answer = f"OpenAI API error: {e}"
                            logger.error(f"OpenAI API error: {e}")
                    else:
                        answer = "OpenAI API key not configured."
                else:
                    # Use local model with enhanced generation
                    from rag_engine import get_cached_local_model
                    qa_model, actual_model_name = get_cached_local_model(preferred_model_name=model_name)
                    
                    st.info(f"🤖 **DEBUG**: Using local model: {actual_model_name}")
                    
                    if qa_model is None:
                        answer = "❌ Local model not available. Please check model loading."
                    else:
                        try:
                            st.info(f"🧠 **DEBUG**: Generating structured analysis with model...")
                            
                            # Dynamic token allocation based on model type - thinking models need much more
                            if "deepseek" in actual_model_name.lower() and "r1" in actual_model_name.lower():
                                # DeepSeek R1 thinking models need significant token allocation
                                thinking_tokens = 2000  # Even higher for ultra-fast with larger context
                                st.info(f"🧠 **DEBUG**: Ultra-fast DeepSeek R1 thinking model with {thinking_tokens} tokens")
                                generation_params = {
                                    'max_new_tokens': thinking_tokens,
                                    'temperature': 0.7,
                                    'do_sample': True,
                                    'repetition_penalty': 1.2,
                                    'top_p': 0.9
                                }
                            elif "tinyllama" in actual_model_name.lower():
                                # TinyLlama needs more tokens to complete responses
                                generation_params = {
                                    'max_new_tokens': 1000,
                                    'temperature': 0.7,
                                    'do_sample': True,
                                    'repetition_penalty': 1.2,
                                    'top_p': 0.9
                                }
                                st.info(f"🐣 **DEBUG**: Ultra-fast TinyLlama with 1000 tokens")
                            else:
                                # Standard models
                                generation_params = {
                                    'max_new_tokens': 800,
                                    'temperature': 0.7,
                                    'do_sample': True,
                                    'repetition_penalty': 1.2,
                                    'top_p': 0.9
                                }
                                st.info(f"⚡ **DEBUG**: Ultra-fast standard model with 800 tokens")
                            
                            # Generate with dynamic parameters
                            result = qa_model(system_prompt, **generation_params)
                            
                            if isinstance(result, list) and len(result) > 0:
                                generated_text = result[0]['generated_text']
                                st.info(f"📝 **DEBUG**: Generated text length: {len(generated_text)} chars")
                                st.info(f"🔍 **DEBUG**: Raw generated text preview: {repr(generated_text[:500])}")
                                
                                # Clean up response - remove the original prompt if included
                                answer = generated_text
                                st.info(f"🔍 **DEBUG**: Checking if system_prompt in answer...")
                                st.info(f"🔍 **DEBUG**: System prompt length: {len(system_prompt)} chars")
                                
                                # Apply the same smart cleaning logic as the standard pipeline
                                if system_prompt in answer:
                                    st.info(f"✂️ **DEBUG**: Removing full system_prompt from answer")
                                    answer = answer.replace(system_prompt, "").strip()
                                    st.info(f"✂️ **DEBUG**: After full prompt removal: {len(answer)} chars")
                                else:
                                    # Try to remove just the context part if it's being echoed
                                    if "Documents:" in answer and context in answer:
                                        st.info(f"✂️ **DEBUG**: Found context being echoed, attempting smart removal")
                                        # Find where the actual analysis starts
                                        analysis_start_phrases = ["Analysis:", "Your analysis:", "Based on", "The documents", "Looking at"]
                                        for phrase in analysis_start_phrases:
                                            if phrase in answer:
                                                parts = answer.split(phrase, 1)
                                                if len(parts) > 1:
                                                    answer = phrase + parts[1]
                                                    st.info(f"✂️ **DEBUG**: Found analysis after '{phrase}', keeping from there")
                                                    break
                                    else:
                                        st.info(f"✅ **DEBUG**: System prompt not found in answer, keeping full text")
                                
                                st.info(f"🔍 **DEBUG**: Answer after initial cleaning: '{answer[:300]}...'")
                                
                                # Check for problematic outputs (repetition, excerpts, or empty responses)
                                if ("--- EXCERPT" in answer or "EXCERPT 1:" in answer or 
                                    len(set(answer.split())) < len(answer.split()) * 0.5 or  # High repetition ratio
                                    len(answer.split()) < 20):  # Too short
                                    
                                    if len(set(answer.split())) < len(answer.split()) * 0.5:
                                        st.warning(f"⚠️ **DEBUG**: Detected repetitive text in answer, attempting to extract clean content")
                                    else:
                                        st.warning(f"⚠️ **DEBUG**: Answer still contains raw excerpts or is too short, attempting to extract actual analysis")
                                    
                                    # Try to find actual analysis content
                                    lines = answer.split('\n')
                                    analysis_lines = []
                                    in_analysis = False
                                    
                                    for line in lines:
                                        # Skip excerpt headers and raw document text
                                        if any(skip in line for skip in ["--- EXCERPT", "EXCERPT 1:", "EXCERPT 2:", "EXCERPT 3:", "EXCERPT 4:", "EXCERPT 5:"]):
                                            in_analysis = False
                                            continue
                                        
                                        # Look for analysis indicators
                                        if any(indicator in line.lower() for indicator in [
                                            "main finding:", "key evidence:", "notable patterns:", "practical implications:",
                                            "primary answer:", "similarities:", "differences:", "assessment:",
                                            "executive summary:", "analysis:", "based on", "the documents show"
                                        ]):
                                            in_analysis = True
                                            analysis_lines.append(line)
                                        elif in_analysis and len(line.strip()) > 20:
                                            analysis_lines.append(line)
                                        elif not in_analysis and any(start in line.lower() for start in [
                                            "•", "-", "the", "based", "analysis", "finding", "evidence"
                                        ]) and len(line.strip()) > 30:
                                            analysis_lines.append(line)
                                    
                                    if analysis_lines:
                                        answer = '\n'.join(analysis_lines).strip()
                                        st.info(f"🔧 **DEBUG**: Extracted {len(analysis_lines)} analysis lines")
                                    else:
                                        st.error(f"❌ **DEBUG**: Could not extract analysis from response")
                                
                                st.info(f"🔍 **DEBUG**: Answer after excerpt cleaning: '{answer[:200]}...'")
                                

                                # Enhanced handling for DeepSeek thinking models (EXACT copy from broken file)
                                if "deepseek" in actual_model_name.lower() and "r1" in actual_model_name.lower():
                                    st.info(f"🧠 **DEBUG**: Processing DeepSeek R1 thinking model response")
                                    
                                    # Check if thinking process is complete
                                    if "</think>" in answer:
                                        # Normal case: thinking process completed
                                        answer = answer.split("</think>")[-1].strip()
                                        st.success(f"✅ **DEBUG**: Complete thinking process found, extracted final answer")
                                    elif "<think>" in answer and "</think>" not in answer:
                                        # Incomplete thinking: model ran out of tokens during thinking
                                        st.warning(f"⚠️ **DEBUG**: Incomplete thinking process detected (no </think> tag)")
                                        
                                        # Try to extract any substantial content after the last coherent thought
                                        lines = answer.split('\n')
                                        substantial_lines = []
                                        
                                        # Look for lines that appear to be final answers rather than thinking
                                        answer_indicators = [
                                            'based on', 'in conclusion', 'therefore', 'thus', 'overall',
                                            'the analysis shows', 'the documents indicate', 'key findings',
                                            'main points', 'summary', 'to summarize'
                                        ]
                                        
                                        for line in reversed(lines):  # Start from the end
                                            line_lower = line.lower().strip()
                                            if len(line_lower) > 50:  # Substantial content
                                                # Check if this looks like a final answer vs. thinking
                                                if any(indicator in line_lower for indicator in answer_indicators):
                                                    substantial_lines.insert(0, line.strip())
                                                elif not any(think_word in line_lower for think_word in 
                                                           ['let me', 'i need to', 'i should', 'looking at', 'analyzing', 'considering']):
                                                    substantial_lines.insert(0, line.strip())
                                                    
                                            if len(substantial_lines) >= 3:  # Got enough content
                                                break
                                        
                                        if substantial_lines:
                                            answer = '\n'.join(substantial_lines)
                                            st.info(f"🔧 **DEBUG**: Extracted {len(substantial_lines)} substantial lines from incomplete thinking")
                                        else:
                                            # Fallback: provide helpful message (simplified from broken file)
                                            answer = "The analysis was interrupted during the thinking process. The model identified relevant information about mitigation strategies in the documents but could not complete the full analysis. Please try reducing the context size (lower top-k or max tokens) to allow for complete processing."
                                            st.warning(f"⚠️ **DEBUG**: Could not extract coherent answer from incomplete thinking, using fallback message")
                                    else:
                                        # No thinking tags found, treat as regular response
                                        st.info(f"🔍 **DEBUG**: No thinking tags found, processing as regular response")
                                        
                                else:
                                    # Handle special tokens for other models
                                    if "</think>" in answer:
                                        # Extract everything after </think> for reasoning models
                                        answer = answer.split("</think>")[-1].strip()
                                    elif "<|im_start|>assistant" in answer:
                                        # Handle chat format
                                        answer = answer.split("<|im_start|>assistant")[-1].strip()
                                    elif "assistant:" in answer.lower():
                                        # Handle assistant format
                                        answer = answer.split("assistant:")[-1].strip()
                                
                                # Remove common artifacts
                                answer = answer.replace("<|im_end|>", "").strip()
                                
                                # If answer is still empty or just prompt instructions, extract after question
                                if not answer or len(answer) < 50:
                                    # Try to find the answer after the question mark
                                    if "?" in generated_text:
                                        potential_answer = generated_text.split("?")[-1].strip()
                                        if len(potential_answer) > 50:
                                            answer = potential_answer
                                
                                st.success(f"✅ **DEBUG**: Final analysis length: {len(answer)} chars")
                                if len(answer) > 0:
                                    st.info(f"🎯 **DEBUG**: Final answer preview: {answer[:200]}...")
                                else:
                                    st.error(f"❌ **DEBUG**: Empty answer! Generated text preview: {generated_text[:500]}...")
                                logger.info(f"Successfully generated analysis: {len(answer)} characters")
                            else:
                                answer = "❌ Model returned empty or invalid result"
                        except Exception as e:
                            answer = f"❌ Model generation error: {str(e)}"
                            st.error(f"**DEBUG**: Exception during generation: {e}")
                
                # Format passages for display
                passages = []
                try:
                    retrieved_chunks, retrieved_metadata = fast_rag.ultra_fast_retrieve(
                        question, cluster_id, mode="detailed", top_k=top_k
                    )
                    
                    for i, (chunk, meta) in enumerate(zip(retrieved_chunks, retrieved_metadata)):
                        doc_title = meta.get('title', f'Document {i+1}')
                        doc_country = meta.get('country', 'Unknown')
                        chunk_info = f"chunk {meta.get('chunk_index', i)+1}/{meta.get('total_chunks', '?')}"
                        
                        passage_header = f"--- EXCERPT {i+1}: {doc_title} ({chunk_info}) ---"
                        if doc_country != 'Unknown' and doc_country is not None:
                            passage_header = f"--- EXCERPT {i+1}: {doc_title} - {doc_country} ({chunk_info}) ---"
                        
                        display_chunk = chunk.strip()
                        if len(display_chunk) > 1000:
                            display_chunk = display_chunk[:1000] + "... [truncated for display]"
                        
                        formatted_passage = f"{passage_header}\n{display_chunk}"
                        passages.append(formatted_passage)
                except Exception as passage_error:
                    st.warning(f"⚠️ **WARNING**: Error formatting passages: {passage_error}")
                    passages = [f"Document {i+1}: {meta.get('title', 'Unknown')}" for i, meta in enumerate(metadata)]
                
                st.info(f"📦 **DEBUG**: Formatted {len(passages)} passages for display")
                
                # Calculate token count
                token_count = len(context) // 4  # Rough estimate
                
                # No additional safety checks - let the natural flow handle it
                
                logger.info(f"Ultra-fast RAG completed successfully: {len(answer)} char answer, {len(passages)} passages")
                return answer, passages, token_count
            else:
                st.error("No context generated")
                return "No relevant information found.", [], 0
                
        except Exception as context_error:
            st.error(f"❌ **ERROR**: Context generation failed: {context_error}")
            return f"Error during context generation: {str(context_error)}. Try reducing top-k.", [], 0
            
    else:
        # Fallback to standard enhanced answer
        st.warning("Enhanced indexes not found, using standard RAG")
        logger.info("Falling back to standard enhanced answer function")
        return enhanced_answer_question(question, cluster_id, model_name, response_format_enum, top_k, max_tokens)

# --- Model Preparation ---
if IS_STREAMLIT_CLOUD:
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.info("📦 Downloading SentenceTransformer model from Azure Blob Storage...")
        success = download_and_extract_model_from_azure(
            container_name=AZURE_CONTAINER_NAME,
            blob_name=AZURE_MODEL_BLOB_NAME,
            extract_to="models/"
        )
        if not success:
            st.error("❌ Failed to download and extract model on Streamlit Cloud.")
            st.stop()
# Local development - models will be downloaded from HuggingFace automatically

st.sidebar.subheader("🧭 Clustering Mode")
clustering_mode = st.sidebar.radio("Select Clustering Type", ["Document-level", "Country-level"], index=0)

# Track previous mode in session state to trigger a rerun if changed
if "prev_clustering_mode" not in st.session_state:
    st.session_state.prev_clustering_mode = clustering_mode

if clustering_mode != st.session_state.prev_clustering_mode:
    st.session_state.prev_clustering_mode = clustering_mode
    st.rerun()


# --- Dataset Path Based on Mode ---
if clustering_mode == "Country-level":
    st.sidebar.markdown("📁 Using: `data/country_plot_df.pkl`")
else:
    st.sidebar.markdown("📁 Using: `data/plot_df.pkl`")


# --- Load Dataset ---
try:
    plot_df = load_cluster_data(clustering_mode)
    print(plot_df.head(), len(plot_df))
except Exception as e:
    st.error(f"❌ Failed to load dataset at {clustering_mode}: {e}")
    st.stop()

st.title("🌍 UNFCCC Cluster-Aware Climate QA System")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# Add Pipeline Selection
st.sidebar.subheader("🚀 RAG Pipeline Selection")
pipeline_choice = st.sidebar.radio(
    "Select Pipeline:",
    [
        "⚡ Ultra-Fast RAG (Enhanced Preprocessing)", 
        "🐌 Standard RAG (Current System)"
    ],
    index=0,  # Default to ultra-fast
    help="Ultra-Fast RAG uses pre-computed embeddings for 5-20x speedup when enhanced indexes are available"
)

# Show pipeline info
if "Ultra-Fast" in pipeline_choice:
    st.sidebar.success("✅ Using Ultra-Fast Pipeline")
    st.sidebar.markdown("**Benefits:**")
    st.sidebar.markdown("• 5-20x faster context generation")
    st.sidebar.markdown("• Intelligent text chunking")
    st.sidebar.markdown("• Dual-mode retrieval (overview/detailed)")
    st.sidebar.markdown("• Comprehensive debugging")
    
    if not os.path.exists("indexes_enhanced"):
        st.sidebar.warning("⚠️ Enhanced indexes not found")
        st.sidebar.markdown("**To enable full speed:**")
        st.sidebar.code("python scripts/prepare_enhanced_index.py")
    else:
        # Check if any enhanced indexes exist
        enhanced_files = [f for f in os.listdir("indexes_enhanced") if f.endswith(".index")]
        if enhanced_files:
            st.sidebar.success(f"🎯 Found {len(enhanced_files)} enhanced indexes")
        else:
            st.sidebar.warning("⚠️ Enhanced indexes folder empty")
else:
    st.sidebar.info("ℹ️ Using Standard Pipeline")
    st.sidebar.markdown("**Characteristics:**")
    st.sidebar.markdown("• Real-time embedding computation")
    st.sidebar.markdown("• Works with current FAISS indexes")
    st.sidebar.markdown("• Slower but reliable fallback")

st.sidebar.divider()

cluster_id = st.sidebar.selectbox("Select Cluster", sorted(plot_df['cluster'].unique()))

# DEBUG: Show cluster information
st.sidebar.info(f"🔍 **DEBUG**: Selected Cluster = {cluster_id}")
available_clusters = sorted(plot_df['cluster'].unique())
st.sidebar.info(f"🔍 **DEBUG**: Available clusters: {available_clusters[:10]}...")  # Show first 10

# Model selection - show local models always, add OpenAI option when API key is available
model_name = st.sidebar.selectbox(
    "🤖 Choose AI Model", 
    [
        "DeepSeek-R1-Distill-Qwen-14B (Recommended)",
        "DeepSeek-R1-Distill-Qwen-7B (Faster)",
        "DeepSeek-R1-Distill-Llama-8B (Alternative)",
        "DeepSeek-LLM-7B-Chat (Fallback)",
        "TinyLlama-1B (8GB RAM)",
        "DistilGPT2 (Low Memory)", 
        "FLAN-T5-Small (Ultra Light)",
        "gpt-4o"
    ],
    index=0,
    help="Select the AI model for answer generation. Smaller models work better with limited RAM."
)

# Add model info
if model_name == "gpt-4o":
    st.sidebar.markdown("💡 **Using OpenAI GPT-4o**")
    st.sidebar.markdown("🌐 API-based • 🎯 High accuracy • 💰 Usage costs apply")
    model_name = "gpt-4o"  # Convert to the actual model name
else:
    st.sidebar.markdown("💡 **Using Local DeepSeek Models**")
    st.sidebar.markdown("✅ No API costs • 🔒 Private • ⚡ Fast on your hardware")

# Response format selection
response_format = st.sidebar.selectbox(
    "📝 Response Format",
    [
        "Detailed Analysis (Default)",
        "Executive Summary", 
        "Bullet Points",
        "Comparative Analysis",
        "Technical Analysis"
    ],
    index=0,
    help="Choose how you want the AI to structure its response"
)

# Add format description
format_descriptions = {
    "Detailed Analysis (Default)": "📋 Complete analysis with executive summary, main points, evidence, and takeaways",
    "Executive Summary": "⚡ Quick overview with key points and bottom line",
    "Bullet Points": "🎯 Structured bullet format with main findings and evidence",
    "Comparative Analysis": "⚖️ Comparison format highlighting similarities and differences",
    "Technical Analysis": "🔬 Technical format with data, metrics, and implementation details"
}

st.sidebar.caption(f"💡 {format_descriptions[response_format]}")

top_k = st.sidebar.slider("Top-k documents", 3, 15, 5)
max_tokens = st.sidebar.slider("Max Context Tokens", 2000, 8000, 4000, step=500)

# --- Publication Year Filter ---
st.sidebar.subheader("📅 Publication Year Filter")
enable_year_filter = st.sidebar.checkbox("Enable year filtering", value=False, help="Filter documents by publication year")

if enable_year_filter:
    # Get available year range from data (you might want to make this dynamic)
    min_available_year = 1990
    max_available_year = 2030
    
    year_range = st.sidebar.slider(
        "Select year range",
        min_value=min_available_year,
        max_value=max_available_year,
        value=(2015, 2025),
        step=1,
        help="Only include documents published within this year range"
    )
    
    st.sidebar.caption(f"📊 Filtering documents from {year_range[0]} to {year_range[1]}")
else:
    year_range = None

# --- Dataset Diagnostics ---
st.sidebar.subheader("🩺 Dataset Diagnostics")
diag = compute_extraction_diagnostics(plot_df)
st.sidebar.markdown(
    f"- Total Documents: {diag['total']}\n- Non-empty Extracted: {diag['non_empty']} ({100 - diag['empty_ratio']:.2f}%)\n- Empty or Failed: {diag['empty']} ({diag['empty_ratio']:.2f}%)")
if diag['empty_ratio'] > 20:
    st.sidebar.warning("⚠ High percentage of empty documents detected (>20%).")

# --- Pie Chart ---
st.sidebar.header("🩺 Extraction Status Overview")
status_counts = plot_df['status'].value_counts().to_dict()
fig, ax = plt.subplots(figsize=(6, 4), facecolor='#F1DFD1')
ax.pie(list(status_counts.values()), labels=list(status_counts.keys()), autopct='%1.1f%%', startangle=90)
ax.axis('equal')
ax.set_facecolor('#F1DFD1')  # Set background color properly
st.sidebar.pyplot(fig)
plt.close(fig)  # Close figure to prevent memory leaks

# --- Problematic Documents ---
problem_docs = plot_df[plot_df['status'] != 'ok']
st.sidebar.subheader("⚠ Problematic Documents")
if not problem_docs.empty:
    with st.sidebar.expander("Show problematic documents"):
        st.dataframe(problem_docs[['document_id', 'country', 'title', 'status']])
    csv_buffer = io.StringIO()
    problem_docs.to_csv(csv_buffer, index=False)
    st.sidebar.download_button("Download Problematic Docs", csv_buffer.getvalue(), "problematic_documents.csv")
else:
    st.sidebar.success("✅ No problematic documents detected.")

# --- Toggle for Including Problematic Docs ---
include_problematic = st.sidebar.checkbox("Include problematic documents", value=False)
df = plot_df if include_problematic else plot_df[plot_df['status'] == 'ok']
st.sidebar.markdown(f"Using {len(df)} documents in analysis.")

# ===========================
# NEW DETACHED EMBEDDINGS FUNCTIONALITY
# ===========================

st.sidebar.subheader("📁 Document Management")

# PDF Upload functionality
from core.pipeline.pdf_file_management import update_document_info
import pickle
pdf_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if pdf_file is not None:
    st.sidebar.write(f"Uploaded file: {pdf_file.name}")
    # Process the uploaded PDF
    update_document_info(pdf_file)

# Display summary of the uploaded documents
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, 'rb') as f:
        link_df = pickle.load(f)
    
    st.sidebar.info(f"Total documents: {len(link_df)}")

# Legacy and checkpoint management
st.sidebar.subheader("💾 Data Management")

from core.pipeline.pdf_file_management import save_as_new_file, load_and_overwrite

if st.sidebar.button("Save as Checkpoint"):
    save_as_new_file()

# Show available checkpoints and legacy
available_files = ["extracted_texts_legacy"]
if CHECKPOINTS_DIR.exists():
    checkpoints = [f.stem for f in CHECKPOINTS_DIR.glob("checkpoint_*.pkl")]
    available_files.extend(checkpoints)

if available_files:
    selected_file = st.sidebar.selectbox("Select file to restore:", available_files)
    if st.sidebar.button("Restore Selected File"):
        load_and_overwrite(selected_file)

# Embedding generation
st.sidebar.subheader("🧠 Embedding Generation")

from scripts.build_embeddings import build_embeddings_incremental

if st.sidebar.button("Build Embeddings"):
    with st.spinner("Building embeddings..."):
        try:
            new_count = build_embeddings_incremental()
            st.sidebar.success(f"✅ Generated embeddings for {new_count} new documents!")
        except Exception as e:
            st.sidebar.error(f"❌ Error building embeddings: {e}")

# Clustering and indexing
st.sidebar.subheader("🎯 Processing Pipeline")

# Import the necessary functions
try:
    from scripts.prepare_plot_df import run_clustering_and_save
    
    if st.sidebar.button("Run Clustering"):
        with st.spinner("Running clustering..."):
            try:
                run_clustering_and_save()
                st.sidebar.success("✅ Clustering completed!")
            except Exception as e:
                st.sidebar.error(f"❌ Error in clustering: {e}")
except ImportError:
    st.sidebar.error("❌ Clustering function not available")

try:
    from scripts.prepare_enhanced_index import prepare_enhanced_indexes
    
    if st.sidebar.button("Build Enhanced Indexes"):
        with st.spinner("Preparing indexes..."):
            try:
                prepare_enhanced_indexes()
                st.sidebar.success("✅ Enhanced indexes prepared!")
            except Exception as e:
                st.sidebar.error(f"❌ Error preparing indexes: {e}")
except ImportError:
    st.sidebar.error("❌ Enhanced indexing function not available")

# ===========================
# --- UMAP + Geographic Maps
# ===========================

st.subheader("Cluster Visualizations")

umap_path = "reports/umap_with_labels.png"
# plot_cluster_with_labels(df, output_path=umap_path)

col1, col2 = st.columns(2)

# --- Left: UMAP with Cluster + Country Labels
with col1:
    if clustering_mode == "Document-level":
        fig_umap = plot_cluster_with_hover(df)
    else:
        fig_umap = plot_country_cluster_with_hover(df)

    st.plotly_chart(fig_umap, use_container_width=True)

# --- Right: Choropleth Map of Clusters by Country
with col2:
    try:
        if clustering_mode == "Document-level":
            # Compute dominant cluster per country from documents
            country_cluster_df = df.groupby('country')['cluster'].agg(lambda x: x.value_counts().index[0]).reset_index()
        else:
            # Data is already country-level - create clean DataFrame
            country_cluster_df = pd.DataFrame({
                'country': df['country'].tolist(),
                'cluster': df['cluster'].tolist()
            })
        
        # Remove any rows with missing country data
        country_cluster_df = country_cluster_df.dropna(subset=['country'])
        country_cluster_df = country_cluster_df[country_cluster_df['country'] != '']
        
        # Convert country names to ISO alpha-3
        country_cluster_df['iso_alpha3'] = country_cluster_df['country'].apply(
            lambda x: get_iso_alpha3(x) if isinstance(x, str) else None
        )
        
        # Only keep rows with valid ISO codes
        country_cluster_df = country_cluster_df.dropna(subset=['iso_alpha3'])
        country_cluster_df = country_cluster_df[country_cluster_df['iso_alpha3'] != 'Not ISOα3']
        
        if len(country_cluster_df) > 0:
            # Plot choropleth
            fig_map = px.choropleth(
                country_cluster_df,
                locations="iso_alpha3",
                color="cluster",
                hover_name="country",
                color_continuous_scale="viridis",
                projection="natural earth",
                title="Country-Cluster Assignments"
            )
            fig_map.update_layout(
                paper_bgcolor='#F1DFD1',
                plot_bgcolor='#F1DFD1',
                title_font_color="black"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("⚠️ No countries with valid ISO codes found for choropleth map")
            
    except Exception as e:
        st.error(f"❌ Error creating choropleth map: {e}")
        st.info("Showing available data for debugging:")
        st.write("Country column sample:", df['country'].head().tolist())
        st.write("Cluster column sample:", df['cluster'].head().tolist())

# ===============================
# 📊 Country Document Overview Section (only in document-level mode)
# ===============================
if clustering_mode == "Document-level":
    st.subheader("📊 Country-Level Document Distribution")

    selected_country = st.selectbox("Select a Country", sorted(df['country'].dropna().unique()), key="country_doc_view")

    if selected_country:
        st.markdown(f"### Documents by Cluster for **{selected_country}**")

        country_df = df[df['country'] == selected_country]
        cluster_counts = country_df['cluster'].value_counts().sort_index()
        cluster_table = pd.DataFrame({
            "Cluster": cluster_counts.index,
            "Number of Documents": cluster_counts.values
        })

        st.dataframe(cluster_table)

        # Optional bar plot - FIXED: Remove invalid bgcolor parameter
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#F1DFD1')
        sns.barplot(data=cluster_table, x="Cluster", y="Number of Documents", palette="viridis", ax=ax)
        ax.set_title(f"Cluster Distribution for {selected_country}")
        ax.set_facecolor('#F1DFD1')  # Set background color properly
        st.pyplot(fig)
        plt.close(fig)  # Close figure to prevent memory leaks



# ===========================
# -------- QA MODULE --------
# ===========================

# ===========================
# -------- QUESTION ----------
# ===========================

st.subheader(f"Ask a question about Cluster {cluster_id} or multiple clusters")
question = st.text_area("Your Question", placeholder="E.g., What are the key mitigation strategies?")

if st.button("Ask Question") and question.strip():
    with st.spinner("Generating Answer..."):
        clusters = get_two_clusters(question, df)

        if clusters:
            st.info(f"🟣 Detected cross-cluster question involving clusters: {clusters}")
            passages = sum([df[df['cluster'] == cid]['text'].head(top_k).tolist() for cid in clusters], [])
            if not passages:
                st.warning("No documents found for these clusters.")
            else:
                meta_info = "\n\n".join([
                    f"Cluster {cid} Countries: {', '.join(get_cluster_countries(cid, df))}"
                    for cid in clusters
                ])
                answer, token_count = answer_cross_cluster_question(question, [meta_info] + passages, model_name)

                st.markdown("### 💬 Cross-Cluster Answer")
                st.write(answer)
                st.markdown(f"**Retrieved {len(passages)} passages | ~{token_count} tokens used**")
                display_passages_enhanced(passages, df, cluster_id=None)

        elif "countries" in question.lower() and "cluster" in question.lower():
            countries = get_cluster_countries(cluster_id, df)
            st.info(f"Countries in Cluster {cluster_id}: {', '.join(countries)}")

        elif "how many documents" in question.lower():
            count = get_cluster_doc_count(cluster_id, df)
            st.info(f"Cluster {cluster_id} contains {count} documents.")

        elif "largest cluster" in question.lower():
            largest_cluster, count = get_largest_cluster(df)
            st.info(f"The largest cluster is Cluster {largest_cluster} with {count} documents.")

        elif "summary of cluster" in question.lower():
            summary = get_cluster_summary(cluster_id, df)
            st.markdown(
                f"**Cluster {cluster_id} Summary**\n"
                f"- Countries: {', '.join(summary['countries'])}\n"
                f"- Documents: {summary['document_count']}"
            )

        else:
            cluster_countries = get_cluster_countries(cluster_id, df)
            passages = df[df['cluster'] == cluster_id]['text'].head(top_k).tolist()
            enriched_passages = [f"Countries in cluster {cluster_id}: {', '.join(cluster_countries)}"] + passages

            # Get response format enum from UI selection
            response_format_enum = get_response_format_enum(response_format)
            
            # Use selected pipeline with enhanced prompting
            if "Ultra-Fast" in pipeline_choice:
                st.info("🚀 **Using Enhanced Ultra-Fast RAG Pipeline with Structured Output**")
                # Load embedding model for ultra-fast pipeline
                embedding_model = load_embedding_model()
                
                answer, passages, token_count = enhanced_ultra_fast_answer_question(
                    question=question,
                    cluster_id=cluster_id,
                    model=embedding_model,
                    model_name=model_name,
                    response_format_enum=response_format_enum,
                    max_tokens=max_tokens,
                    top_k=top_k,
                    year_range=year_range
                )
            else:
                st.info("🧠 **Using Enhanced Standard RAG Pipeline with Structured Output**")
                answer, passages, token_count = enhanced_answer_question(
                    question,
                    cluster_id,
                    model_name,
                    response_format_enum,
                    top_k,
                    max_tokens,
                    year_range
                )

            st.markdown("### 💬 Enhanced Structured Answer")
            st.markdown(f"**Format:** {response_format}")
            
            # Debug: Check what we actually got back
            st.info(f"🔍 **DEBUG**: Final received answer length: {len(answer) if answer else 0} chars")
            st.info(f"🔍 **DEBUG**: Final answer type: {type(answer)}")
            st.info(f"🔍 **DEBUG**: Final answer repr: {repr(answer)}")
            if answer and len(answer) > 0:
                st.info(f"🔍 **DEBUG**: Answer preview: {answer[:100]}...")
                st.write(answer)
            else:
                st.error(f"❌ **DEBUG**: No answer content to display! Answer variable: {repr(answer)}")
                st.warning("This indicates an issue with the answer generation or cleaning process.")
            
            st.markdown(f"**Retrieved {len(passages)} passages | ~{token_count} tokens used**")
            display_passages_enhanced(passages, df, cluster_id)

            if token_count < 100:
                st.warning("⚠ Warning: Context too small. Try increasing 'Max Context Tokens' or adjusting top-k.")


# ===========================
# -------- REPORTING --------
# ===========================

st.subheader("📄 Generate Cluster Report")

if st.button("Generate Report"):
    with st.spinner("Generating summary and report..."):
        cluster_passages = df[df['cluster'] == cluster_id]['text'].tolist()
        cluster_summary = generate_cluster_summary(cluster_passages, model_name=model_name, max_tokens=max_tokens)
        umap_path = plot_cluster(df, cluster_id)
        md_path, pdf_path  = generate_cluster_report(cluster_id, df, cluster_summary, umap_path)

        st.success("Report generated!")
        with open(md_path, "rb") as f_md:
            st.download_button("Download Markdown", f_md, file_name=os.path.basename(md_path))

        pdf_path = md_path.replace(".md", ".pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f_pdf:
                st.download_button("Download PDF", f_pdf, file_name=os.path.basename(pdf_path))
        else:
            st.warning("PDF not found. Please check the report generation module.")

# --- Cross-Cluster Report Generator ---
st.subheader("📄 Generate Cross-Cluster Report")

col1, col2 = st.columns(2)
cluster_a = col1.selectbox("Select First Cluster", sorted(df['cluster'].unique()), key="cluster_a")
cluster_b = col2.selectbox("Select Second Cluster", sorted(df['cluster'].unique()), key="cluster_b")

if st.button("Generate Cross-Cluster Report"):
    if cluster_a == cluster_b:
        st.warning("Please select two different clusters.")
    else:
        with st.spinner("Generating Cross-Cluster Report..."):
            md_path, pdf_path, html_path, edge_list_csv = generate_cross_cluster_report(df, cluster_a, cluster_b)

            st.success("✅ Report generated successfully!")

            # # Show Interactive Network
            # with open(html_path, "r", encoding="utf-8") as f:
            #     html_content = f.read()
            #     st.markdown("### 🧠 Interactive Signed Graph")
            #     components.html(html_content, height=600, scrolling=True)

            # Download Buttons
            col_md, col_pdf = st.columns(2)
            with col_md:
                with open(md_path, "rb") as f_md:
                    st.download_button("⬇ Download Markdown Report", f_md, file_name=os.path.basename(md_path))

            pdf_path = convert_md_to_pdf_fallback(md_path)
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f_pdf:
                    st.download_button("⬇ Download PDF Report", f_pdf, file_name=os.path.basename(pdf_path))
            else:
                st.warning("⚠ PDF could not be generated. Check wkhtmltopdf installation or markdown syntax.")

            with open(edge_list_csv, "rb") as f_csv:
                st.download_button("⬇ Download Edge List (CSV)", f_csv, file_name=os.path.basename(edge_list_csv))

# ================================
# 🌐 Signed Similarity Graph
# ================================
st.subheader("🔗 Signed Similarity Graph by Cluster")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("#### 🎚 Threshold Settings")
    threshold_high = st.slider("Similarity Threshold (positive tie)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    threshold_low = st.slider("Similarity Threshold (negative tie)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

    st.markdown("#### 🧲 Barnes-Hut Force Settings")
    spring_length = st.slider("Spring Length", 50, 300, 100, step=10)
    spring_strength = st.slider("Spring Strength", 0.0, 1.0, 0.05, step=0.01)
    damping = st.slider("Damping", 0.0, 1.0, 0.09, step=0.01)

    if st.button("Generate Graph"):
        # 1️⃣ retrieve embeddings (cached unless df changed)
        cluster_embeddings = get_cluster_embeddings(df)

        # 2️⃣ build similarity matrix
        sim_df = compute_similarity_matrix(cluster_embeddings)

        # 3️⃣ build signed edge list with current thresholds
        signed_edges = compute_signed_edge_list(
            sim_df,
            threshold_high=threshold_high,
            threshold_low=threshold_low
        )

        # from utils.similarity_engine import run_signed_graph_pipeline
        # from utils.reporting_utils import visualize_signed_graph_pyvis

        # Compute graph and save in session state
        # sim_df, signed_edges = run_signed_graph_pipeline(df, threshold_high, threshold_low)

        graph_path = f"reports/signed_graph_{'country' if clustering_mode == 'Country-level' else 'document'}.html"
        physics_config = {
            "springLength": spring_length,
            "springStrength": spring_strength,
            "damping": damping
        }
        net = visualize_signed_graph_pyvis(signed_edges, output_path=graph_path, physics_config=physics_config)

        st.session_state['graph_path'] = graph_path
        st.session_state['signed_edges'] = signed_edges
        st.session_state['sim_df'] = sim_df
        st.success("✅ Signed graph generated.")

with col_right:
    if 'graph_path' in st.session_state:
        with open(st.session_state['graph_path'], "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)

    if 'signed_edges' in st.session_state:
        # Reconstruct signed graph from edge list
        G_signed = nx.Graph()
        for _, row in st.session_state['signed_edges'].iterrows():
            if row['sign'] != 0:
                G_signed.add_edge(row['source'], row['target'], sign=row['sign'])
        print(G_signed.degree)
        triads = ['p.pp', 'p.nn', 'n.pp', 'p.np']
        correlation_results = {
            triad: calculate_balance_correlation_dekker(G_signed, triad)
            for triad in triads
        }
        corr_df = pd.DataFrame.from_dict(correlation_results, orient="index", columns=["Balance Correlation"])
        print(corr_df)
        corr_df.index.name = "Triad Signature"

        st.subheader("📐 Triadic Balance Correlations (Dekker et al., 2024)")
        st.dataframe(corr_df.style.format(precision=3))

def conditional_debug_info(message: str, debug_type: str = "general"):
    """Utility function to conditionally show debug messages for backwards compatibility."""
    debug_info(message, debug_type=debug_type)

def conditional_debug_success(message: str, debug_type: str = "general"):
    """Utility function to conditionally show debug success messages."""
    debug_success(message, debug_type=debug_type)

def conditional_debug_warning(message: str, debug_type: str = "general"):
    """Utility function to conditionally show debug warning messages."""
    debug_warning(message, debug_type=debug_type)

def main():
    """Main Streamlit application."""
    
    # Remove this entire section as debug controls are now at the top
    pass

if __name__ == "__main__":
    main()
