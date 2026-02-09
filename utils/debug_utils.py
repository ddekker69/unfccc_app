"""
Debug utilities for the BalanceMitigations project.

This module provides convenient debugging functions that can be toggled on/off
globally using environment variables or runtime controls.

Usage:
    # Environment variables
    export DEBUG_MODE=true          # Enable all debugging
    export DEBUG_RAG=true           # Enable only RAG debugging
    export DEBUG_PERFORMANCE=false  # Disable performance debugging
    
    # Runtime control
    from utils.debug_utils import set_debug_mode, get_debug_status
    
    set_debug_mode(True)  # Enable all debugging
    set_debug_mode(False, ["rag", "search"])  # Disable specific categories
    
    # Usage in code
    from utils.debug_utils import debug_info, debug_success, debug_warning, debug_error
    
    debug_info("Starting RAG pipeline", debug_type="rag")
    debug_success("Context ready", debug_type="rag")
    debug_warning("Large context detected", debug_type="performance")
    debug_error("Generation failed", debug_type="generation")
"""

import logging
import time
from functools import wraps
from typing import Any, Optional, Callable
import streamlit as st
from config import (
    DEBUG_ENABLED, DEBUG_RAG, DEBUG_EMBEDDING, DEBUG_GENERATION, 
    DEBUG_SEARCH, DEBUG_PERFORMANCE, debug_print, debug_streamlit,
    set_debug_mode as config_set_debug_mode, get_debug_status as config_get_debug_status
)

logger = logging.getLogger(__name__)

def debug_info(message: str, debug_type: str = "general") -> None:
    """
    Display an info-level debug message.
    
    Args:
        message: Debug message to display
        debug_type: Category of debug message
    """
    debug_streamlit(message, level="info", debug_type=debug_type)

def debug_success(message: str, debug_type: str = "general") -> None:
    """
    Display a success-level debug message.
    
    Args:
        message: Debug message to display
        debug_type: Category of debug message
    """
    debug_streamlit(message, level="success", debug_type=debug_type)

def debug_warning(message: str, debug_type: str = "general") -> None:
    """
    Display a warning-level debug message.
    
    Args:
        message: Debug message to display
        debug_type: Category of debug message
    """
    debug_streamlit(message, level="warning", debug_type=debug_type)

def debug_error(message: str, debug_type: str = "general") -> None:
    """
    Display an error-level debug message.
    
    Args:
        message: Debug message to display
        debug_type: Category of debug message
    """
    debug_streamlit(message, level="error", debug_type=debug_type)

def debug_performance(func_name: str, duration: float, debug_type: str = "performance") -> None:
    """
    Display performance timing information.
    
    Args:
        func_name: Name of the function being timed
        duration: Duration in seconds
        debug_type: Category of debug message
    """
    if duration > 10:
        debug_warning(f"{func_name} took {duration:.2f}s (slow)", debug_type=debug_type)
    elif duration > 1:
        debug_info(f"{func_name} completed in {duration:.2f}s", debug_type=debug_type)
    else:
        debug_success(f"{func_name} completed in {duration:.3f}s", debug_type=debug_type)

def debug_context_info(context: str, max_preview: int = 200, debug_type: str = "rag") -> None:
    """
    Display context information for debugging.
    
    Args:
        context: The context string
        max_preview: Maximum characters to show in preview
        debug_type: Category of debug message
    """
    excerpt_count = context.count("--- DOCUMENT") if "--- DOCUMENT" in context else context.count("DOCUMENT")
    debug_success(f"Context ready: {excerpt_count} excerpts, {len(context)} chars", debug_type=debug_type)
    
    if context and len(context) > max_preview:
        debug_info(f"Context preview (first {max_preview} chars):", debug_type=debug_type)
        # Use st.code for better formatting if available
        try:
            st.code(context[:max_preview] + "...")
        except:
            debug_print(f"Preview: {context[:max_preview]}...", debug_type=debug_type)

def debug_model_info(model_name: str, operation: str, debug_type: str = "generation") -> None:
    """
    Display model operation information.
    
    Args:
        model_name: Name of the model
        operation: Operation being performed
        debug_type: Category of debug message
    """
    if model_name:
        if "deepseek" in model_name.lower() or "r1" in model_name.lower():
            debug_info(f"Using DeepSeek R1 model for {operation}: {model_name}", debug_type=debug_type)
        elif "openai" in model_name.lower() or "gpt" in model_name.lower():
            debug_info(f"Using OpenAI model for {operation}: {model_name}", debug_type=debug_type)
        else:
            debug_info(f"Using model for {operation}: {model_name}", debug_type=debug_type)
    else:
        debug_info(f"Auto-detecting model for {operation}", debug_type=debug_type)

def debug_retrieval_results(passages: list, retrieval_time: float, debug_type: str = "search") -> None:
    """
    Display retrieval results information.
    
    Args:
        passages: List of retrieved passages
        retrieval_time: Time taken for retrieval
        debug_type: Category of debug message
    """
    debug_success(f"Retrieved {len(passages)} passages in {retrieval_time:.2f}s", debug_type=debug_type)
    
    if passages and DEBUG_SEARCH:
        debug_info("Top retrieval results:", debug_type=debug_type)
        for i, passage in enumerate(passages[:3]):  # Show top 3
            preview = str(passage)[:100] if isinstance(passage, str) else str(passage)[:100]
            debug_info(f"  {i+1}. {preview}...", debug_type=debug_type)

def debug_cluster_info(cluster_id: int, total_docs: int, debug_type: str = "rag") -> None:
    """
    Display cluster information.
    
    Args:
        cluster_id: ID of the cluster
        total_docs: Total number of documents in cluster
        debug_type: Category of debug message
    """
    debug_info(f"Processing cluster {cluster_id} with {total_docs} documents", debug_type=debug_type)

def performance_timer(debug_type: str = "performance", func_name: Optional[str] = None):
    """
    Decorator to automatically time function execution and display debug info.
    
    Args:
        debug_type: Category of debug message
        func_name: Custom function name (uses actual function name if None)
        
    Example:
        @performance_timer(debug_type="rag")
        def generate_answer(question, context):
            # function implementation
            return answer
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            debug_info(f"Starting {name}...", debug_type=debug_type)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                debug_performance(name, duration, debug_type=debug_type)
                return result
            except Exception as e:
                duration = time.time() - start_time
                debug_error(f"{name} failed after {duration:.2f}s: {e}", debug_type=debug_type)
                raise
        return wrapper
    return decorator

def conditional_debug(condition: bool, message: str, level: str = "info", debug_type: str = "general") -> None:
    """
    Display debug message only if condition is True.
    
    Args:
        condition: Whether to display the message
        message: Debug message
        level: Message level ("info", "success", "warning", "error")
        debug_type: Category of debug message
    """
    if condition:
        if level == "info":
            debug_info(message, debug_type)
        elif level == "success":
            debug_success(message, debug_type)
        elif level == "warning":
            debug_warning(message, debug_type)
        elif level == "error":
            debug_error(message, debug_type)

def debug_pipeline_start(pipeline_name: str, question: str, cluster_id: Optional[int] = None, 
                        max_tokens: Optional[int] = None, debug_type: str = "rag") -> None:
    """
    Standard debug message for pipeline start.
    
    Args:
        pipeline_name: Name of the pipeline
        question: User question (truncated for display)
        cluster_id: Cluster ID if applicable
        max_tokens: Token limit if applicable
        debug_type: Category of debug message
    """
    debug_info(f"{pipeline_name} PIPELINE STARTING", debug_type=debug_type)
    debug_info(f"Question: '{question[:50]}...'", debug_type=debug_type)
    if cluster_id is not None:
        debug_info(f"Cluster: {cluster_id}", debug_type=debug_type)
    if max_tokens is not None:
        debug_info(f"Max tokens: {max_tokens}", debug_type=debug_type)

def debug_pipeline_complete(pipeline_name: str, total_time: float, 
                           components: Optional[dict] = None, debug_type: str = "rag") -> None:
    """
    Standard debug message for pipeline completion.
    
    Args:
        pipeline_name: Name of the pipeline
        total_time: Total execution time
        components: Dict of component times (e.g., {"retrieval": 1.2, "generation": 3.4})
        debug_type: Category of debug message
    """
    debug_success(f"{pipeline_name} COMPLETE in {total_time:.2f}s", debug_type=debug_type)
    
    if components:
        for component, duration in components.items():
            debug_info(f"  {component.capitalize()}: {duration:.2f}s", debug_type=debug_type)

# Expose config functions for convenience
set_debug_mode = config_set_debug_mode
get_debug_status = config_get_debug_status

def create_debug_expander(title: str = "🔍 Debug Information", expanded: bool = False):
    """
    Create a Streamlit expander for debug information.
    Only creates the expander if debugging is enabled.
    
    Args:
        title: Title for the expander
        expanded: Whether to expand by default
        
    Returns:
        Streamlit expander object or None if debugging disabled
    """
    if DEBUG_ENABLED:
        try:
            return st.expander(title, expanded=expanded)
        except:
            return None
    return None

def debug_status_display() -> None:
    """Display current debug status in Streamlit sidebar."""
    try:
        if st.sidebar.checkbox("Show Debug Status", value=False):
            status = get_debug_status()
            st.sidebar.write("**Debug Status:**")
            for category, enabled in status.items():
                emoji = "✅" if enabled else "❌"
                st.sidebar.write(f"{emoji} {category.capitalize()}: {enabled}")
    except:
        pass  # Not in Streamlit context

# Initialize debug status display
debug_status_display() 