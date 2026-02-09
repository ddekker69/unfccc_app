"""
Memory management utilities for UNFCCC Climate QA System.
Helps optimize performance for different RAM configurations.
"""

import psutil
import gc
import torch
import streamlit as st
from typing import Dict, List, Tuple


def get_system_memory_info() -> Dict[str, float]:
    """
    Get comprehensive system memory information.
    
    Returns:
        Dictionary with memory statistics in GB
    """
    memory = psutil.virtual_memory()
    
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent_used': memory.percent,
        'free_gb': (memory.total - memory.used) / (1024**3)
    }


def recommend_models_for_ram(total_ram_gb: float) -> List[str]:
    """
    Recommend appropriate models based on available RAM.
    
    Args:
        total_ram_gb: Total system RAM in GB
        
    Returns:
        List of recommended model names in order of preference
    """
    if total_ram_gb >= 32:
        return [
            "DeepSeek-R1-Distill-Qwen-14B (Recommended)",
            "DeepSeek-R1-Distill-Qwen-7B (Faster)",
            "DeepSeek-R1-Distill-Llama-8B (Alternative)",
            "DeepSeek-LLM-7B-Chat (Fallback)"
        ]
    elif total_ram_gb >= 16:
        return [
            "DeepSeek-R1-Distill-Qwen-7B (Faster)",
            "DeepSeek-LLM-7B-Chat (Fallback)",
            "TinyLlama-1B (8GB RAM)"
        ]
    elif total_ram_gb >= 8:
        return [
            "TinyLlama-1B (8GB RAM)",
            "DistilGPT2 (Low Memory)",
            "FLAN-T5-Small (Ultra Light)"
        ]
    else:
        return [
            "DistilGPT2 (Low Memory)",
            "FLAN-T5-Small (Ultra Light)",
            "gpt-4o"  # Suggest API for very low RAM
        ]


def get_model_memory_requirements() -> Dict[str, Dict[str, float]]:
    """
    Get memory requirements for different models.
    
    Returns:
        Dictionary mapping model names to their memory requirements
    """
    return {
        "DeepSeek-R1-Distill-Qwen-14B (Recommended)": {
            "model_size_gb": 16,
            "min_ram_gb": 28,
            "recommended_ram_gb": 32
        },
        "DeepSeek-R1-Distill-Qwen-7B (Faster)": {
            "model_size_gb": 7,
            "min_ram_gb": 14,
            "recommended_ram_gb": 16
        },
        "DeepSeek-R1-Distill-Llama-8B (Alternative)": {
            "model_size_gb": 8,
            "min_ram_gb": 16,
            "recommended_ram_gb": 20
        },
        "DeepSeek-LLM-7B-Chat (Fallback)": {
            "model_size_gb": 7,
            "min_ram_gb": 14,
            "recommended_ram_gb": 16
        },
        "TinyLlama-1B (8GB RAM)": {
            "model_size_gb": 1.1,
            "min_ram_gb": 4,
            "recommended_ram_gb": 8
        },
        "DistilGPT2 (Low Memory)": {
            "model_size_gb": 0.3,
            "min_ram_gb": 2,
            "recommended_ram_gb": 4
        },
        "FLAN-T5-Small (Ultra Light)": {
            "model_size_gb": 0.08,
            "min_ram_gb": 1,
            "recommended_ram_gb": 2
        }
    }


def cleanup_memory():
    """
    Clean up memory by running garbage collection and clearing GPU cache.
    """
    # Python garbage collection
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def check_model_compatibility(model_name: str) -> Tuple[bool, str]:
    """
    Check if a model is compatible with current system memory.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        Tuple of (is_compatible, message)
    """
    memory_info = get_system_memory_info()
    model_reqs = get_model_memory_requirements()
    
    if model_name not in model_reqs:
        return True, f"Model {model_name} not in database, assuming compatible"
    
    req = model_reqs[model_name]
    available_gb = memory_info['available_gb']
    
    if available_gb >= req['recommended_ram_gb']:
        return True, f"✅ Excellent: {available_gb:.1f}GB available, {req['recommended_ram_gb']}GB recommended"
    elif available_gb >= req['min_ram_gb']:
        return True, f"⚡ Good: {available_gb:.1f}GB available, {req['min_ram_gb']}GB minimum"
    else:
        return False, f"❌ Insufficient: {available_gb:.1f}GB available, {req['min_ram_gb']}GB required"


def display_memory_status():
    """
    Display current memory status in Streamlit.
    """
    memory_info = get_system_memory_info()
    
    # Create columns for memory display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total RAM",
            f"{memory_info['total_gb']:.1f} GB",
            help="Total system memory"
        )
    
    with col2:
        st.metric(
            "Available RAM", 
            f"{memory_info['available_gb']:.1f} GB",
            help="Memory available for applications"
        )
    
    with col3:
        usage_color = "🟢" if memory_info['percent_used'] < 70 else "🟡" if memory_info['percent_used'] < 85 else "🔴"
        st.metric(
            "Memory Usage",
            f"{memory_info['percent_used']:.1f}% {usage_color}",
            help="Percentage of memory currently in use"
        )
    
    # Memory recommendations
    recommended_models = recommend_models_for_ram(memory_info['total_gb'])
    
    st.info(f"💡 **Recommended models for your {memory_info['total_gb']:.1f}GB system:**")
    for i, model in enumerate(recommended_models[:3]):  # Show top 3
        st.write(f"   {i+1}. {model}")


def get_optimal_context_size(model_name: str, available_ram_gb: float) -> int:
    """
    Get optimal context size based on model and available RAM.
    
    Args:
        model_name: Name of the model
        available_ram_gb: Available RAM in GB
        
    Returns:
        Recommended max context tokens
    """
    if "tinyllama" in model_name.lower() or available_ram_gb < 8:
        return 2000  # Keep it small for low RAM
    elif "distilgpt2" in model_name.lower() or available_ram_gb < 12:
        return 3000  # Medium context
    elif "flan-t5" in model_name.lower():
        return 1000  # T5 models are efficient but prefer shorter context
    elif available_ram_gb < 16:
        return 4000  # Standard context for medium RAM
    else:
        return 8000  # Full context for high RAM systems 