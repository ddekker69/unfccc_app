# rag_engine.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.token_counting import count_tokens
from utils.text_cleaning import truncate_passages, token_aware_compressor
from utils.azure_blob_utils import download_and_extract_model_from_azure
from azure.storage.blob import BlobServiceClient
import openai
import os
import traceback
import streamlit as st
from utils.data_loader import resolve_path
import time
import psutil

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_CONTEXT_TOKENS,
    LOCAL_MODEL_PATH,
    IS_STREAMLIT_CLOUD,
    AZURE_CONNECTION_STRING,
    AZURE_CONTAINER_NAME,
    AZURE_MODEL_BLOB_NAME,
    EMBEDDING_MODEL_NAME
)
# print("🔑 OpenAI API key loaded:", bool(OPENAI_API_KEY))

# === Configuration for token limits ===
# Local models: High limits since there are no API costs
LOCAL_MODEL_MAX_TOKENS = 800

# OpenAI API: Conservative limits to control costs  
OPENAI_MAX_TOKENS = 250  # ~$0.01-0.02 per response with GPT-4

# Ensure no proxy variables are passed by unsetting them (optional, for clarity)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

# === Suppress annoying PyTorch warnings on Apple Silicon ===
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Suppress torch warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*_path.*")

# Also suppress transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Reduce logging verbosity for cleaner output
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Initialize OpenAI client (v1.0+ style)
try:
    if OPENAI_API_KEY and OPENAI_API_KEY != "your-openai-api-key-here":
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized successfully")
    else:
        client = None
        print("⚠️ OpenAI API key not configured - using local models only")
except Exception as e:
    client = None
    print(f"⚠️ OpenAI client initialization failed: {e} - using local models only")

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    """
    Load SentenceTransformer lazily.
    This avoids blocking app startup with model download/loading on import.
    """
    if IS_STREAMLIT_CLOUD:
        if not os.path.exists(LOCAL_MODEL_PATH):
            print("📦 Model not found locally. Downloading from Azure...")
            success = download_and_extract_model_from_azure(
                container_name=AZURE_CONTAINER_NAME,
                blob_name=AZURE_MODEL_BLOB_NAME,
                extract_to="models/",
            )
            if not success:
                print("❌ Failed to download from Azure, using Hugging Face model")
                return SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"✅ Model loaded from Azure: {LOCAL_MODEL_PATH}")
            return SentenceTransformer(str(LOCAL_MODEL_PATH))
        print(f"✅ Model already exists at {LOCAL_MODEL_PATH}")
        return SentenceTransformer(str(LOCAL_MODEL_PATH))

    # Local development: prefer cached local model path, fallback to HF model name.
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"✅ Using local model: {LOCAL_MODEL_PATH}")
        return SentenceTransformer(str(LOCAL_MODEL_PATH))
    print(f"⚠️ Local model not found at {LOCAL_MODEL_PATH}, downloading from Hugging Face")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# # === OpenAI client ===
# client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Load FAISS Indexes ---
@st.cache_resource(show_spinner="Loading model...")
def load_cluster_index(cluster_id):
    # Use resolve_path to check if the file exists locally, or download from Azure if running on Streamlit Cloud
    index_file = resolve_path(f"indexes/cluster_{cluster_id}.index")
    embeddings_file = resolve_path(f"embeddings/cluster_{cluster_id}.pkl")

    # Check if files exist
    if not os.path.exists(str(index_file)) or not os.path.exists(str(embeddings_file)):
        st.warning(f"⚠️ Legacy standard index files for cluster {cluster_id} were not found.")
        st.info("💡 This project now uses enhanced-only retrieval. Build enhanced indexes with `python scripts/automated_pipeline.py --skip-app`.")
        return None, None

    # Ensure that if the return value is a Path object, convert it to string for faiss and open()
    try:
        index = faiss.read_index(str(index_file))
        with open(str(embeddings_file), 'rb') as f:
            data = pickle.load(f)
        return index, data
    except Exception as e:
        st.error(f"❌ Error loading index for cluster {cluster_id}: {e}")
        return None, None

# @st.cache_resource(show_spinner="Loading model...")
# def load_cluster_index(cluster_id):
#     index = faiss.read_index(f'indexes/cluster_{cluster_id}.index')
#     with open(f'embeddings/cluster_{cluster_id}.pkl', 'rb') as f:
#         data = pickle.load(f)
#     return index, data
@st.cache_resource(show_spinner="Loading model...")
def load_global_index():
    try:
        if not os.path.exists('indexes/global.index') or not os.path.exists('embeddings/global.pkl'):
            st.warning("⚠️ Global index not found. Q&A functionality limited.")
            return None, None
        
        index = faiss.read_index('indexes/global.index')
        with open('embeddings/global.pkl', 'rb') as f:
            data = pickle.load(f)
        return index, data
    except Exception as e:
        st.error(f"❌ Error loading global index: {e}")
        return None, None

# --- Retrieve top-k matching documents ---
@st.cache_resource(show_spinner="Loading index...")
def retrieve(query, _index, data, top_k=5, year_range=None):
    embedding_model = get_embedding_model()
    query_vec = embedding_model.encode([query])
    
    # If year filtering is enabled and publication years are available
    if year_range and 'publication_years' in data:
        min_year, max_year = year_range
        
        # Search more results to allow for filtering
        search_k = min(top_k * 3, _index.ntotal)
        scores, indices = _index.search(np.array(query_vec), search_k)
        
        # Filter by year
        filtered_results = []
        for score, idx in zip(scores[0], indices[0]):
            pub_year = data['publication_years'][idx]
            
            # Include if year is in range (or None if we allow unknown years)
            if pub_year is None or (min_year <= pub_year <= max_year):
                filtered_results.append((score, idx))
        
        # Sort by score and take top_k
        filtered_results.sort(key=lambda x: x[0])  # Lower score is better for L2 distance
        filtered_results = filtered_results[:top_k]
        
        # Return texts
        return [data['texts'][idx] for score, idx in filtered_results]
    else:
        # No filtering, return original results
        scores, indices = _index.search(np.array(query_vec), top_k)
        return [data['texts'][i] for i in indices[0]]

# --- Compress context with token counting ---
def compress_context(passages, max_tokens=4000, model_name="gpt-4o"):
    """Build context with better structure and document separation."""
    context_parts = []
    total_tokens = 0
    
    for i, passage in enumerate(passages):
        # Add clear document separation
        doc_header = f"\n--- DOCUMENT {i+1} ---\n"
        formatted_passage = doc_header + passage.strip()
        
        passage_tokens = count_tokens(formatted_passage, model_name)
        if total_tokens + passage_tokens <= max_tokens or total_tokens == 0:
            context_parts.append(formatted_passage)
            total_tokens += passage_tokens
        else:
            break
    
    # Add a summary header
    context = f"=== RETRIEVED DOCUMENTS ({len(context_parts)} documents) ===" + "\n".join(context_parts)
    return context

# --- Generate answer using OpenAI or local model ---
def generate_answer(question, context, model_name=None):
    st.info(f"🧠 **DEBUG**: Starting answer generation")
    st.info(f"   🎯 Model: {model_name or 'auto-detect'}")
    st.info(f"   📏 Context length: {len(context)} characters")
    
    if not client:
        st.info(f"🔧 **DEBUG**: No OpenAI client - using local DeepSeek model")
        # Fallback to local model if OpenAI not available - FIXED: Pass model_name
        return generate_answer_local(question, context, model_name=model_name)
    
    model_name = model_name or OPENAI_MODEL
    
    # ENHANCED PROMPTING: Use centralized PromptTemplate system
    from utils.prompt_templates import PromptTemplate, ResponseFormat
    
    enhanced_prompt = PromptTemplate.build_enhanced_prompt(
        question=question,
        context=context,
        format_type=ResponseFormat.DETAILED,
        model_type="openai" if "gpt" in model_name.lower() else "general"
    )
    
    try:
        st.info(f"☁️ **DEBUG**: Calling OpenAI API ({model_name})")
        st.info(f"   🎯 Max tokens: {OPENAI_MAX_TOKENS}")
        
        # Note: Using OpenAI API with conservative token limits to control costs
        # Each response costs ~$0.01-0.02 with GPT-4o at current pricing
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": enhanced_prompt}],
            max_tokens=OPENAI_MAX_TOKENS,  # Conservative limit to control API costs
            temperature=0.7
        )
        
        answer = response.choices[0].message.content.strip()
        st.success(f"✅ **DEBUG**: OpenAI response received ({len(answer)} chars)")
        return answer
    except Exception as e:
        st.warning(f"⚠️ **DEBUG**: OpenAI error: {e}. Falling back to local DeepSeek model.")
        return generate_answer_local(question, context, model_name=model_name)

# Initialize global model cache - ENHANCED: Support multiple models
_local_qa_models = {}  # Dictionary to cache multiple models
_default_model_name = None
MODEL_NAME_MAPPING = {
    "DeepSeek-R1-Distill-Qwen-14B (Recommended)": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-7B (Faster)": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B (Alternative)": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen3-4B-Instruct (Fast Recommended)": "Qwen/Qwen3-4B-Instruct-2507",
    "Phi-4-mini-instruct (Efficient)": "microsoft/Phi-4-mini-instruct",
    "Qwen2.5-3B-Instruct (Light)": "Qwen/Qwen2.5-3B-Instruct",
    "SmolLM2-1.7B-Instruct (Ultra Light)": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "TinyLlama-1B (8GB RAM)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt-4o": "gpt-4o",
}

@st.cache_resource(show_spinner="Loading local model (one-time setup)...")
def get_cached_local_model(preferred_model_name=None):
    """Load and cache the local model, with support for specific model selection."""
    global _local_qa_models, _default_model_name
    
    # Convert display name to actual model name if needed
    if preferred_model_name and preferred_model_name in MODEL_NAME_MAPPING:
        actual_model_name = MODEL_NAME_MAPPING[preferred_model_name]
        st.info(f"🎯 **DEBUG**: Requested specific model: {preferred_model_name} -> {actual_model_name}")
    else:
        actual_model_name = preferred_model_name
    
    # If OpenAI model requested, return None (handled elsewhere)
    if actual_model_name == "gpt-4o":
        st.info(f"☁️ **DEBUG**: OpenAI model requested, handled by OpenAI API")
        return None, actual_model_name
    
    # Check if the specific model is already cached
    if actual_model_name and actual_model_name in _local_qa_models:
        st.info(f"✅ **DEBUG**: Using cached model: {actual_model_name}")
        return _local_qa_models[actual_model_name], actual_model_name
    
    # If we have a default model and no specific model requested, use it
    if not actual_model_name and _default_model_name and _default_model_name in _local_qa_models:
        st.info(f"✅ **DEBUG**: Using default cached model: {_default_model_name}")
        return _local_qa_models[_default_model_name], _default_model_name
    
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    
    # Configure device settings for cross-platform compatibility
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()  # Clear MPS cache (Apple Silicon)
        # Additional MPS configuration
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    elif torch.cuda.is_available():
        # Windows/Linux CUDA optimizations
        torch.cuda.empty_cache()  # Clear CUDA cache
        # Optimize CUDA memory allocation for Windows
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    
    # Auto-detect RAM and suggest appropriate models
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    st.info(f"💾 **DEBUG**: Detected {total_ram_gb:.1f}GB total RAM")
    
    # Model options organized by RAM requirements
    if total_ram_gb >= 28:
        # High-end systems (28GB+)
        model_options = [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",    # Best reasoning model
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",     # Faster reasoning model  
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",    # Alternative reasoning model
            "Qwen/Qwen3-4B-Instruct-2507",                 # Strong quality/speed tradeoff
            "microsoft/Phi-4-mini-instruct",               # Efficient modern small model
            "Qwen/Qwen2.5-3B-Instruct",                    # Lightweight fallback
        ]
        st.info(f"🚀 **DEBUG**: High RAM system - using premium models")
    elif total_ram_gb >= 14:
        # Medium systems (14-28GB)
        model_options = [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",     # Best option for medium RAM
            "Qwen/Qwen3-4B-Instruct-2507",                 # Fast recommended
            "microsoft/Phi-4-mini-instruct",               # Efficient
            "Qwen/Qwen2.5-3B-Instruct",                    # Light fallback
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",         # Ultra light fallback
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",          # Emergency fallback
        ]
        st.info(f"⚡ **DEBUG**: Medium RAM system - using balanced local models")
    else:
        # Low RAM systems (< 14GB) - Use small models only
        model_options = [
            "Qwen/Qwen2.5-3B-Instruct",                    # Best lightweight modern model
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",         # Very light
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",          # Best small model
            "distilgpt2",                                   # Legacy emergency fallback
        ]
        st.warning(f"💾 **DEBUG**: Low RAM system ({total_ram_gb:.1f}GB) - using lightweight models")
        st.info("💡 **TIP**: Consider using OpenAI API for better quality responses")
    
    # If a specific model is requested, try it first
    if actual_model_name and actual_model_name not in model_options:
        model_options.insert(0, actual_model_name)
    elif actual_model_name and actual_model_name in model_options:
        # Move the requested model to the front
        model_options.remove(actual_model_name)
        model_options.insert(0, actual_model_name)
    
    for model_name in model_options:
        try:
            st.info(f"🤖 Loading {model_name} (will cache in memory)...")
            
            if "deepseek" in model_name.lower():
                # Use DeepSeek models with optimized settings for M4 Max
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Set pad token if missing
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Optimized settings for Apple Silicon
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Use float16 for memory efficiency
                    device_map="auto",          # Auto device mapping
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,     # Optimize for memory
                )
                
                qa_model = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=4096,            # Increased to accommodate longer responses
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    return_full_text=False      # Only return new text
                )
            elif any(keyword in model_name.lower() for keyword in ["qwen", "phi-4-mini", "smollm2"]):
                # Modern lightweight instruct models
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                preferred_dtype = torch.float16 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=preferred_dtype,
                    device_map="auto" if (torch.cuda.is_available() or torch.backends.mps.is_available()) else "cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

                qa_model = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=3072,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    return_full_text=False,
                )
            elif "tinyllama" in model_name.lower():
                # Optimized settings for TinyLlama (good for 8GB RAM)
                st.info(f"🐣 **DEBUG**: Loading TinyLlama with 8GB RAM optimizations")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Lightweight settings for low RAM
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Essential for memory efficiency
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                
                qa_model = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=2048,            # Smaller for speed
                    do_sample=True,
                    temperature=0.8,            # Slightly higher for creativity
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    return_full_text=False
                )
            elif "distilgpt2" in model_name.lower() or "dialogpt" in model_name.lower():
                # Very lightweight models for emergency fallback
                st.info(f"⚡ **DEBUG**: Loading lightweight model with minimal settings")
                qa_model = pipeline(
                    "text-generation", 
                    model=model_name,
                    max_length=1024,            # Keep it small
                    do_sample=True,
                    temperature=0.7,
                    device=0 if torch.cuda.is_available() else -1
                )
            else:
                # Fallback to simpler models
                qa_model = pipeline("text-generation", model=model_name, max_length=1024)
            
            # Cache the successfully loaded model
            _local_qa_models[model_name] = qa_model
            
            # Set as default if this is the first successful model
            if _default_model_name is None:
                _default_model_name = model_name
            
            st.success(f"✅ Model {model_name} loaded and cached in memory!")
            st.info(f"📊 **DEBUG**: Total cached models: {len(_local_qa_models)}")
            
            return qa_model, model_name
            
        except Exception as e:
            st.warning(f"Failed to load {model_name}: {e}")
            continue
    
    raise Exception("No models could be loaded")

def generate_answer_local(question, context, model_name=None):
    """Generate answer using cached local Hugging Face model (free)."""
    st.info(f"🤖 **DEBUG**: Starting LOCAL DeepSeek generation")
    st.info(f"   🎯 **DEBUG**: Requested model: {model_name or 'default'}")
    st.info(f"   📏 Context: {len(context)} chars")
    
    try:
        # Get the model - ENHANCED: Pass preferred model name for selection
        qa_model, actual_model_name = get_cached_local_model(preferred_model_name=model_name)
        
        # Handle OpenAI models (return None from get_cached_local_model)
        if qa_model is None:
            st.info(f"☁️ **DEBUG**: Model {model_name} is handled by OpenAI API, not local generation")
            return "This model requires OpenAI API configuration."
        
        st.success(f"🧠 **DEBUG**: Using model: {actual_model_name}")
        
        # Add context length warning with model-specific time estimates (now that we have actual_model_name)
        if len(context) > 15000:
            time_estimate = get_model_time_estimate(actual_model_name, is_ultra_fast=False)
            st.warning(f"⚠️ **DEBUG**: Large context ({len(context)} chars) - this may take {time_estimate}")
            st.info("💡 **TIP**: Consider using enhanced preprocessing for faster generation")
        
        # Show which model is actually being used vs requested
        if model_name and model_name != actual_model_name:
            mapped_name = MODEL_NAME_MAPPING.get(model_name, model_name)
            if mapped_name == actual_model_name:
                st.success(f"✅ **DEBUG**: Using requested model: {model_name}")
            else:
                st.info(f"📝 **DEBUG**: Requested: {model_name}, Using: {actual_model_name}")
        
        # Optimize context for DeepSeek models
        if len(context) > 20000:
            st.warning(f"📏 **DEBUG**: Truncating large context from {len(context)} to 15000 chars")
            # Keep the most relevant parts (beginning and end)
            context = context[:7500] + "\n\n... [middle content truncated for performance] ...\n\n" + context[-7500:]
        
        # Enhanced prompt for reasoning models
        if any(keyword in actual_model_name.lower() for keyword in ["deepseek", "r1"]):
            st.info(f"🧠 **DEBUG**: Using DeepSeek R1 reasoning prompt format")
            prompt = f"""<|im_start|>system
You are a specialized AI assistant for climate policy analysis. Your role is to provide accurate, well-structured, and actionable insights from UNFCCC climate documents.

CORE RESPONSIBILITIES:
• Analyze climate policy documents with precision and depth
• Synthesize information from multiple sources coherently
• Provide evidence-based answers with proper context
• Maintain objectivity and scientific accuracy
• Focus on practical implications and policy relevance

RESPONSE FORMAT - DETAILED ANALYSIS:
1. **Executive Summary** (2-3 sentences)
   - Key finding or direct answer to the question
   - Most critical insight or recommendation

2. **Main Analysis** (structured paragraphs)
   - Detailed explanation with supporting evidence
   - Reference specific documents, countries, or policies
   - Include relevant context and background

3. **Supporting Evidence**
   - Quote or reference specific document excerpts
   - Mention countries, dates, or specific commitments
   - Highlight patterns or trends across documents

4. **Key Takeaways** (bullet points)
   - 3-5 actionable insights
   - Clear, concise statements
   - Prioritized by importance

IMPORTANT FORMATTING GUIDELINES:
• Start responses immediately with substantive content
• Use markdown formatting for structure (##, **, •)
• Keep explanations clear and focused
• Cite specific countries or document details when relevant
• End with actionable insights
<|im_end|>
<|im_start|>user
Based on the following excerpts from UNFCCC climate policy documents, please answer the question with the specified format and quality standards.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

Please provide a comprehensive, well-structured response following the format guidelines above.
<|im_end|>
<|im_start|>assistant
"""
        else:
            st.info(f"🧠 **DEBUG**: Using enhanced standard prompt format")
            prompt = f"""Human: You are a specialized climate policy expert. Based on the following relevant excerpts from UNFCCC documents, please provide a comprehensive, well-structured analysis.

RESPONSE FORMAT:
1. **Executive Summary** - Direct answer and key insight
2. **Main Analysis** - Detailed explanation with evidence
3. **Supporting Evidence** - Specific quotes and references
4. **Key Takeaways** - Actionable insights (3-5 bullet points)

QUALITY STANDARDS:
• Use clear, professional language appropriate for policy makers
• Structure responses logically with proper flow between ideas
• Support statements with specific evidence from the documents
• Distinguish between facts, interpretations, and recommendations

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

Please provide a comprehensive analysis that synthesizes information from multiple sources and follows the specified format.
"""
        
        st.info(f"📤 **DEBUG**: Sending to DeepSeek model...")
        st.info(f"   📏 Full prompt: {len(prompt)} chars")
        st.info(f"   🎯 Max new tokens: {LOCAL_MODEL_MAX_TOKENS}")
        
        # Add progress indicator with model-specific time estimates
        progress_placeholder = st.empty()
        model_display_name = actual_model_name.split('/')[-1] if '/' in actual_model_name else actual_model_name
        
        time_estimate = get_model_time_estimate(actual_model_name, is_ultra_fast=False)
        progress_placeholder.info(f"🔄 **{model_display_name} is thinking... This may take {time_estimate} for complex questions**")
        
        # Optimize generation settings for faster responses
        generation_params = {
            'max_new_tokens': min(LOCAL_MODEL_MAX_TOKENS, 400),  # Reduce for faster generation
            'num_return_sequences': 1,
            'temperature': 0.7,
            'do_sample': True,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'pad_token_id': qa_model.tokenizer.eos_token_id
        }
        
        # ENHANCED: Model-specific generation parameters
        if "tinyllama" in actual_model_name.lower():
            # TinyLlama needs more tokens to complete full responses (was getting truncated)
            generation_params.update({
                'max_new_tokens': 800,  # INCREASED: Allow full responses from TinyLlama
                'temperature': 0.8,      # Slightly higher for better creativity
                'do_sample': True,
                'early_stopping': False, # Let it complete the response
                'repetition_penalty': 1.15  # Prevent repetition in longer responses
            })
            st.info(f"🐣 **DEBUG**: Using TinyLlama-optimized generation (max_tokens={generation_params['max_new_tokens']})")
        elif any(
            keyword in actual_model_name.lower()
            for keyword in ["qwen", "phi-4-mini", "smollm2", "distilgpt2"]
        ):
            # Lightweight instruct models
            generation_params.update({
                'max_new_tokens': 600,   # INCREASED: More tokens for complete responses
                'temperature': 0.7,
                'early_stopping': False
            })
            st.info(f"⚡ **DEBUG**: Using lightweight model generation (max_tokens={generation_params['max_new_tokens']})")
        elif any(keyword in actual_model_name.lower() for keyword in ["deepseek", "r1"]):
            # DeepSeek models with thinking process - keep shorter for speed
            generation_params.update({
                'max_new_tokens': min(LOCAL_MODEL_MAX_TOKENS, 300),  # Shorter for R1 models
                'early_stopping': True,
                'num_beams': 1,  # Disable beam search for speed
                'do_sample': True
            })
            st.info(f"🧠 **DEBUG**: Using DeepSeek R1 generation (max_tokens={generation_params['max_new_tokens']})")
        
        st.info(f"⚙️ **DEBUG**: Generation params: max_tokens={generation_params['max_new_tokens']}")
        
        # Generate response with timeout and progress tracking
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Temporarily redirect stderr to suppress C++ level warnings
            import sys
            from io import StringIO
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            
            generation_start = time.time()
            try:
                # Add periodic progress updates
                import threading
                import time as time_module
                
                def progress_updater():
                    elapsed = 0
                    while elapsed < 120:  # 2 minute timeout
                        time_module.sleep(5)
                        elapsed += 5
                        try:
                            # Only update progress if we're still in Streamlit context
                            if elapsed <= 60:
                                progress_placeholder.info(f"🔄 **{model_display_name} thinking... {elapsed}s elapsed**")
                            else:
                                progress_placeholder.warning(f"⏰ **{model_display_name} still working... {elapsed}s elapsed (this is unusually long)**")
                        except Exception:
                            # Ignore Streamlit context errors from background thread
                            break
                
                # Start progress thread
                progress_thread = threading.Thread(target=progress_updater, daemon=True)
                progress_thread.start()
                
                result = qa_model(prompt, **generation_params)
                
                # Stop progress updates
                progress_placeholder.empty()
                
            except Exception as e:
                progress_placeholder.empty()
                st.error(f"❌ **DEBUG**: Generation failed: {e}")
                raise e
            finally:
                # Restore stderr
                sys.stderr = old_stderr
        
        generation_time = time.time() - generation_start
        
        if generation_time > 30:
            st.warning(f"⏰ **DEBUG**: Generation took {generation_time:.1f}s (consider using enhanced preprocessing)")
        else:
            st.success(f"⚡ **DEBUG**: {model_display_name} generation completed in {generation_time:.2f}s")
        
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0]['generated_text']
            st.info(f"📤 **DEBUG**: Raw output: {len(generated_text)} chars")
            
            # Clean up the response for DeepSeek R1 reasoning models
            if any(keyword in actual_model_name.lower() for keyword in ["deepseek", "r1"]):
                st.info(f"🧹 **DEBUG**: Cleaning DeepSeek R1 response (removing <think> tags)")
                # DeepSeek R1 models include thinking process - extract only the final answer
                answer = clean_deepseek_response_simple(generated_text)
                
                # If answer is too short after cleaning, it might be stuck in thinking
                if len(answer.strip()) < 50:
                    st.warning(f"⚠️ **DEBUG**: Short answer after cleaning - model may be stuck in thinking mode")
                    # Try to extract any meaningful content
                    if "</think>" in generated_text:
                        after_think = generated_text.split("</think>")[-1].strip()
                        if len(after_think) > 20:
                            answer = after_think
                        else:
                            model_display_name = actual_model_name.split('/')[-1] if '/' in actual_model_name else actual_model_name
                            answer = f"{model_display_name} is still thinking. Try reducing context size or using enhanced preprocessing."
                    else:
                        model_display_name = actual_model_name.split('/')[-1] if '/' in actual_model_name else actual_model_name
                        answer = f"{model_display_name} is still thinking. Try reducing context size or using enhanced preprocessing."
            else:
                st.info(f"🧹 **DEBUG**: Cleaning standard response")
                # Standard response cleaning for other models
                if "<|im_start|>assistant" in generated_text:
                    answer = generated_text.split("<|im_start|>assistant")[-1].strip()
                elif "Assistant:" in generated_text:
                    answer = generated_text.split("Assistant:")[-1].strip()
                elif "Answer:" in generated_text:
                    answer = generated_text.split("Answer:")[-1].strip()
                else:
                    answer = generated_text.strip()
                
                # Remove any end tokens
                answer = answer.replace("<|im_end|>", "").strip()
        else:
            st.error(f"❌ **DEBUG**: No valid result from model")
            answer = "Unable to generate response"
        
        st.success(f"✅ **DEBUG**: Final answer: {len(answer)} chars")
        st.info(f"💬 **DEBUG**: Answer preview: '{answer[:100]}...'")
        
        # Include model name in response for clarity
        model_display_name = actual_model_name.split('/')[-1] if '/' in actual_model_name else actual_model_name
        return f"🧠 **{model_display_name}**:\n\n{answer}"
        
    except Exception as e:
        model_display_name = actual_model_name.split('/')[-1] if '/' in actual_model_name else actual_model_name
        st.error(f"❌ **DEBUG**: {model_display_name} model error: {e}")
        # Fallback to simple Q&A model
        return generate_answer_simple_fallback(question, context)

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

def generate_answer_simple_fallback(question, context):
    """Simple fallback using basic Q&A model."""
    try:
        from transformers import pipeline
        
        @st.cache_resource
        def load_simple_qa():
            return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        
        qa_model = load_simple_qa()
        
        # Truncate context if too long
        max_length = 512
        if len(context) > max_length:
            context = context[:max_length] + "..."
        
        result = qa_model(question=question, context=context)
        confidence = result.get('score', 0)
        
        answer = result['answer']
        if confidence < 0.1:
            answer = f"⚠️ Low confidence answer: {answer}"
        
        return f"📖 Basic QA Answer: {answer}\n(Confidence: {confidence:.2f})"
        
    except Exception as e:
        return f"❌ All local models failed: {str(e)}. Please install: pip install transformers torch"

# --- Cross-cluster QA ---
def answer_cross_cluster_question(question, passages, model_name):
    embedding_model = get_embedding_model()
    context = token_aware_compressor(passages, question, embedding_model, max_tokens=MAX_CONTEXT_TOKENS)
    if not context.strip():
        return "No sufficient information found.", 0
    answer = generate_answer(question, context, model_name)
    token_count = count_tokens(context, model_name=model_name)
    return answer, token_count

# --- Cluster Summary Generator ---
def generate_cluster_summary(passages, model_name=None, max_tokens=4000):
    if not client:
        return "❌ OpenAI API key not configured. Cannot generate summary."
        
    model_name = model_name or OPENAI_MODEL
    context = truncate_passages(passages, max_tokens=max_tokens, model_name=model_name)
    prompt = f"Summarize the following collection of climate-related documents.\n\nContext:\n{context}\n\nSummary:"

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except openai.error.RateLimitError as e:
        return f"❌ Rate limit error: {str(e)}"

# --- Cluster-level QA ---
def answer_question(question, cluster_id, model_name=None, top_k=5, max_tokens=MAX_CONTEXT_TOKENS, year_range=None):
    # DEBUG: Add cluster information
    st.info(f"🔍 **DEBUG**: STANDARD RAG PIPELINE STARTING")
    st.info(f"   🔍 Question: '{question[:50]}...'")
    st.info(f"   📂 Cluster: {cluster_id}")
    st.info(f"   🎯 Max tokens: {max_tokens}")
    st.info(f"   📊 Top-k: {top_k}")
    
    index, data = load_cluster_index(cluster_id)
    
    # Handle missing indexes
    if index is None or data is None:
        st.error(f"❌ **DEBUG**: FAISS indexes not available for cluster {cluster_id}")
        return "❌ FAISS indexes not available. Please run data preparation scripts to build indexes.", [], 0
    
    # DEBUG: Show what data we have
    st.info(f"🔍 **DEBUG**: Found standard index with {index.ntotal} documents for cluster {cluster_id}")
    if 'texts' in data:
        st.info(f"🔍 **DEBUG**: First few chars of first document: {data['texts'][0][:100]}...")
    
    st.info(f"🔍 **DEBUG**: Retrieving passages...")
    retrieval_start = time.time()
    passages = retrieve(question, index, data, top_k=top_k, year_range=year_range)
    retrieval_time = time.time() - retrieval_start
    
    st.success(f"📦 **DEBUG**: Retrieved {len(passages)} passages in {retrieval_time:.2f}s")
    
    # Use fast compression that ranks passages by semantic similarity to the question
    st.info(f"🔧 **DEBUG**: Compressing context...")
    compression_start = time.time()
    embedding_model = get_embedding_model()
    context = fast_context_compressor(passages, question, embedding_model, max_tokens=max_tokens)
    compression_time = time.time() - compression_start
    
    # Minimal debug info
    if context:
        excerpt_count = context.count("--- DOCUMENT")
        st.success(f"📄 **DEBUG**: Context compressed in {compression_time:.2f}s - {excerpt_count} excerpts, {len(context)} chars")
        
        # Show context preview
        st.info(f"📄 **DEBUG**: Context preview (first 200 chars):")
        st.code(context[:200] + "..." if len(context) > 200 else context)
    else:
        st.warning("⚠️ **DEBUG**: No context generated from compression")

    if not context.strip():
        st.error(f"❌ **DEBUG**: Empty context after compression")
        return "No sufficient information found.", passages, 0
        
    st.info(f"🧠 **DEBUG**: Generating answer...")
    answer_start = time.time()
    answer = generate_answer(question, context, model_name)
    answer_time = time.time() - answer_start
    
    st.success(f"✅ **DEBUG**: Answer generated in {answer_time:.2f}s")
    
    token_count = count_tokens(context, model_name=model_name or OPENAI_MODEL)
    
    total_time = retrieval_time + compression_time + answer_time
    st.success(f"🎉 **DEBUG**: STANDARD RAG COMPLETE in {total_time:.2f}s")
    st.info(f"   📦 Retrieval: {retrieval_time:.2f}s")
    st.info(f"   🔧 Compression: {compression_time:.2f}s")
    st.info(f"   🧠 Generation: {answer_time:.2f}s")
    
    return answer, passages, token_count

def fast_context_compressor(passages, question, embedding_model, max_tokens=4000):
    """
    Fast context compression - optimized for speed while maintaining quality.
    """
    try:
        from sentence_transformers import util
        
        # Quick token estimation (faster than precise counting)
        def estimate_tokens(text):
            return len(text) // 4  # Rough but fast estimate
        
        # Step 1: Quick filter - remove passages that are way too large
        filtered_passages = []
        for passage in passages:
            if estimate_tokens(passage) < max_tokens:  # Only keep manageable passages
                filtered_passages.append(passage)
            else:
                # Quick chunking for very large passages - just take first part
                chunk_size = max_tokens * 3  # Characters, not tokens
                filtered_passages.append(passage[:chunk_size])
        
        if not filtered_passages:
            # Fallback: just take first part of first passage
            return passages[0][:max_tokens * 3] if passages else ""
        
        # Step 2: Compute similarities (only for filtered passages)
        question_embedding = embedding_model.encode(question, convert_to_tensor=True)
        passage_embeddings = embedding_model.encode(filtered_passages, convert_to_tensor=True)
        similarities = util.cos_sim(question_embedding, passage_embeddings)[0]
        
        # Step 3: Sort and select best passages
        ranked = sorted(zip(filtered_passages, similarities), key=lambda x: x[1], reverse=True)
        
        # Step 4: Build context efficiently with automatic size adjustment for DeepSeek
        selected_parts = []
        total_chars = 0
        
        # Reduce max size if we detect this will go to DeepSeek (slow model)
        if max_tokens > 6000:
            max_chars = min(max_tokens * 3, 15000)  # Limit to 15k chars for DeepSeek
            st.info(f"🎯 **DEBUG**: Limiting context to {max_chars} chars for DeepSeek performance")
        else:
            max_chars = max_tokens * 3
        
        for i, (passage, score) in enumerate(ranked):
            header = f"\n--- DOCUMENT {i+1} (relevance: {score:.2f}) ---\n"
            formatted = header + passage.strip()
            
            if total_chars + len(formatted) <= max_chars:
                selected_parts.append(formatted)
                total_chars += len(formatted)
            else:
                # Try to fit a truncated version
                remaining_chars = max_chars - total_chars - len(header)
                if remaining_chars > 500:  # Only if we can fit a meaningful chunk
                    truncated = header + passage[:remaining_chars-100] + "... [truncated]"
                    selected_parts.append(truncated)
                break
        
        context = "".join(selected_parts)
        
        # Final size check and warning
        if len(context) > 20000:
            st.warning(f"⚠️ **DEBUG**: Large context generated ({len(context)} chars) - may be slow with DeepSeek")
            # Emergency truncation
            context = context[:15000] + "\n\n... [context truncated for performance] ..."
            st.info(f"✂️ **DEBUG**: Emergency truncation applied - reduced to {len(context)} chars")
        
        return context
        
    except Exception as e:
        # Simple fallback - just concatenate first few passages
        simple_context = "\n\n".join(passages[:3])
        max_chars = min(max_tokens * 3, 15000)  # Always limit for DeepSeek
        return simple_context[:max_chars] if len(simple_context) > max_chars else simple_context

def get_model_time_estimate(model_name, is_ultra_fast=False):
    """
    Get realistic time estimates based on model size and complexity.
    
    Args:
        model_name: The actual model name (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        is_ultra_fast: Whether this is for ultra-fast pipeline (generally faster)
    
    Returns:
        String with time estimate (e.g., "10-20 seconds")
    """
    model_lower = model_name.lower()
    
    if is_ultra_fast:
        # Ultra-fast estimates (enhanced preprocessing makes things faster)
        if "tinyllama" in model_lower:
            return "5-15 seconds"
        elif "smollm2" in model_lower:
            return "5-15 seconds"
        elif "phi-4-mini" in model_lower:
            return "8-20 seconds"
        elif "qwen3-4b" in model_lower:
            return "8-22 seconds"
        elif "qwen2.5-3b" in model_lower:
            return "7-20 seconds"
        elif "distilgpt2" in model_lower:
            return "3-10 seconds"
        elif any(keyword in model_lower for keyword in ["deepseek-r1-distill-qwen-7b", "7b"]):
            return "10-30 seconds"
        elif any(keyword in model_lower for keyword in ["deepseek-r1-distill-qwen-14b", "14b"]):
            return "15-40 seconds"
        elif "deepseek" in model_lower:
            return "12-35 seconds"
        else:
            return "8-25 seconds"
    else:
        # Standard pipeline estimates
        if "tinyllama" in model_lower:
            return "10-20 seconds"
        elif "smollm2" in model_lower:
            return "10-22 seconds"
        elif "phi-4-mini" in model_lower:
            return "12-28 seconds"
        elif "qwen3-4b" in model_lower:
            return "12-30 seconds"
        elif "qwen2.5-3b" in model_lower:
            return "10-26 seconds"
        elif "distilgpt2" in model_lower:
            return "5-15 seconds"
        elif any(keyword in model_lower for keyword in ["deepseek-r1-distill-qwen-7b", "7b"]):
            return "20-45 seconds"
        elif any(keyword in model_lower for keyword in ["deepseek-r1-distill-qwen-14b", "14b"]):
            return "30-60 seconds"
        elif "deepseek" in model_lower:
            return "25-50 seconds"
        else:
            return "15-40 seconds"
