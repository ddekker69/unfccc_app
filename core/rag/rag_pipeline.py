# rag_pipeline.py

import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from utils.token_counting import count_tokens
from utils.text_cleaning import truncate_passages, quick_summarize
from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_CONTEXT_TOKENS,
    LOCAL_MODEL_PATH,
    IS_STREAMLIT_CLOUD,
    AZURE_MODEL_BLOB_NAME,
    AZURE_CONTAINER_NAME
)
from utils.azure_blob_utils import download_blob

# Step 1: Ensure model exists
import os

if IS_STREAMLIT_CLOUD:
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("📦 Downloading SentenceTransformer model from Azure...")
        success = download_blob(AZURE_CONTAINER_NAME, AZURE_MODEL_BLOB_NAME, LOCAL_MODEL_PATH)
        if not success:
            raise RuntimeError("❌ Failed to download model from Azure Blob Storage.")
else:
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"❌ Local model not found at {LOCAL_MODEL_PATH}. Please download it manually.")

# Step 2: Load model + OpenAI client
model = SentenceTransformer(LOCAL_MODEL_PATH)
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# --- Compress, truncate, and select top-k passages ---
def retrieve_context(passages, top_k=5):
    selected_passages = passages[:top_k]

    if count_tokens(" ".join(selected_passages)) > MAX_CONTEXT_TOKENS:
        print("⚠ Context too long, triggering pre-summarization")
        selected_passages = quick_summarize(selected_passages)

    context = truncate_passages(selected_passages, MAX_CONTEXT_TOKENS)
    return context, selected_passages


# --- Generate Answer via OpenAI ---
def generate_answer(question, context, model_name=None):
    if not model_name:
        model_name = OPENAI_MODEL

    # ENHANCED PROMPTING: Use structured format with better instructions
    # Use centralized PromptTemplate system
    from utils.prompt_templates import PromptTemplate, ResponseFormat
    
    enhanced_prompt = PromptTemplate.build_enhanced_prompt(
        question=question,
        context=context,
        format_type=ResponseFormat.DETAILED,
        model_type="openai"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": enhanced_prompt}]
    )

    return response.choices[0].message.content.strip(), count_tokens(enhanced_prompt)


# --- Unified Answer Pipeline (document or country level) ---
def answer_question(question, passages, model_name=None, top_k=5):
    context, used_passages = retrieve_context(passages, top_k=top_k)

    if not context.strip():
        return "No sufficient information found.", used_passages, 0

    answer, token_count = generate_answer(question, context, model_name)
    return answer, used_passages, token_count

# # # rag_pipeline.py
# #
# # import numpy as np
# # import openai
# # from sentence_transformers import SentenceTransformer
# # from utils.token_counting import count_tokens
# # from utils.text_cleaning import truncate_passages, quick_summarize
# # from config import OPENAI_API_KEY, OPENAI_MODEL, USE_GPT, MAX_CONTEXT_TOKENS
# #
# # client = openai.OpenAI(api_key=OPENAI_API_KEY)
# #
# # model = SentenceTransformer('all-MiniLM-L6-v2')
# #
# #
# # def retrieve_context(passages, top_k=5):
# #     """
# #     Compress, truncate, and select top_k passages dynamically to fit token budget.
# #     """
# #     selected_passages = passages[:top_k]  # TODO: later improve with similarity search
# #
# #     if count_tokens(" ".join(selected_passages)) > MAX_CONTEXT_TOKENS:
# #         print("⚠ Context too long, triggering pre-summarization")
# #         selected_passages = quick_summarize(selected_passages)
# #
# #     context = truncate_passages(selected_passages, MAX_CONTEXT_TOKENS)
# #
# #     return context, selected_passages
# #
# #
# # def generate_answer(question, context, model_name=None):
# #     if not model_name:
# #         model_name = OPENAI_MODEL
# #
# #     prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
# #
# #     response = client.chat.completions.create(
# #         model=model_name,
# #         messages=[{"role": "user", "content": prompt}]
# #     )
# #
# #     return response.choices[0].message.content.strip(), count_tokens(prompt)
# #
# #
# # def answer_question(question, passages, model_name=None, top_k=5):
# #     context, used_passages = retrieve_context(passages, top_k=top_k)
# #
# #     if not context.strip():
# #         return "No sufficient information found.", used_passages, 0
# #
# #     answer, token_count = generate_answer(question, context, model_name)
# #
# #     return answer, used_passages, token_count
#
# # revised_rag_pipeline.py
#
# import numpy as np
# import openai
# from sentence_transformers import SentenceTransformer
# from utils.token_counting import count_tokens
# from utils.text_cleaning import truncate_passages, quick_summarize
# from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_CONTEXT_TOKENS
# import os
#
# client = openai.OpenAI(api_key=OPENAI_API_KEY)
#
# # Try local model first
# LOCAL_MODEL_PATH = "models/all-MiniLM-L6-v2"
# if os.path.exists(LOCAL_MODEL_PATH):
#     model = SentenceTransformer(LOCAL_MODEL_PATH)
# else:
#     print("⚠ Local model not found. Attempting to load from HuggingFace.")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#
#
# def retrieve_context(passages, top_k=5):
#     """
#     Compress, truncate, and select top_k passages dynamically to fit token budget.
#     """
#     selected_passages = passages[:top_k]
#
#     total_tokens = count_tokens(" ".join(selected_passages))
#     if total_tokens > MAX_CONTEXT_TOKENS:
#         print("⚠ Context too long, triggering pre-summarization.")
#         selected_passages = quick_summarize(selected_passages)
#
#     context = truncate_passages(selected_passages, max_tokens=MAX_CONTEXT_TOKENS)
#     return context, selected_passages
#
#
# def generate_answer(question, context, model_name=None):
#     model_name = model_name or OPENAI_MODEL
#     prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
#     response = client.chat.completions.create(
#         model=model_name,
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content.strip(), count_tokens(prompt)
#
#
# def answer_question(question, passages, model_name=None, top_k=5):
#     context, used_passages = retrieve_context(passages, top_k=top_k)
#     if not context.strip():
#         return "No sufficient information found.", used_passages, 0
#     answer, token_count = generate_answer(question, context, model_name)
#     return answer, used_passages, token_count
