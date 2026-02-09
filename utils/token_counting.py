# utils/token_counting.py

import tiktoken

# Auto-selects correct tokenizer for OpenAI models
def get_tokenizer(model_name="gpt-4o"):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text, model_name="gpt-4o"):
    """
    Accurate token count using OpenAI's tokenizer.
    """
    enc = get_tokenizer(model_name)
    tokens = enc.encode(text)
    return len(tokens)
