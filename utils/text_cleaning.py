# utils/text_cleaning.py

from typing import List
from utils.token_counting import count_tokens
from sentence_transformers import util


def token_aware_compressor(passages, question, model, max_tokens=4000):
    # Step 1: Compute embeddings
    question_embedding = model.encode(question, convert_to_tensor=True)
    passage_embeddings = model.encode(passages, convert_to_tensor=True)

    # Step 2: Compute similarities
    similarities = util.cos_sim(question_embedding, passage_embeddings)[0]

    # Step 3: Sort passages by similarity
    ranked = sorted(zip(passages, similarities), key=lambda x: x[1], reverse=True)

    # Step 4: Collect top passages until token budget is full
    selected_passages = []
    token_sum = 0

    for passage, score in ranked:
        t = count_tokens(passage)
        if token_sum + t <= max_tokens:
            selected_passages.append(passage)
            token_sum += t
        if token_sum >= max_tokens:
            break

    return "\n\n".join(selected_passages)


def truncate_passages(passages, max_tokens=4000, model_name="gpt-4o"):
    """
    Build context by counting real tokens, not characters.
    """
    context = ""
    total_tokens = 0

    for i, passage in enumerate(passages):
        passage_tokens = count_tokens(passage, model_name)

        # Always allow first passage even if it slightly exceeds the budget
        if total_tokens + passage_tokens <= max_tokens or i == 0:
            context += "\n\n" + passage
            total_tokens += passage_tokens
        else:
            break

    return context


def quick_summarize(passages: List[str], max_passages: int = 10) -> List[str]:
    """
    Simple strategy: take first 500 characters of the top N passages.
    """
    return [p[:500] for p in passages[:max_passages]]
