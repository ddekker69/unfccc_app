# build_embeddings.py
"""
Compute *all* document‑level and chunk‑level embeddings once and stash them on disk.
Rerunning is incremental: only unseen doc_ids are processed.
"""
import pickle
from core.pipeline.embedding_store import EmbeddingStore
from config import EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE
from core.pipeline.pipeline_bootstrap import check_folders, check_dependencies
from scripts.prepare_enhanced_index import chunk_text_intelligently   # reuse existing splitter

def build_embeddings_incremental():
    """Build embeddings incrementally for new documents only."""
    check_folders()
    check_dependencies()

    with open("data/extracted_texts.pkl", "rb") as f:
        df = pickle.load(f)

    df = df[df.status.eq("ok")].reset_index(drop=True)

    store = EmbeddingStore(EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE)

    processed_count = 0
    total_new = 0
    
    for _, row in df.iterrows():
        if store.has_doc(row.document_id):
            continue                               # already cached
        chunks = chunk_text_intelligently(str(row.text), chunk_size=400, overlap=50)
        store.add_document(row.document_id, str(row.text), chunks)
        processed_count += 1
        total_new += 1
        if processed_count % 10 == 0:
            print(f"✅ processed {processed_count} new documents")
    
    print(f"✅ Embedding generation complete! Processed {total_new} new documents.")
    return total_new

if __name__ == "__main__":
    build_embeddings_incremental()
