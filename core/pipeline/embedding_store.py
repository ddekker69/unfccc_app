# embedding_store.py
import os, pickle, numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
# from prepare_enhanced_index import create_document_summary

EMB_FILE = Path("embeddings_cache/doc_embeddings.pkl")      # doc_id → np.ndarray
CHUNK_FILE = Path("embeddings_cache/doc_chunks.pkl")        # doc_id → list[str]
CHUNK_EMB_FILE = Path("embeddings_cache/chunk_embeddings.pkl")  # doc_id → list[np.ndarray]
SUMMARY_FILE      = Path("embeddings_cache/doc_summaries.pkl")          # doc_id → str
SUMMARY_EMB_FILE  = Path("embeddings_cache/summary_embeddings.pkl")     # doc_id → np.ndarray


def create_document_summary(text, max_length=200):
    """Create a brief summary of the document (first meaningful paragraph)."""
    # Simple extraction of first substantial paragraph
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]

    if paragraphs:
        summary = paragraphs[0]
        if len(summary) > max_length:
            # Truncate at sentence boundary
            sentences = summary.split('. ')
            summary = '. '.join(sentences[:2]) + '.'
        return summary

    # Fallback: first 200 chars
    return text[:max_length] + "..." if len(text) > max_length else text

class EmbeddingStore:
    def __init__(self, model_name, device):
        self.model = SentenceTransformer(model_name, device=device)
        self._load()

    # ---------- public helpers ----------
    def has_doc(self, doc_id):          return doc_id in self.doc_embs
    def doc_vec(self, doc_id):          return self.doc_embs[doc_id]
    def chunks(self,  doc_id):          return self.doc_chunks[doc_id]
    def summary(self,     doc_id): return self.doc_summaries[doc_id]
    def summary_vec(self, doc_id): return self.doc_summary_embs[doc_id]
    def chunk_vecs(self, doc_id):       return self.doc_chunk_embs[doc_id]

    def add_document(self, doc_id, text, chunks):
        """Compute & cache everything for one new document."""
        summary         = create_document_summary(text)          # <- import / copy util
        doc_vec         = self.model.encode([text])[0]
        chunk_vecs      = self.model.encode(chunks, show_progress_bar=False)
        summary_vec     = self.model.encode([summary])[0]
        self.doc_embs[doc_id]         = doc_vec
        self.doc_chunks[doc_id]       = chunks
        self.doc_chunk_embs[doc_id]   = chunk_vecs
        self.doc_summaries[doc_id]    = summary
        self.doc_summary_embs[doc_id] = summary_vec
        self._flush()

    # ---------- internal ----------
    def _load(self):
        def _maybe_load(path, default):
            return pickle.loads(path.read_bytes()) if path.exists() else default
        self.doc_embs       = _maybe_load(EMB_FILE, {})
        self.doc_chunks     = _maybe_load(CHUNK_FILE, {})
        self.doc_chunk_embs = _maybe_load(CHUNK_EMB_FILE, {})
        self.doc_summaries     = _maybe_load(SUMMARY_FILE, {})
        self.doc_summary_embs  = _maybe_load(SUMMARY_EMB_FILE, {})
    def _flush(self):
        # ensure parent folder(s) exist
        for path, obj in [
            (EMB_FILE, self.doc_embs),
            (CHUNK_FILE, self.doc_chunks),
            (CHUNK_EMB_FILE, self.doc_chunk_embs),
            (SUMMARY_FILE, self.doc_summaries),
            (SUMMARY_EMB_FILE, self.doc_summary_embs),
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)  # <-- create folder if missing
            path.write_bytes(pickle.dumps(obj))
        EMB_FILE.write_bytes(pickle.dumps(self.doc_embs))
        CHUNK_FILE.write_bytes(pickle.dumps(self.doc_chunks))
        CHUNK_EMB_FILE.write_bytes(pickle.dumps(self.doc_chunk_embs))
        SUMMARY_FILE.write_bytes(pickle.dumps(self.doc_summaries))
        SUMMARY_EMB_FILE.write_bytes(pickle.dumps(self.doc_summary_embs))