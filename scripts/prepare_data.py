# prepare_index.py
import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME
import pickle
from config import EMBEDDING_MODEL_NAME
import sys

# Ensure stdout can display emoji on Windows CMD
if sys.stdout.encoding is None or "UTF-8" not in sys.stdout.encoding.upper():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Configuration
# Use the model name from config
# EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Or replace with ClimateBERT model name
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

DATA_FILE = r'C:\Users\david\OneDrive - Heriot-Watt University\Hackathon\unfccc_data\unfccc_data\unfccc_documents_topic_mitigation_paginated_enriched.pkl'  # Full path to your document metadata
TEXT_COLUMN = 'combined_text'
CLUSTER_COLUMN = 'cluster'
ID_COLUMN = 'document_id'  # or 'country', as needed

EMBED_SAVE_PATH = 'embeddings/'
INDEX_SAVE_PATH = 'indexes/'
os.makedirs(EMBED_SAVE_PATH, exist_ok=True)
os.makedirs(INDEX_SAVE_PATH, exist_ok=True)

df = pd.read_pickle(DATA_FILE)
df = df.dropna(subset=[TEXT_COLUMN, CLUSTER_COLUMN])

# === Save cluster-wise chunks ===
for cluster in sorted(df[CLUSTER_COLUMN].unique()):
    cluster_df = df[df[CLUSTER_COLUMN] == cluster]
    texts = cluster_df[TEXT_COLUMN].tolist()
    ids = cluster_df[ID_COLUMN].tolist()

    embeddings = model.encode(texts, show_progress_bar=True)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))

    # Save index + metadata
    faiss.write_index(faiss_index, f'{INDEX_SAVE_PATH}/cluster_{cluster}.index')
    with open(f'{EMBED_SAVE_PATH}/cluster_{cluster}.pkl', 'wb') as f:
        pickle.dump({'texts': texts, 'ids': ids}, f)

# === Global Index ===
texts = df[TEXT_COLUMN].tolist()
ids = df[ID_COLUMN].tolist()
embeddings = model.encode(texts, show_progress_bar=True)
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(np.array(embeddings))
faiss.write_index(f'{INDEX_SAVE_PATH}/global.index')
with open(f'{EMBED_SAVE_PATH}/global.pkl', 'wb') as f:
    pickle.dump({'texts': texts, 'ids': ids}, f)

print("✅ Index preparation complete.")
