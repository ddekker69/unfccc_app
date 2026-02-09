# prepare_country_index.py

import os
import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME
import sys

# Ensure stdout can display emoji on Windows CMD
if sys.stdout.encoding is None or "UTF-8" not in sys.stdout.encoding.upper():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

def build_country_index():
    # === Config ===
    EMBEDDING_DIR = 'embeddings'
    INDEX_DIR = 'indexes'
    COUNTRY_DF_PATH = 'data/country_plot_df.pkl'
    TEXT_COLUMN = 'combined_text'
    ID_COLUMN = 'country'

    # === Setup ===
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # === Load Country-Level Data ===
    df = pd.read_pickle(COUNTRY_DF_PATH).dropna(subset=[TEXT_COLUMN, ID_COLUMN])

    # === Embed Country Representations ===
    texts = df[TEXT_COLUMN].tolist()
    ids = df[ID_COLUMN].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    # === Build and Save Index ===
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, os.path.join(INDEX_DIR, "country.index"))
    with open(os.path.join(EMBEDDING_DIR, "country.pkl"), "wb") as f:
        pickle.dump({'combined_texts': texts, 'ids': ids}, f)

    print("✅ Country-level index and embeddings saved.")

if __name__ == "__main__":
    build_country_index()

# # prepare_country_index.py
#
# import os
# import pandas as pd
# import faiss
# import numpy as np
# import pickle
# from sentence_transformers import SentenceTransformer
#
# # === Config ===
# MODEL_NAME = 'all-MiniLM-L6-v2'
# EMBEDDING_DIR = 'embeddings'
# INDEX_DIR = 'indexes'
# COUNTRY_DF_PATH = 'data/country_plot_df.pkl'
# TEXT_COLUMN = 'combined_text'
# ID_COLUMN = 'country'
#
# # === Setup ===
# os.makedirs(EMBEDDING_DIR, exist_ok=True)
# os.makedirs(INDEX_DIR, exist_ok=True)
# model = SentenceTransformer(MODEL_NAME)
#
# # === Load Country-Level Data ===
# df = pd.read_pickle(COUNTRY_DF_PATH).dropna(subset=[TEXT_COLUMN, ID_COLUMN])
#
# # === Embed Country Representations ===
# texts = df[TEXT_COLUMN].tolist()
# ids = df[ID_COLUMN].tolist()
# embeddings = model.encode(texts, show_progress_bar=True)
#
# # === Build and Save Index ===
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(np.array(embeddings))
#
# faiss.write_index(index, os.path.join(INDEX_DIR, "country.index"))
# with open(os.path.join(EMBEDDING_DIR, "country.pkl"), "wb") as f:
#     pickle.dump({'combined_texts': texts, 'ids': ids}, f)
#
# print("✅ Country-level index and embeddings saved.")