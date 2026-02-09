# prepare_index.py

import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
import gc
import numpy as np
import torch
import sys

# Ensure stdout can display emoji on Windows CMD
if sys.stdout.encoding is None or "UTF-8" not in sys.stdout.encoding.upper():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from utils.azure_blob_utils import upload_blob
from config import AZURE_CONTAINER_NAME, EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE
from core.pipeline.pipeline_bootstrap import check_folders, check_dependencies

check_folders()
check_dependencies()

print(f"🤖 Using embedding model: {EMBEDDING_MODEL_NAME}")

# Use smart device selection
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=OPTIMAL_DEVICE)
print(f"✅ Model loaded on {model.device} with {model.get_sentence_embedding_dimension()} dimensions")

plot_df = pd.read_pickle("data/plot_df.pkl")
print(f"📊 Loaded {len(plot_df)} documents across {plot_df['cluster'].nunique()} clusters")

os.makedirs("indexes", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

# Process each cluster individually to manage memory
for cluster in sorted(plot_df['cluster'].unique()):
    print(f"\n🔧 Processing cluster {cluster}...")
    
    cluster_df = plot_df[plot_df['cluster'] == cluster]
    texts = cluster_df['text'].tolist()
    ids = cluster_df['document_id'].tolist()
    publication_years = cluster_df['publication_year'].tolist() if 'publication_year' in cluster_df.columns else [None] * len(texts)
    
    print(f"   📄 {len(texts)} documents in cluster {cluster}")
    
    if len(texts) == 0:
        print(f"   ⚠️ Skipping empty cluster {cluster}")
        continue

    # Generate embeddings for this cluster
    print(f"   🧮 Generating embeddings...")
    embeddings = model.encode(texts, batch_size=1, show_progress_bar=False)
    
    # Create and populate FAISS index
    print(f"   🔍 Creating FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    # Save files
    index_path = f"indexes/cluster_{cluster}.index"
    emb_path = f"embeddings/cluster_{cluster}.pkl"

    faiss.write_index(index, index_path)
    with open(emb_path, "wb") as f:
        pickle.dump({'texts': texts, 'ids': ids, 'publication_years': publication_years}, f)

    print(f"   ✅ Saved index ({embeddings.shape[1]}D) and metadata")
    
    # Upload to Azure Blob (ignore errors)
    try:
        upload_blob(AZURE_CONTAINER_NAME, f"indexes/cluster_{cluster}.index", index_path)
        upload_blob(AZURE_CONTAINER_NAME, f"embeddings/cluster_{cluster}.pkl", emb_path)
    except Exception as e:
        print(f"   ⚠️ Azure upload failed (continuing anyway): {e}")
    
    # Clear memory
    del embeddings, index
    gc.collect()

print("\n🎉 Index preparation complete!")

# # prepare_index.py
#
# import pandas as pd
# import faiss
# import pickle
# from sentence_transformers import SentenceTransformer
# import os
# from pipeline_bootstrap import check_folders, check_dependencies, check_dataframe
#
# check_folders()
# check_dependencies()
# check_dataframe(link_df)
#
# model = SentenceTransformer('all-MiniLM-L6-v2')
#
# plot_df = pd.read_pickle("data/plot_df.pkl")
#
# os.makedirs("indexes", exist_ok=True)
# os.makedirs("embeddings", exist_ok=True)
#
# for cluster in sorted(plot_df['cluster'].unique()):
#     cluster_df = plot_df[plot_df['cluster'] == cluster]
#     texts = cluster_df['text'].tolist()
#     ids = cluster_df['document_id'].tolist()
#
#     embeddings = model.encode(texts, show_progress_bar=True)
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)
#
#     faiss.write_index(index, f"indexes/cluster_{cluster}.index")
#     with open(f"embeddings/cluster_{cluster}.pkl", "wb") as f:
#         pickle.dump({'texts': texts, 'ids': ids}, f)
#
# print("✅ Index preparation complete.")
