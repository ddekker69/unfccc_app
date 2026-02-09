# cluster_documents.py

import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME
import umap.umap_ as umap
import hdbscan
from scripts.prepare_enhanced_index import chunk_text_intelligently
import sys

# Ensure stdout can display emoji on Windows CMD
if sys.stdout.encoding is None or "UTF-8" not in sys.stdout.encoding.upper():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

with open("data/extracted_texts.pkl", "rb") as f:
    texts = pickle.load(f)

df = pd.DataFrame(texts)
df = df.dropna(subset=['country', 'text'])

# Compute document embeddings with sentence-aware chunking
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = []
for text in df['text'].astype(str):
    chunks = chunk_text_intelligently(text, chunk_size=400, overlap=50)
    if chunks:
        chunk_embeds = model.encode(chunks, show_progress_bar=False)
        doc_embeddings.append(np.mean(chunk_embeds, axis=0))
    else:
        doc_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
embeddings = np.vstack(doc_embeddings)

# UMAP
reducer = umap.UMAP(n_neighbors=5, min_dist=0.2, metric='cosine', random_state=42)
reduced_embeddings = reducer.fit_transform(embeddings)

# HDBSCAN Clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
clusters = clusterer.fit_predict(reduced_embeddings)

# Prepare plot_df
df['x'] = reduced_embeddings[:, 0]
df['y'] = reduced_embeddings[:, 1]
df['cluster'] = clusters

df.to_pickle("data/plot_df.pkl")
print("✅ Clustering complete, plot_df.pkl saved.")

from sklearn.cluster import KMeans

def compute_country_level_clustering(df, num_clusters=12, model_name=EMBEDDING_MODEL_NAME):
    model = SentenceTransformer(model_name)

    # Group all text by country
    grouped = df.groupby('country')['text'].apply(lambda texts: ' '.join(texts.dropna())).reset_index()

    # Filter out empty entries
    grouped = grouped[grouped['text'].str.strip().astype(bool)]

    # Encode all-country texts with chunking
    country_embeds = []
    for text in grouped['text'].astype(str):
        chunks = chunk_text_intelligently(text, chunk_size=400, overlap=50)
        if chunks:
            chunk_embeds = model.encode(chunks, show_progress_bar=False)
            country_embeds.append(np.mean(chunk_embeds, axis=0))
        else:
            country_embeds.append(np.zeros(model.get_sentence_embedding_dimension()))
    embeddings = np.vstack(country_embeds)

    # Run KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    grouped['country_cluster'] = kmeans.fit_predict(embeddings)

    return grouped[['country', 'country_cluster']]
