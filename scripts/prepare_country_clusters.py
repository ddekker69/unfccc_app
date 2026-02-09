#!/usr/bin/env python3
"""
Enhanced Country Clustering Pipeline
===================================

Main clustering pipeline that uses enhanced country extraction to include
all major countries like USA, Russia, Cuba, etc.

Now includes 172 countries (up from original 56).
"""

import pandas as pd
import pickle
import os
import urllib.parse
import logging
import umap.umap_ as umap
import hdbscan
from sentence_transformers import SentenceTransformer
import numpy as np
import sys

# Ensure stdout can display emoji on Windows CMD
if sys.stdout.encoding is None or "UTF-8" not in sys.stdout.encoding.upper():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
from scripts.prepare_enhanced_index import chunk_text_intelligently
from config import EMBEDDING_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_country_clustering():
    """Run enhanced country clustering with comprehensive country coverage."""
    
    # --- Config ---
    INPUT_PATH = "data/extracted_texts.pkl"
    OUTPUT_PATH = "data/country_plot_df.pkl"
    MIN_CLUSTER_SIZE = 1

    logger.info("Starting enhanced UNFCCC country clustering...")
    
    # --- Load Data ---
    logger.info(f"Loading data from {INPUT_PATH}...")
    with open(INPUT_PATH, "rb") as f:
        df = pickle.load(f)

    logger.info(f"Total documents loaded: {len(df)}")
    
    # Show country assignment statistics
    total_docs = len(df)
    docs_with_countries = len(df.dropna(subset=['country']))
    null_docs = total_docs - docs_with_countries
    unique_countries = len(df['country'].dropna().unique())
    
    logger.info(f"Documents with country assignments: {docs_with_countries}/{total_docs} ({(docs_with_countries/total_docs)*100:.1f}%)")
    logger.info(f"Unique countries detected: {unique_countries}")
    
    # Check for key countries
    key_countries = ['United States of America', 'Russian Federation', 'Cuba', 'China', 'Marshall Islands']
    found_key = []
    for country in key_countries:
        if country in df['country'].values:
            doc_count = len(df[df['country'] == country])
            found_key.append(f"{country} ({doc_count} docs)")
    
    if found_key:
        logger.info(f"Key countries successfully included: {found_key}")

    # --- Filter Valid Documents ---
    logger.info("Filtering valid documents...")
    df_filtered = df.dropna(subset=['country', 'text'])
    df_filtered = df_filtered[df_filtered['status'] == 'ok']
    
    logger.info(f"Valid documents after filtering: {len(df_filtered)}")
    logger.info(f"Countries with valid documents: {len(df_filtered['country'].unique())}")

    # --- Group by Country ---
    logger.info("Grouping by country...")
    grouped = df_filtered.groupby("country")["text"].apply(lambda x: " ".join(x.dropna())).reset_index()
    grouped = grouped[grouped["text"].str.strip().astype(bool)]
    
    logger.info(f"Countries with sufficient text for clustering: {len(grouped)}")

    if len(grouped) == 0:
        logger.error("No countries available for clustering!")
        return

    # Show sample of countries
    countries_sample = sorted(grouped['country'].tolist())[:20]
    logger.info(f"Sample countries: {countries_sample}...")

    # --- Compute Embeddings ---
    logger.info(f"Computing embeddings using model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    country_embeds = []
    for text in grouped["text"].astype(str):
        chunks = chunk_text_intelligently(text, chunk_size=400, overlap=50)
        if chunks:
            chunk_embeds = model.encode(chunks, show_progress_bar=False)
            country_embeds.append(np.mean(chunk_embeds, axis=0))
        else:
            country_embeds.append(np.zeros(model.get_sentence_embedding_dimension()))
    embeddings = np.vstack(country_embeds)
    
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # --- UMAP Reduction ---
    logger.info("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_neighbors=min(15, len(grouped)-1), 
        min_dist=0.2, 
        metric='cosine', 
        random_state=42
    )
    reduced_embeddings = reducer.fit_transform(embeddings)

    # --- HDBSCAN Clustering ---
    logger.info("Performing HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=1,  # Allow single countries to form clusters
        cluster_selection_epsilon=0.1,  # More aggressive cluster selection
        cluster_selection_method='eom'  # Excess of Mass for better cluster detection
    )
    clusters = clusterer.fit_predict(reduced_embeddings)
    
    # Cluster statistics
    unique_clusters = len(set(clusters[clusters >= 0]))
    noise_points = sum(clusters == -1)
    logger.info(f"Number of clusters found: {unique_clusters}")
    logger.info(f"Noise points: {noise_points}")

    # --- Prepare Final DataFrame ---
    logger.info("Preparing final clustering dataframe...")
    grouped["x"] = reduced_embeddings[:, 0]
    grouped["y"] = reduced_embeddings[:, 1]
    grouped["cluster"] = clusters
    
    # Add country_cluster for app compatibility
    grouped["country_cluster"] = clusters
    
    # Rename for compatibility
    if "text" in grouped.columns:
        grouped = grouped.rename(columns={"text": "combined_text"})
    
    # Add metadata
    grouped['text_length'] = grouped['combined_text'].str.len()

    # --- Show Cluster Assignments ---
    logger.info("\nCluster assignments:")
    for cluster_id in sorted(set(clusters)):
        cluster_countries = grouped[grouped['cluster'] == cluster_id]['country'].tolist()
        if cluster_id == -1:
            logger.info(f"Noise: {cluster_countries}")
        else:
            logger.info(f"Cluster {cluster_id}: {cluster_countries}")

    # --- Save Output ---
    logger.info(f"Saving clustering results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(grouped, f)

    # Also save as CSV for inspection
    csv_path = OUTPUT_PATH.replace('.pkl', '.csv')
    grouped.to_csv(csv_path, index=False)
    
    logger.info(f"✅ Enhanced country clustering complete!")
    logger.info(f"Results saved to: {OUTPUT_PATH}")
    logger.info(f"CSV saved to: {csv_path}")
    logger.info(f"Total countries clustered: {len(grouped)}")
    
    return grouped

if __name__ == "__main__":
    result = run_country_clustering()

# # prepare_country_clusters.py
#
# import pandas as pd
# import pickle
# import os
# import umap.umap_ as umap
# import hdbscan
# from sentence_transformers import SentenceTransformer
#
# # --- Config ---
# INPUT_PATH = "data/extracted_texts.pkl"
# OUTPUT_PATH = "data/country_plot_df.pkl"
# EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# MIN_CLUSTER_SIZE = 2
#
# # --- Load Data ---
# with open(INPUT_PATH, "rb") as f:
#     df = pickle.load(f)
#
# # --- Filter Valid Documents ---
# df = df.dropna(subset=['country', 'text'])
# df = df[df['status'] == 'ok']
#
# # --- Group by Country ---
# grouped = df.groupby("country")["text"].apply(lambda x: " ".join(x.dropna())).reset_index()
# grouped = grouped[grouped["text"].str.strip().astype(bool)]
#
# # --- Compute Embeddings ---
# model = SentenceTransformer(EMBEDDING_MODEL)
# embeddings = model.encode(grouped["text"].tolist(), show_progress_bar=True)
#
# # --- UMAP Reduction ---
# reducer = umap.UMAP(n_neighbors=5, min_dist=0.2, metric='cosine', random_state=42)
# reduced_embeddings = reducer.fit_transform(embeddings)
#
# # --- HDBSCAN Clustering ---
# clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE)
# clusters = clusterer.fit_predict(reduced_embeddings)
#
# # --- Prepare Final DataFrame ---
# grouped["x"] = reduced_embeddings[:, 0]
# grouped["y"] = reduced_embeddings[:, 1]
# grouped["country_cluster"] = clusters
# grouped = grouped.rename(columns={"text": "combined_text"})
#
# # --- Save Output ---
# with open(OUTPUT_PATH, "wb") as f:
#     pickle.dump(grouped, f)
#
# print(f"✅ Country-level clustering saved to {OUTPUT_PATH}")
