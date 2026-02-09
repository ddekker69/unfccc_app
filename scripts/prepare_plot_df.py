# prepare_plot_df.py

import pandas as pd
import numpy as np
import umap.umap_ as umap
import hdbscan
import pickle
import os
from sentence_transformers import SentenceTransformer
from core.pipeline.pipeline_bootstrap import check_folders, check_dependencies
from utils.azure_blob_utils import upload_blob
from core.pipeline.embedding_store import EmbeddingStore
# Azure container
from config import AZURE_CONTAINER_NAME, EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE, OUTPUT_PATH
import streamlit as st

# Step 0: Check requirements
check_folders()
check_dependencies()

# Step 1: Load extracted documents
# with open("data/extracted_texts.pkl", "rb") as f:
#     link_df = pickle.load(f)
#
# # Step 2: Filter OK documents
# link_df = link_df[link_df["status"] == "ok"].reset_index(drop=True)
# # Randomly select 10 rows from link_df
# # sampled_df = link_df.sample(n=10, random_state=42)  # random_state for reproducibility
# #
# # # Remove these rows from the original DataFrame
# # link_df = link_df.drop(sampled_df.index).reset_index(drop=True)
# #
# # # Reset index of the sampled DataFrame (optional)
# # sampled_df = sampled_df.reset_index(drop=True)
# # sampled_df.to_csv("data/sampled_df.csv", index=False)
# # Step 3: Embed texts
# print(f"🤖 Using embedding model: {EMBEDDING_MODEL_NAME}")
# model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=OPTIMAL_DEVICE)
#
#
# embeddings = model.encode(link_df["text"].tolist(), show_progress_bar=True)

# Step 1: load data
# with open("data/extracted_texts.pkl", "rb") as f:
#     link_df = pickle.load(f)
# link_df = link_df[link_df.status == "ok"].reset_index(drop=True)
#
# # NEW: grab vectors from cache instead of re‑encoding
# store = EmbeddingStore(EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE)
# missing = [d for d in link_df.document_id if not store.has_doc(d)]
# if missing:
#     raise RuntimeError(f"{len(missing)} documents miss embeddings – run 01_build_embeddings.py first.")
#
# embeddings = np.vstack([store.doc_vec(d) for d in link_df.document_id])
#
# # Step 4: UMAP dimensionality reduction
# reducer = umap.UMAP()
# reduced_embeddings = reducer.fit_transform(embeddings)
#
# # Step 5: HDBSCAN clustering
# clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
# clusters = clusterer.fit_predict(reduced_embeddings)
#
# # Step 6: Add origin folders
# # Inherit folder info from the extracted data (if available)
# if 'folder' in link_df.columns:
#     folder_column = link_df['folder']
# else:
#     folder_column = ['unknown_folder'] * len(link_df)  # fallback
#
# # Step 6: Construct document-level DataFrame
# plot_df = pd.DataFrame({
#     "document_id": link_df["document_id"],
#     "country": link_df["country"],
#     "title": link_df["title"],
#     "x": reduced_embeddings[:, 0],
#     "y": reduced_embeddings[:, 1],
#     "cluster": clusters,
#     "text": link_df["text"],
#     "status": link_df["status"],
#     "folder": folder_column  # ✅ add folder column here
# })
#
# doc_path = "data/plot_df_full.pkl"
# plot_df.to_pickle(doc_path)
# upload_blob(AZURE_CONTAINER_NAME, "data/plot_df_full.pkl", doc_path)
# print("✅ Document-level plot_df saved and uploaded.")
#
# # Step 7: Aggregate to country-level (mean vector per country)
# print("🔄 Aggregating to country-level...")
#
# link_df["embedding"] = list(embeddings)
# country_groups = link_df.groupby("country")["embedding"].apply(list)
#
# country_texts = []
# country_embeddings = []
# country_names = []
#
# for country, vecs in country_groups.items():
#     if not country or len(vecs) < 1:
#         continue
#     stacked = np.vstack(vecs)
#     avg_vec = stacked.mean(axis=0)
#     full_text = " ".join(link_df[link_df["country"] == country]["text"].tolist())
#
#     country_names.append(country)
#     country_embeddings.append(avg_vec)
#     country_texts.append(full_text)
#
# # UMAP + Clustering at country level
# country_embeddings = np.vstack(country_embeddings)
# reduced = umap.UMAP().fit_transform(country_embeddings)
# clusters = hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(reduced)
#
# country_df = pd.DataFrame({
#     "country": country_names,
#     "combined_text": country_texts,
#     "x": reduced[:, 0],
#     "y": reduced[:, 1],
#     "country_cluster": clusters
# })
#
# country_path = "data/country_plot_df_full.pkl"
# country_df.to_pickle(country_path)
# upload_blob(AZURE_CONTAINER_NAME, "data/country_plot_df_full.pkl", country_path)
# print("✅ Country-level plot_df saved and uploaded.")


def run_clustering_and_save():
    # Load the extracted texts
    with open(OUTPUT_PATH, "rb") as f:
        link_df = pickle.load(f)

    # Filter documents with status "ok"
    link_df = link_df[link_df.status == "ok"].reset_index(drop=True)

    # Embedding store initialization
    store = EmbeddingStore(EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE)

    # Check for missing embeddings
    missing = [d for d in link_df.document_id if not store.has_doc(d)]
    if missing:
        raise RuntimeError(f"{len(missing)} documents miss embeddings – click build embeddings button first.")

    # Get embeddings from the store
    embeddings = np.vstack([store.doc_vec(d) for d in link_df.document_id])

    # Step 4: UMAP dimensionality reduction
    reducer = umap.UMAP()
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Step 5: HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    clusters = clusterer.fit_predict(reduced_embeddings)

    # Step 6: Add origin folders
    folder_column = link_df.get('folder', ['unknown_folder'] * len(link_df))

    # Step 6: Construct document-level DataFrame
    plot_df = pd.DataFrame({
        "document_id": link_df["document_id"],
        "country": link_df["country"],
        "title": link_df["title"],
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "cluster": clusters,
        "text": link_df["text"],
        "status": link_df["status"],
        "folder": folder_column  # Add folder column here
    })

    doc_path = "data/plot_df.pkl"
    plot_df.to_pickle(doc_path)
    st.success("✅ Document-level plot_df saved and uploaded.")

    # Step 7: Aggregate to country-level (mean vector per country)
    link_df["embedding"] = list(embeddings)
    country_groups = link_df.groupby("country")["embedding"].apply(list)

    country_texts = []
    country_embeddings = []
    country_names = []

    for country, vecs in country_groups.items():
        if not country or len(vecs) < 1:
            continue
        stacked = np.vstack(vecs)
        avg_vec = stacked.mean(axis=0)
        full_text = " ".join(link_df[link_df["country"] == country]["text"].tolist())

        country_names.append(country)
        country_embeddings.append(avg_vec)
        country_texts.append(full_text)

    # UMAP + Clustering at country level
    country_embeddings = np.vstack(country_embeddings)
    reduced = umap.UMAP().fit_transform(country_embeddings)
    country_clusters = hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(reduced)

    country_df = pd.DataFrame({
        "country": country_names,
        "combined_text": country_texts,
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "country_cluster": country_clusters
    })

    country_path = "data/country_plot_df.pkl"
    country_df.to_pickle(country_path)
    st.success("✅ Country-level plot_df saved and uploaded.")

    # --- Reporting Changes in Clusters ---
    st.write("### Clustering Summary:")
    st.write("Clusters:", clusters)
    # Number of clusters at the document level
    num_document_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)  # Exclude noise points (-1)
    st.write(f"Number of document-level clusters: {num_document_clusters}")

    # Number of clusters at the country level
    num_country_clusters = len(set(country_clusters)) - (
        1 if -1 in country_clusters else 0)  # Exclude noise points (-1)
    st.write(f"Number of country-level clusters: {num_country_clusters}")

    # Show cluster size distribution for both document and country levels
    st.write("### Document-Level Cluster Sizes:")
    document_cluster_sizes = pd.Series(clusters).value_counts()
    st.write(document_cluster_sizes)

    st.write("### Country-Level Cluster Sizes:")
    country_cluster_sizes = pd.Series(country_clusters).value_counts()
    st.write(country_cluster_sizes)

# # prepare_plot_df.py
#
# import pandas as pd
# import umap.umap_ as umap
# import hdbscan
# from sentence_transformers import SentenceTransformer
# import pickle
# from pipeline_bootstrap import check_folders, check_dependencies, check_dataframe
#
# check_folders()
# check_dependencies()
# check_dataframe(link_df)
#
# # --- Load extracted dataframe ---
# with open("data/extracted_texts.pkl", "rb") as f:
#     link_df = pickle.load(f)
#
# # --- Filter optional (but recommended) ---
# link_df = link_df[link_df['status'] == 'ok'].reset_index(drop=True)
#
# # --- Embedding ---
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(link_df['text'].tolist(), show_progress_bar=True)
#
# # --- UMAP ---
# reducer = umap.UMAP()
# reduced_embeddings = reducer.fit_transform(embeddings)
#
# # --- HDBSCAN ---
# clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
# clusters = clusterer.fit_predict(reduced_embeddings)
#
# # --- Build plot_df ---
# plot_df = pd.DataFrame({
#     'document_id': link_df['document_id'],
#     'country': link_df['country'],
#     'title': link_df['title'],
#     'x': reduced_embeddings[:, 0],
#     'y': reduced_embeddings[:, 1],
#     'cluster': clusters,
#     'text': link_df['text'],
#     'status': link_df['status']  # ✅ Keep it even if all are 'ok' now
# })
#
# plot_df.to_pickle("data/plot_df.pkl")
#
# print("✅ plot_df prepared and saved.")
