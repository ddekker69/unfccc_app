#!/usr/bin/env python3
"""Create document-level and country-level clustering dataframes."""

import pickle
import os
import sys
from pathlib import Path

# Avoid numba cache failures when importing UMAP in constrained environments.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import hdbscan
import numpy as np
import pandas as pd
import umap.umap_ as umap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE, OUTPUT_PATH
from core.pipeline.embedding_store import EmbeddingStore
from core.pipeline.pipeline_bootstrap import check_dependencies, check_folders


def run_clustering_and_save() -> None:
    check_folders()
    check_dependencies()

    with open(OUTPUT_PATH, "rb") as f:
        link_df = pickle.load(f)

    link_df = link_df[link_df.status == "ok"].reset_index(drop=True)

    store = EmbeddingStore(EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE)
    missing = [doc_id for doc_id in link_df.document_id if not store.has_doc(doc_id)]
    if missing:
        raise RuntimeError(
            f"{len(missing)} documents miss embeddings - click build embeddings button first."
        )

    embeddings = np.vstack([store.doc_vec(doc_id) for doc_id in link_df.document_id])

    reducer = umap.UMAP()
    reduced_embeddings = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    clusters = clusterer.fit_predict(reduced_embeddings)

    folder_column = link_df.get("folder", ["unknown_folder"] * len(link_df))

    plot_df = pd.DataFrame(
        {
            "document_id": link_df["document_id"],
            "country": link_df["country"],
            "title": link_df["title"],
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "cluster": clusters,
            "text": link_df["text"],
            "status": link_df["status"],
            "folder": folder_column,
        }
    )

    plot_df.to_pickle("data/plot_df.pkl")
    print("✅ Document-level plot_df saved.")

    link_df["embedding"] = list(embeddings)
    country_groups = link_df.groupby("country")["embedding"].apply(list)

    country_texts = []
    country_embeddings = []
    country_names = []

    for country, vectors in country_groups.items():
        if not country or len(vectors) < 1:
            continue
        stacked = np.vstack(vectors)
        avg_vec = stacked.mean(axis=0)
        full_text = " ".join(link_df[link_df["country"] == country]["text"].tolist())

        country_names.append(country)
        country_embeddings.append(avg_vec)
        country_texts.append(full_text)

    country_embeddings = np.vstack(country_embeddings)
    reduced_country = umap.UMAP().fit_transform(country_embeddings)
    country_clusters = hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(reduced_country)

    country_df = pd.DataFrame(
        {
            "country": country_names,
            "combined_text": country_texts,
            "x": reduced_country[:, 0],
            "y": reduced_country[:, 1],
            "country_cluster": country_clusters,
        }
    )

    country_df.to_pickle("data/country_plot_df.pkl")
    print("✅ Country-level plot_df saved.")


if __name__ == "__main__":
    run_clustering_and_save()
