#!/usr/bin/env python3
"""Build FAISS indexes from `data/plot_df.pkl` for cluster-level retrieval."""

import gc
import os
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AZURE_CONTAINER_NAME, EMBEDDING_MODEL_NAME, OPTIMAL_DEVICE
from core.pipeline.pipeline_bootstrap import check_dependencies, check_folders
from utils.azure_blob_utils import upload_blob


def build_indexes(plot_df_path: str = "data/plot_df.pkl") -> None:
    check_folders()
    check_dependencies()

    print(f"🤖 Using embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=OPTIMAL_DEVICE)
    print(
        f"✅ Model loaded on {model.device} "
        f"with {model.get_sentence_embedding_dimension()} dimensions"
    )

    plot_df = pd.read_pickle(plot_df_path)
    print(f"📊 Loaded {len(plot_df)} documents across {plot_df['cluster'].nunique()} clusters")

    os.makedirs("indexes", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)

    for cluster in sorted(plot_df["cluster"].unique()):
        print(f"\n🔧 Processing cluster {cluster}...")
        cluster_df = plot_df[plot_df["cluster"] == cluster]
        texts = cluster_df["text"].tolist()
        ids = cluster_df["document_id"].tolist()
        publication_years = (
            cluster_df["publication_year"].tolist()
            if "publication_year" in cluster_df.columns
            else [None] * len(texts)
        )

        if not texts:
            print(f"   ⚠️ Skipping empty cluster {cluster}")
            continue

        print(f"   🧮 Generating embeddings for {len(texts)} documents...")
        embeddings = model.encode(texts, batch_size=1, show_progress_bar=False)

        print("   🔍 Creating FAISS index...")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))

        index_path = f"indexes/cluster_{cluster}.index"
        emb_path = f"embeddings/cluster_{cluster}.pkl"

        faiss.write_index(index, index_path)
        with open(emb_path, "wb") as f:
            pickle.dump({"texts": texts, "ids": ids, "publication_years": publication_years}, f)

        print(f"   ✅ Saved index + metadata for cluster {cluster}")

        try:
            upload_blob(AZURE_CONTAINER_NAME, f"indexes/cluster_{cluster}.index", index_path)
            upload_blob(AZURE_CONTAINER_NAME, f"embeddings/cluster_{cluster}.pkl", emb_path)
        except Exception as exc:
            print(f"   ⚠️ Azure upload failed (continuing anyway): {exc}")

        del embeddings, index
        gc.collect()

    print("\n🎉 Index preparation complete!")


if __name__ == "__main__":
    build_indexes()
