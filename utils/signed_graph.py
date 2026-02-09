# utils/signed_graph.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import networkx as nx
import logging
from config import EMBEDDING_MODEL_NAME

# Configure logging
logger = logging.getLogger(__name__)

# === Configuration ===
# Use the model name from config
# MODEL_NAME = 'all-MiniLM-L6-v2'  # Now imported from config
os.makedirs('data', exist_ok=True)  # Ensure output folder exists

# --- Load Model ---
def load_sentence_transformer_offline(model_name: str) -> SentenceTransformer:
    """
    Load SentenceTransformer model with offline-first approach.
    
    Args:
        model_name: Name or path of the model to load
        
    Returns:
        Loaded SentenceTransformer model
        
    Raises:
        ConnectionError: If model cannot be loaded offline and no internet connection
        RuntimeError: If model loading fails for other reasons
    """
    logger.debug(f"Loading SentenceTransformer model: {model_name}")
    
    try:
        # First try to load offline by setting HF_HUB_OFFLINE environment variable
        logger.info("Attempting to load model from local cache (offline mode)")
        
        # Store original value of HF_HUB_OFFLINE
        original_offline = os.environ.get('HF_HUB_OFFLINE')
        
        try:
            # Force offline mode
            os.environ['HF_HUB_OFFLINE'] = '1'
            model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded model {model_name} from local cache")
            return model
            
        finally:
            # Restore original environment variable
            if original_offline is None:
                os.environ.pop('HF_HUB_OFFLINE', None)
            else:
                os.environ['HF_HUB_OFFLINE'] = original_offline
        
    except Exception as offline_error:
        logger.warning(f"Failed to load model offline: {offline_error}")
        logger.info("Attempting to download model (requires internet connection)")
        
        try:
            # If offline fails, try to download (requires internet)
            # Make sure HF_HUB_OFFLINE is not set to force online mode
            original_offline = os.environ.get('HF_HUB_OFFLINE')
            try:
                os.environ.pop('HF_HUB_OFFLINE', None)
                model = SentenceTransformer(model_name)
                logger.info(f"Successfully downloaded and loaded model {model_name}")
                return model
            finally:
                # Restore original environment variable
                if original_offline is not None:
                    os.environ['HF_HUB_OFFLINE'] = original_offline
            
        except Exception as online_error:
            logger.error(f"Failed to load model both offline and online")
            logger.error(f"Offline error: {offline_error}")
            logger.error(f"Online error: {online_error}")
            
            raise ConnectionError(
                f"Cannot load model '{model_name}'. "
                f"Model not found in local cache and internet connection unavailable. "
                f"Please ensure internet connection to download the model initially, "
                f"or check if the model name is correct."
            ) from online_error

# Initialize model with offline-first loading
model = load_sentence_transformer_offline(EMBEDDING_MODEL_NAME)

# --- Embedding Generator ---
def compute_cluster_embeddings(df):
    """
    Compute embeddings for each cluster using the mean of document embeddings.
    
    Args:
        df: DataFrame with 'cluster' and 'text' columns
        
    Returns:
        Dictionary mapping cluster_id to mean embedding vector
        
    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    logger.debug(f"Computing cluster embeddings for {len(df)} documents")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if 'cluster' not in df.columns or 'text' not in df.columns:
        raise ValueError("DataFrame must contain 'cluster' and 'text' columns")
    
    cluster_embeddings = {}
    unique_clusters = sorted(df['cluster'].unique())
    logger.info(f"Processing {len(unique_clusters)} clusters")
    
    for cluster_id in unique_clusters:
        texts = df[df['cluster'] == cluster_id]['text'].dropna().tolist()
        
        if len(texts) < 2:
            logger.warning(f"Cluster {cluster_id} has less than 2 documents. Skipped.")
            continue
            
        logger.debug(f"Encoding {len(texts)} texts for cluster {cluster_id}")
        embeddings = model.encode(texts, show_progress_bar=True)
        mean_embedding = np.mean(embeddings, axis=0)
        cluster_embeddings[cluster_id] = mean_embedding
        
    logger.info(f"Successfully computed embeddings for {len(cluster_embeddings)} clusters")
    return cluster_embeddings

# --- Similarity Matrix ---
def compute_similarity_matrix(cluster_embeddings):
    """
    Compute cosine similarity matrix between cluster embeddings.
    
    Args:
        cluster_embeddings: Dictionary mapping cluster_id to embedding vector
        
    Returns:
        DataFrame with similarity matrix (cluster_ids as index and columns)
        
    Raises:
        ValueError: If no valid cluster embeddings provided
    """
    logger.debug(f"Computing similarity matrix for {len(cluster_embeddings)} clusters")
    
    if not cluster_embeddings:
        raise ValueError("❌ No valid cluster embeddings computed.")
        
    cluster_ids = list(cluster_embeddings.keys())
    vectors = np.stack([cluster_embeddings[c] for c in cluster_ids])
    
    logger.debug("Computing cosine similarity matrix")
    sim_matrix = cosine_similarity(vectors)
    sim_df = pd.DataFrame(sim_matrix, index=cluster_ids, columns=cluster_ids)
    
    logger.info(f"Successfully computed {sim_df.shape[0]}x{sim_df.shape[1]} similarity matrix")
    return sim_df

# --- Signed Edge List ---
def compute_signed_edge_list(similarity_df, threshold_high=0.7, threshold_low=0.3):
    """
    Convert similarity matrix to signed edge list based on thresholds.
    
    Args:
        similarity_df: DataFrame with similarity values
        threshold_high: Threshold for positive edges (default: 0.7)
        threshold_low: Threshold for negative edges (default: 0.3)
        
    Returns:
        DataFrame with columns: source, target, similarity, sign
    """
    logger.debug(f"Computing signed edge list with thresholds: high={threshold_high}, low={threshold_low}")
    
    edges = []
    for i in similarity_df.index:
        for j in similarity_df.columns:
            if i < j:  # Avoid duplicate edges
                sim = similarity_df.loc[i, j]
                if sim >= threshold_high:
                    sign = 1
                elif sim <= threshold_low:
                    sign = -1
                else:
                    sign = 0
                edges.append({'source': i, 'target': j, 'similarity': sim, 'sign': sign})
    
    edge_df = pd.DataFrame(edges)
    positive_edges = len(edge_df[edge_df['sign'] == 1])
    negative_edges = len(edge_df[edge_df['sign'] == -1])
    neutral_edges = len(edge_df[edge_df['sign'] == 0])
    
    logger.info(f"Generated {len(edge_df)} edges: {positive_edges} positive, {negative_edges} negative, {neutral_edges} neutral")
    return edge_df

# --- Save ---
def save_signed_edge_list(edge_df, output_path="data/signed_edge_list.pkl"):
    """
    Save signed edge list to pickle file.
    
    Args:
        edge_df: DataFrame with edge data
        output_path: Path to save the pickle file (default: "data/signed_edge_list.pkl")
        
    Returns:
        The input DataFrame (for chaining)
    """
    logger.debug(f"Saving edge list to {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(edge_df, f)
        
    logger.info(f"✅ Signed Edge List saved to {output_path}")
    return edge_df

# --- Full Pipeline ---
def run_signed_graph_pipeline(df, threshold_high=0.7, threshold_low=0.3):
    """
    Run the complete signed graph pipeline.
    
    Args:
        df: Input DataFrame with 'cluster' and 'text' columns
        threshold_high: Threshold for positive edges (default: 0.7)
        threshold_low: Threshold for negative edges (default: 0.3)
        
    Returns:
        Tuple of (similarity_df, signed_edges_df)
    """
    logger.info("📌 Starting Signed-Graph Pipeline")
    
    try:
        cluster_embeddings = compute_cluster_embeddings(df)
        sim_df = compute_similarity_matrix(cluster_embeddings)
        signed_edges = compute_signed_edge_list(sim_df, threshold_high=threshold_high, threshold_low=threshold_low)
        save_signed_edge_list(signed_edges)
        
        logger.info("✅ Signed-Graph Pipeline completed successfully")
        return sim_df, signed_edges
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

# --- NetworkX Export ---
def export_graph(edge_df, output_path="data/signed_graph.gml"):
    """
    Export signed graph to NetworkX GML format.
    
    Args:
        edge_df: DataFrame with edge data
        output_path: Path to save the GML file (default: "data/signed_graph.gml")
    """
    logger.debug(f"Exporting graph to {output_path}")
    
    G = nx.Graph()
    non_neutral_edges = 0
    
    for _, row in edge_df.iterrows():
        if row['sign'] != 0:  # exclude neutral edges
            G.add_edge(row['source'], row['target'], weight=row['similarity'], sign=row['sign'])
            non_neutral_edges += 1
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nx.write_gml(G, output_path)
    
    logger.info(f"✅ Graph with {non_neutral_edges} non-neutral edges saved to {output_path}")

# --- Country-specific Embedding Generator ---
def compute_country_embeddings(df: pd.DataFrame, text_column: str = "combined_text", id_column: str = "country") -> dict:
    """
    Compute embeddings for each country using their combined text.
    
    Args:
        df: DataFrame with country data
        text_column: Column name containing the combined text (default: "combined_text")
        id_column: Column name containing the country identifier (default: "country")
        
    Returns:
        Dictionary mapping country to embedding vector
        
    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    logger.debug(f"Computing country embeddings for {len(df)} rows")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if id_column not in df.columns or text_column not in df.columns:
        raise ValueError(f"DataFrame must contain '{id_column}' and '{text_column}' columns")
    
    country_embeddings = {}
    unique_countries = df[id_column].unique()
    logger.info(f"Processing {len(unique_countries)} countries")
    
    for country in unique_countries:
        text = df[df[id_column] == country][text_column].values[0]
        
        if not isinstance(text, str) or not text.strip():
            logger.warning(f"Country {country} has no valid text. Skipped.")
            continue
            
        logger.debug(f"Encoding text for country: {country}")
        embedding = model.encode([text])[0]
        country_embeddings[country] = embedding
        
    logger.info(f"Successfully computed embeddings for {len(country_embeddings)} countries")
    return country_embeddings

# --- Country-specific Pipeline ---
def run_country_signed_graph_pipeline(df: pd.DataFrame, threshold_high: float = 0.7, threshold_low: float = 0.3):
    """
    Run the complete signed graph pipeline for country-level analysis.
    
    Args:
        df: Input DataFrame with country data
        threshold_high: Threshold for positive edges (default: 0.7)
        threshold_low: Threshold for negative edges (default: 0.3)
        
    Returns:
        Tuple of (similarity_df, signed_edges_df)
    """
    logger.info("📌 Starting Country-Level Signed-Graph Pipeline")
    
    try:
        country_embeddings = compute_country_embeddings(df)
        sim_df = compute_similarity_matrix(country_embeddings)
        signed_edges = compute_signed_edge_list(sim_df, threshold_high=threshold_high, threshold_low=threshold_low)
        
        # Save with country-specific filenames
        save_signed_edge_list(signed_edges, output_path="data/signed_edge_list_countries.pkl")
        export_graph(signed_edges, output_path="data/signed_graph_countries.gml")
        
        logger.info("✅ Country-Level Signed-Graph Pipeline completed successfully")
        return sim_df, signed_edges
        
    except Exception as e:
        logger.error(f"Country pipeline failed: {str(e)}", exc_info=True)
        raise

# --- Legacy Compatibility Functions (for backwards compatibility) ---
def compute_edge_list(similarity_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Compute simple edge list from similarity matrix (legacy compatibility function).
    
    Args:
        similarity_df: DataFrame with similarity values
        threshold: Minimum similarity threshold for edges (default: 0.5)
        
    Returns:
        DataFrame with columns: source, target, weight
    """
    logger.debug(f"Computing edge list with threshold={threshold}")
    
    edges = []
    for i in similarity_df.index:
        for j in similarity_df.columns:
            if i < j and similarity_df.loc[i, j] >= threshold:
                edges.append({'source': i, 'target': j, 'weight': similarity_df.loc[i, j]})
    
    edge_df = pd.DataFrame(edges)
    logger.info(f"Generated {len(edge_df)} edges with threshold >= {threshold}")
    return edge_df

def save_similarity_outputs(similarity_df: pd.DataFrame, edge_list: pd.DataFrame, output_prefix: str = "data/cluster_similarity"):
    """
    Save similarity matrix and edge list (legacy compatibility function).
    
    Args:
        similarity_df: DataFrame with similarity matrix
        edge_list: DataFrame with edge data
        output_prefix: Output file prefix (default: "data/cluster_similarity")
    """
    logger.debug(f"Saving similarity outputs with prefix: {output_prefix}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Save similarity matrix as CSV
    csv_path = f"{output_prefix}_matrix.csv"
    similarity_df.to_csv(csv_path)
    logger.info(f"Similarity matrix saved to {csv_path}")
    
    # Save edge list as pickle
    pkl_path = f"{output_prefix}_edges.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(edge_list, f)
    logger.info(f"Edge list saved to {pkl_path}")

def run_similarity_pipeline(df: pd.DataFrame, threshold: float = 0.5):
    """
    Run the basic similarity pipeline (legacy compatibility function).
    
    Args:
        df: Input DataFrame with 'cluster' and 'text' columns
        threshold: Minimum similarity threshold for edges (default: 0.5)
        
    Returns:
        Tuple of (similarity_df, edge_list)
    """
    logger.info("📌 Starting Legacy Similarity Pipeline")
    
    try:
        cluster_embeddings = compute_cluster_embeddings(df)
        sim_df = compute_similarity_matrix(cluster_embeddings)
        edge_list = compute_edge_list(sim_df, threshold=threshold)
        save_similarity_outputs(sim_df, edge_list)
        
        logger.info("✅ Legacy Similarity Pipeline completed successfully")
        return sim_df, edge_list
        
    except Exception as e:
        logger.error(f"Legacy pipeline failed: {str(e)}", exc_info=True)
        raise

