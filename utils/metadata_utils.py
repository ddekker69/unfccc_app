# utils/metadata_utils.py
import re

def get_cluster_countries(cluster_id, df):
    return sorted(df[df['cluster'] == cluster_id]['country'].dropna().unique().tolist())

def get_cluster_doc_count(cluster_id, df):
    return len(df[df['cluster'] == cluster_id])

def get_largest_cluster(df):
    counts = df['cluster'].value_counts()
    return counts.idxmax(), counts.max()

def get_cluster_summary(cluster_id, df):
    cluster_docs = df[df['cluster'] == cluster_id]
    return {
        "countries": sorted(cluster_docs['country'].dropna().unique().tolist()),
        "document_count": len(cluster_docs),
        "example_titles": cluster_docs['title'].head(5).tolist()
    }

def get_cluster_id_from_question(question, df):
    if "largest cluster" in question.lower():
        return get_largest_cluster(df)[0]
    elif "cluster" in question.lower():
        # Extract cluster number if mentioned
        import re
        match = re.search(r'cluster\s*(\d+)', question.lower())
        if match:
            return int(match.group(1))
    return None

def extract_clusters_from_question(question):
    import re
    clusters = [int(c) for c in re.findall(r'cluster\s*(\d+)', question.lower())]
    return list(set(clusters))  # remove duplicates

def get_two_clusters(question, df):
    """
    Extract two cluster numbers from the question.
    """
    clusters = sorted(df['cluster'].unique())
    # Match only whole numbers (\b ensures word boundaries)
    found = re.findall(r'\bcluster\s+(\d+)\b', question.lower())
    found_ids = []

    for num in found:
        try:
            num_int = int(num)
            if num_int in clusters:
                found_ids.append(num_int)
        except ValueError:
            continue

    # Remove duplicates and ensure at least 2 clusters detected
    return list(set(found_ids)) if len(found_ids) >= 2 else None

def extract_cluster_id(question, df):
    clusters = sorted(df['cluster'].unique())
    found = re.findall(r'\b\d+\b', question)
    for num in found:
        num_int = int(num)
        if num_int in clusters:
            return num_int
    return None