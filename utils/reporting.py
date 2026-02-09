# reporting.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from utils.signed_graph import (
    compute_cluster_embeddings,
    compute_similarity_matrix,
    compute_signed_edge_list,
)
from utils.reporting_utils import convert_md_to_pdf_fallback, visualize_signed_graph_pyvis
from utils.azure_blob_utils import upload_blob
from config import AZURE_CONTAINER_NAME
import pickle

os.makedirs("reports", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

def generate_cluster_report(cluster_id, df, cluster_summary, umap_path, output_dir="reports"):
    report_md = f"# Cluster {cluster_id} Report\n\n"
    report_md += f"## Summary\n\n{cluster_summary}\n\n"

    report_md += f"## Countries\n"
    countries = df[df['cluster'] == cluster_id]['country'].unique()
    if countries is not None and len(countries) > 0:
        clean_countries = [c for c in countries if isinstance(c, str)]
        report_md += ", ".join(sorted(clean_countries)) + "\n\n"
    else:
        report_md += "No country data available.\n\n"

    report_md += "## Example Documents\n"
    example_titles = df[df['cluster'] == cluster_id]['title'].head(10).tolist()
    for title in example_titles:
        report_md += f"- {title}\n"

    report_md += f"\n## UMAP Visualization\n"
    report_md += f"![UMAP](./{os.path.basename(umap_path)})\n"

    md_path = f"{output_dir}/cluster_{cluster_id}_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    if os.path.exists(umap_path):
        upload_blob(AZURE_CONTAINER_NAME, f"graphs/{os.path.basename(umap_path)}", umap_path)

    return md_path, convert_md_to_pdf_fallback(md_path)

def generate_cross_cluster_report(df, cluster_a, cluster_b, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)

    df_a = df[df['cluster'] == cluster_a]
    df_b = df[df['cluster'] == cluster_b]

    combined_df = pd.concat([df_a, df_b])
    cluster_embeddings = compute_cluster_embeddings(combined_df)
    sim_df = compute_similarity_matrix(cluster_embeddings)
    signed_edges = compute_signed_edge_list(sim_df)
    signed_edges = signed_edges[signed_edges['sign'] != 0]

    edge_list_csv = os.path.join(output_dir, f"signed_edge_list_{cluster_a}_{cluster_b}.csv")
    signed_edges.to_csv(edge_list_csv, index=False)

    md_path = f"{output_dir}/cross_cluster_{cluster_a}_{cluster_b}_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Cross-Cluster Report: Cluster {cluster_a} vs Cluster {cluster_b}\n\n")
        f.write(f"## Document Counts\n")
        f.write(f"- Cluster {cluster_a}: {len(df_a)} documents\n")
        f.write(f"- Cluster {cluster_b}: {len(df_b)} documents\n\n")
        f.write(f"## Countries\n")
        f.write(f"**Cluster {cluster_a}**: {', '.join(sorted(df_a['country'].dropna().unique()))}\n\n")
        f.write(f"**Cluster {cluster_b}**: {', '.join(sorted(df_b['country'].dropna().unique()))}\n\n")
        f.write(f"## Signed Similarity Graph\n")
        f.write(f"- [Download Signed Edge List (CSV)]({os.path.basename(edge_list_csv)})\n")

    pdf_path = convert_md_to_pdf_fallback(md_path)

    html_path = os.path.join(output_dir, f"interactive_graph_{cluster_a}_{cluster_b}.html")
    visualize_signed_graph_pyvis(signed_edges, output_path=html_path)

    upload_blob(AZURE_CONTAINER_NAME, f"graphs/{os.path.basename(html_path)}", html_path)

    return md_path, pdf_path, html_path, edge_list_csv

def export_signed_graph(signed_edges, cluster_a, cluster_b, output_dir="graphs"):
    G = nx.Graph()
    for _, row in signed_edges.iterrows():
        if row['sign'] != 0:
            G.add_edge(row['source'], row['target'], weight=row['similarity'], sign=row['sign'])

    gml_path = f"{output_dir}/signed_graph_{cluster_a}_vs_{cluster_b}.gml"
    nx.write_gml(G, gml_path)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 6))
    colors = ['green' if G[u][v]['sign'] == 1 else 'red' for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, edge_color=colors, node_color='lightblue', node_size=500)
    png_path = f"{output_dir}/signed_graph_{cluster_a}_vs_{cluster_b}.png"
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()

    upload_blob(AZURE_CONTAINER_NAME, f"{output_dir}/{os.path.basename(png_path)}", png_path)
    upload_blob(AZURE_CONTAINER_NAME, f"{output_dir}/{os.path.basename(gml_path)}", gml_path)

    return png_path, gml_path

def plot_cluster(df, cluster_id, output_dir="graphs"):
    cluster_df = df[df['cluster'] == cluster_id]

    plt.figure(figsize=(6, 6))
    plt.scatter(cluster_df['x'], cluster_df['y'], c=cluster_df['cluster'], cmap='viridis', s=30)
    plt.title(f"Cluster {cluster_id} UMAP Projection")
    plt.xlabel("UMAP-Dimension-1")
    plt.ylabel("UMAP-Dimension-2")

    output_path = os.path.join(output_dir, f"cluster_{cluster_id}_umap.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return output_path

# # reporting.py
#
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import networkx as nx
# import streamlit as st
# from utils.similarity_engine import (
#     compute_cluster_embeddings,
#     compute_similarity_matrix,
#     compute_signed_edge_list,
# )
# from utils.reporting_utils import convert_md_to_pdf_fallback, visualize_signed_graph_pyvis
# from pyvis.network import Network
# from utils.azure_blob_utils import upload_blob
# import pickle
#
# os.makedirs("reports", exist_ok=True)
#
# # --- Cluster Report (Single Cluster) ---
# def generate_cluster_report(cluster_id, df, cluster_summary, umap_path, output_dir="reports"):
#     report_md = f"# Cluster {cluster_id} Report\n\n"
#     report_md += f"## Summary\n\n{cluster_summary}\n\n"
#
#     report_md += f"## Countries\n"
#     countries = df[df['cluster'] == cluster_id]['country'].unique()
#     if countries is not None and len(countries) > 0:
#         clean_countries = [c for c in countries if isinstance(c, str)]
#         report_md += ", ".join(sorted(clean_countries)) + "\n\n"
#     else:
#         report_md += "No country data available.\n\n"
#
#     report_md += "## Example Documents\n"
#     example_titles = df[df['cluster'] == cluster_id]['title'].head(10).tolist()
#     for title in example_titles:
#         report_md += f"- {title}\n"
#
#     report_md += f"\n## UMAP Visualization\n"
#     report_md += f"![UMAP](./{os.path.basename(umap_path)})\n"
#
#     md_path = f"{output_dir}/cluster_{cluster_id}_report.md"
#     with open(md_path, "w", encoding="utf-8") as f:
#         f.write(report_md)
#
#     # Upload the UMAP plot to Azure
#     if os.path.exists(umap_path):
#         upload_blob("unfccccontainer", umap_path, f"graphs/{os.path.basename(umap_path)}")
#
#
#     return md_path, convert_md_to_pdf_fallback(md_path)
#
#
# # --- Cross-Cluster Report ---
# def generate_cross_cluster_report(df, cluster_a, cluster_b, output_dir="reports"):
#     os.makedirs(output_dir, exist_ok=True)
#
#     df_a = df[df['cluster'] == cluster_a]
#     df_b = df[df['cluster'] == cluster_b]
#
#     # --- Compute Embeddings & Similarity ---
#     combined_df = pd.concat([df_a, df_b])
#     cluster_embeddings = compute_cluster_embeddings(combined_df)
#     sim_df = compute_similarity_matrix(cluster_embeddings)
#
#     # --- Compute Signed Graph ---
#     signed_edges = compute_signed_edge_list(sim_df)
#     signed_edges = signed_edges[signed_edges['sign'] != 0]  # Remove neutral edges
#
#     # Export edge list CSV
#     edge_list_csv = os.path.join(output_dir, f"signed_edge_list_{cluster_a}_{cluster_b}.csv")
#     signed_edges.to_csv(edge_list_csv, index=False)
#
#     # --- Markdown Report ---
#     md_path = f"{output_dir}/cross_cluster_{cluster_a}_{cluster_b}_report.md"
#     with open(md_path, "w", encoding="utf-8") as f:
#         f.write(f"# Cross-Cluster Report: Cluster {cluster_a} vs Cluster {cluster_b}\n\n")
#         f.write(f"## Document Counts\n")
#         f.write(f"- Cluster {cluster_a}: {len(df_a)} documents\n")
#         f.write(f"- Cluster {cluster_b}: {len(df_b)} documents\n\n")
#
#         f.write(f"## Countries\n")
#         f.write(f"**Cluster {cluster_a}**: {', '.join(sorted(df_a['country'].dropna().unique()))}\n\n")
#         f.write(f"**Cluster {cluster_b}**: {', '.join(sorted(df_b['country'].dropna().unique()))}\n\n")
#
#         f.write(f"## Signed Similarity Graph\n")
#         f.write(f"- [Download Signed Edge List (CSV)]({os.path.basename(edge_list_csv)})\n")
#
#     pdf_path = convert_md_to_pdf_fallback(md_path)
#
#     # --- PyVis Interactive Network Visualization ---
#     html_path = os.path.join(output_dir, f"interactive_graph_{cluster_a}_{cluster_b}.html")
#
#     # Updated version with pre-added nodes to avoid AssertionError
#     from pyvis.network import Network
#
#     net = visualize_signed_graph_pyvis(signed_edges, output_path=html_path)
#     upload_blob("unfccccontainer", html_path, f"graphs/{os.path.basename(html_path)}")
#
#     return md_path, pdf_path, html_path, edge_list_csv
#
#
# # --- Export Signed Graph PNG + GML ---
# def export_signed_graph(signed_edges, cluster_a, cluster_b, output_dir="graphs"):
#     G = nx.Graph()
#     for _, row in signed_edges.iterrows():
#         if row['sign'] != 0:
#             G.add_edge(row['source'], row['target'], weight=row['similarity'], sign=row['sign'])
#
#     # Save GML
#     gml_path = f"{output_dir}/signed_graph_{cluster_a}_vs_{cluster_b}.gml"
#     nx.write_gml(G, gml_path)
#
#     # Save PNG
#     pos = nx.spring_layout(G, seed=42)
#     plt.figure(figsize=(6, 6))
#     colors = ['green' if G[u][v]['sign'] == 1 else 'red' for u, v in G.edges()]
#     nx.draw(G, pos, with_labels=True, edge_color=colors, node_color='lightblue', node_size=500)
#     png_path = f"{output_dir}/signed_graph_{cluster_a}_vs_{cluster_b}.png"
#     plt.savefig(png_path, bbox_inches='tight')
#     plt.close()
#     upload_blob("unfccccontainer", png_path, f"{output_dir}/{os.path.basename(png_path)}")
#     upload_blob("unfccccontainer", gml_path, f"{output_dir}/{os.path.basename(gml_path)}")
#
#     return png_path, gml_path
#
#
# # --- UMAP Plot per Cluster ---
# def plot_cluster(df, cluster_id, output_dir="graphs"):
#     cluster_df = df[df['cluster'] == cluster_id]
#
#     plt.figure(figsize=(6, 6))
#     plt.scatter(cluster_df['x'], cluster_df['y'], c=cluster_df['cluster'], cmap='viridis', s=30)
#     plt.title(f"Cluster {cluster_id} UMAP Projection")
#     plt.xlabel("UMAP-Dimension-1")
#     plt.ylabel("UMAP-Dimension-2")
#
#     output_path = os.path.join(output_dir, f"cluster_{cluster_id}_umap.png")
#     plt.savefig(output_path, bbox_inches='tight')
#     plt.close()
#     return output_path
