# utils/reporting_utils.py
import json
import markdown2
import os
import pdfkit
import pandas as pd
from pathlib import Path
from pyvis.network import Network
import streamlit as st

from utils.signed_graph import run_signed_graph_pipeline
from utils.azure_blob_utils import upload_blob
from config import AZURE_CONTAINER_NAME

# === Generate Signed Graph and Upload ===
def generate_signed_graph_visual(df, output_dir="graphs", threshold_high=0.7, threshold_low=0.3, physics_config=None):
    os.makedirs(output_dir, exist_ok=True)
    graph_path = os.path.join(output_dir, "interactive_signed_graph.html")

    sim_df, signed_edges = run_signed_graph_pipeline(df, threshold_high=threshold_high, threshold_low=threshold_low)

    net = visualize_signed_graph_pyvis(signed_edges, output_path=graph_path, physics_config=physics_config)

    # Upload to Azure Blob Storage
    upload_blob(AZURE_CONTAINER_NAME, f"graphs/{os.path.basename(graph_path)}", graph_path)

    return graph_path, signed_edges

# === Markdown to PDF Converter ===
def convert_md_to_pdf_fallback(md_path):
    html_path = md_path.replace('.md', '.html')
    pdf_path = md_path.replace('.md', '.pdf')

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        html_text = markdown2.markdown(md_text)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_text)

        pdfkit.from_file(html_path, pdf_path)

        if os.path.exists(pdf_path):
            return pdf_path
        else:
            print("❌ PDF file was not created despite no exception.")
            return None

    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        return None

# === Signed Graph Visualization with PyVis ===
def visualize_signed_graph_pyvis(
    edge_df: pd.DataFrame,
    output_path: str = "graphs/interactive_signed_graph.html",
    physics_config: dict = None,
    node_attrs: dict | None = None,
) -> Network:
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#F1DFD1",
        font_color="black",
        notebook=False,
        directed=False,
    )

    edge_df['source'] = edge_df['source'].astype(str)
    edge_df['target'] = edge_df['target'].astype(str)
    nodes = set(edge_df['source']).union(set(edge_df['target']))
    node_attrs = node_attrs or {}

    for node in nodes:
        attrs = node_attrs.get(node, {})
        net.add_node(
            node,
            label=attrs.get("label", node),
            title=attrs.get("title"),
            shape="dot",
            size=attrs.get("size", 15),
            font={"size": 16, "color": "black"},
            color=attrs.get("color", "#7FC97F")
        )

    for _, row in edge_df.iterrows():
        if row["sign"] == 0:
            continue
        color = "green" if row["sign"] == 1 else "red"
        net.add_edge(
            row['source'],
            row['target'],
            value=row['similarity'],
            color=color,
        )

    if physics_config:
        net.set_options(json.dumps(physics_config))
    else:
        net.set_options("""{
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.3
            },
            "minVelocity": 0.75
          }
        }""")

    net.write_html(output_path)
    return net

# # utils/reporting_utils.py
# import json
# import markdown2
# from pathlib import Path
# import pdfkit
# import matplotlib.pyplot as plt
# import networkx as nx
# import streamlit as st
# from pyvis.network import Network
# import os
# from utils.similarity_engine import run_signed_graph_pipeline
# from pyvis.network import Network
# import pandas as pd
#
# def generate_signed_graph_visual(df, output_dir="graphs", threshold_high=0.7, threshold_low=0.3, physics_config=None):
#     os.makedirs(output_dir, exist_ok=True)
#     graph_path = os.path.join(output_dir, "interactive_signed_graph.html")
#     sim_df, signed_edges = run_signed_graph_pipeline(df, threshold_high=threshold_high, threshold_low=threshold_low)
#     net = visualize_signed_graph_pyvis(signed_edges, output_path=graph_path, physics_config=physics_config)
#     return graph_path, signed_edges
#
# def convert_md_to_pdf_fallback(md_path):
#     html_path = md_path.replace('.md', '.html')
#     pdf_path = md_path.replace('.md', '.pdf')
#
#     try:
#         with open(md_path, "r", encoding="utf-8") as f:
#             md_text = f.read()
#
#         html_text = markdown2.markdown(md_text)
#         with open(html_path, "w", encoding="utf-8") as f:
#             f.write(html_text)
#
#         pdfkit.from_file(html_path, pdf_path)
#
#         if os.path.exists(pdf_path):
#             return pdf_path
#         else:
#             print("❌ PDF file was not created despite no exception.")
#             return None
#
#     except Exception as e:
#         print(f"❌ PDF generation failed: {e}")
#         return None
#
# # utils/reporting_utils.py
# def visualize_signed_graph_pyvis(
#     edge_df: pd.DataFrame,
#     output_path: str = "graphs/interactive_signed_graph.html",
#     physics_config: dict = None,
# ) -> Network:
#     net = Network(
#         height="600px",
#         width="100%",
#         bgcolor="#F1DFD1",
#         font_color="black",
#         notebook=False,
#         directed=False,
#     )
#
#     edge_df['source'] = edge_df['source'].astype(str)
#     edge_df['target'] = edge_df['target'].astype(str)
#     nodes = set(edge_df['source']).union(set(edge_df['target']))
#
#     for node in nodes:
#         net.add_node(
#             node,
#             label=node,
#             shape="dot",
#             size=15,
#             font={"size": 16, "color": "black"},
#             color="#7FC97F"
#         )
#
#     for _, row in edge_df.iterrows():
#         if row["sign"] == 0:
#             continue
#         color = "green" if row["sign"] == 1 else "red"
#         net.add_edge(
#             row['source'],
#             row['target'],
#             value=row['similarity'],
#             color=color,
#         )
#
#     # Apply layout settings
#     if physics_config:
#         net.set_options(json.dumps(physics_config))
#     else:
#         net.set_options("""
#         {
#           "physics": {
#             "barnesHut": {
#               "gravitationalConstant": -8000,
#               "centralGravity": 0.3,
#               "springLength": 95,
#               "springConstant": 0.04,
#               "damping": 0.09,
#               "avoidOverlap": 0.3
#             },
#             "minVelocity": 0.75
#           }
#         }
#         """)
#
#     net.write_html(output_path)
#     return net
