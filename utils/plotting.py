# utils/plotting.py

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import plotly.express as px
def plot_cluster_with_labels(df, output_path="reports/umap_with_labels.png"):
    plt.figure(figsize=(10, 8))
    unique_clusters = sorted(df['cluster'].unique())
    colors = cm.get_cmap('viridis', len(unique_clusters))

    for i, cluster_id in enumerate(unique_clusters):
        cluster_df = df[df['cluster'] == cluster_id]
        plt.scatter(cluster_df['x'], cluster_df['y'], s=40, alpha=0.6, color=colors(i), label=f"Cluster {cluster_id}")

        # Add country labels
        for _, row in cluster_df.iterrows():
            if isinstance(row['country'], str):
                plt.text(row['x'], row['y'], row['country'], fontsize=6, alpha=0.8)

        # Label cluster center
        mean_x = cluster_df['x'].mean()
        mean_y = cluster_df['y'].mean()
        plt.text(mean_x, mean_y, f"Cluster {cluster_id}", fontsize=12, fontweight='bold', color='black')

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP Projection with Cluster and Country Labels")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_cluster_with_hover(df, output_path=None):
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        hover_name='country',
        title="UMAP Clustering (Hover for more info)",
        color_continuous_scale='viridis'
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(
        paper_bgcolor='#F1DFD1',
        plot_bgcolor='#F1DFD1',
        font=dict(size=12),
        title_font=dict(size=18, color='black'),
        hoverlabel=dict(bgcolor="black", font_size=12)
    )
    fig.update_coloraxes(showscale=False)

    if output_path:
        fig.write_image(output_path)
    return fig

import plotly.express as px
import pandas as pd

def plot_cluster_with_hover(df: pd.DataFrame, title="UMAP Cluster View", bg_color="#F1DFD1"):
    # Handle country_cluster to cluster rename if needed
    if "country_cluster" in df.columns and "cluster" not in df.columns:
        df = df.rename(columns={"country_cluster": "cluster"})

    # Ensure required columns exist
    if not all(col in df.columns for col in ["x", "y", "cluster", "country"]):
        raise ValueError("Required columns for plotting not found in dataframe.")

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_data={"country": True, "cluster": True, "x": False, "y": False},
        labels={"cluster": "Cluster"},
        title=title,
        color_continuous_scale='viridis'
    )

    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(size=12, color="black"),
        title=dict(font=dict(size=18, color="black")),
        hoverlabel=dict(bgcolor="black", font_size=12, font_color="white")
    )
    fig.update_coloraxes(showscale=False)


    return fig

def plot_country_cluster_with_hover(df):
    import plotly.express as px

    fig = px.scatter(
        df,
        x="x", y="y",
        color="cluster",
        hover_name="country",
        hover_data=["cluster"],
        title="UMAP Projection of Countries",
        color_continuous_scale="viridis"
    )

    fig.update_layout(
        paper_bgcolor='#F1DFD1',
        plot_bgcolor='#F1DFD1',
        title_font_color="black",
        font=dict(color="black")
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='DarkSlateGrey')))
    return fig
