"""
Plotly visualization functions for customer segmentation
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List


def create_distribution_plots(df: pd.DataFrame) -> go.Figure:
    """Create distribution plots for numerical features"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Age Distribution",
            "Annual Income Distribution",
            "Spending Score Distribution",
            "Gender Distribution",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "histogram"}],
            [{"type": "histogram"}, {"type": "bar"}],
        ],
    )

    # Age distribution
    fig.add_trace(
        go.Histogram(x=df["Age"], name="Age", marker_color="lightblue", nbinsx=30),
        row=1,
        col=1,
    )

    # Income distribution
    fig.add_trace(
        go.Histogram(
            x=df["Annual Income (k$)"],
            name="Income",
            marker_color="lightgreen",
            nbinsx=30,
        ),
        row=1,
        col=2,
    )

    # Spending Score distribution
    fig.add_trace(
        go.Histogram(
            x=df["Spending Score (1-100)"],
            name="Spending Score",
            marker_color="lightcoral",
            nbinsx=30,
        ),
        row=2,
        col=1,
    )

    # Gender distribution
    if "Gender" in df.columns:
        gender_counts = df["Gender"].value_counts()
        fig.add_trace(
            go.Bar(
                x=gender_counts.index,
                y=gender_counts.values,
                name="Gender",
                marker_color="lightsalmon",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=800, showlegend=False, title_text="Data Distribution Analysis"
    )
    return fig


def create_elbow_plots(elbow_data: Dict) -> go.Figure:
    """Create elbow method plots with multiple metrics"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Elbow Method (WCSS)",
            "Silhouette Score",
            "Calinski-Harabasz Score",
            "Davies-Bouldin Index",
        ),
    )

    k_range = elbow_data["k_range"]

    # WCSS
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=elbow_data["wcss"],
            mode="lines+markers",
            marker=dict(size=10, color="blue"),
            line=dict(width=2),
            name="WCSS",
        ),
        row=1,
        col=1,
    )

    # Silhouette Score
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=elbow_data["silhouette_scores"],
            mode="lines+markers",
            marker=dict(size=10, color="red"),
            line=dict(width=2),
            name="Silhouette",
        ),
        row=1,
        col=2,
    )

    # Calinski-Harabasz Score
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=elbow_data["ch_scores"],
            mode="lines+markers",
            marker=dict(size=10, color="green"),
            line=dict(width=2),
            name="Calinski-Harabasz",
        ),
        row=2,
        col=1,
    )

    # Davies-Bouldin Index
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=elbow_data["db_scores"],
            mode="lines+markers",
            marker=dict(size=10, color="purple"),
            line=dict(width=2),
            name="Davies-Bouldin",
        ),
        row=2,
        col=2,
    )

    # Update axes
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=2, col=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=2, col=2)

    fig.update_yaxes(title_text="WCSS", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    fig.update_yaxes(title_text="CH Score", row=2, col=1)
    fig.update_yaxes(title_text="DB Index (lower is better)", row=2, col=2)

    # Add optimal k annotations
    optimal_sil = elbow_data.get("optimal_k_silhouette", 5)
    fig.add_annotation(
        x=optimal_sil,
        y=elbow_data["silhouette_scores"][k_range.index(optimal_sil)],
        text=f"Optimal: {optimal_sil}",
        showarrow=True,
        arrowhead=2,
        row=1,
        col=2,
    )

    fig.update_layout(
        height=800, showlegend=False, title_text="Optimal Cluster Selection"
    )
    return fig


def create_2d_cluster_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str = "Cluster",
    centroids: np.ndarray = None,
) -> go.Figure:
    """Create 2D scatter plot with clusters"""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=cluster_col,
        title=f"Customer Segments: {x_col} vs {y_col}",
        labels={cluster_col: "Cluster"},
        color_continuous_scale="Viridis",
        hover_data=["Age", "Annual Income (k$)", "Spending Score (1-100)"],
    )

    # Add centroids if provided
    if centroids is not None:
        # Assuming centroids are in order: Income, Spending, Age
        if x_col == "Annual Income (k$)" and y_col == "Spending Score (1-100)":
            centroid_x = centroids[:, 0]
            centroid_y = centroids[:, 1]
        elif x_col == "Annual Income (k$)" and y_col == "Age":
            centroid_x = centroids[:, 0]
            centroid_y = centroids[:, 2]
        elif x_col == "Spending Score (1-100)" and y_col == "Age":
            centroid_x = centroids[:, 1]
            centroid_y = centroids[:, 2]
        else:
            centroid_x = None
            centroid_y = None

        if centroid_x is not None:
            fig.add_trace(
                go.Scatter(
                    x=centroid_x,
                    y=centroid_y,
                    mode="markers",
                    marker=dict(
                        size=20,
                        symbol="x",
                        color="red",
                        line=dict(width=2, color="black"),
                    ),
                    name="Centroids",
                    showlegend=True,
                )
            )

    fig.update_layout(height=600, width=900)
    return fig


def create_3d_cluster_plot(df: pd.DataFrame, cluster_col: str = "Cluster") -> go.Figure:
    """Create 3D scatter plot of clusters"""
    fig = px.scatter_3d(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        z="Age",
        color=cluster_col,
        title="3D Customer Segmentation",
        labels={cluster_col: "Cluster"},
        color_continuous_scale="Viridis",
        hover_data=["CustomerID"],
    )

    fig.update_layout(height=700, width=1000)
    return fig


def create_pca_plot(pca_data: Dict, labels: List[int]) -> go.Figure:
    """Create PCA visualization"""
    pca_components = np.array(pca_data["components"])
    explained_var = pca_data["explained_variance_ratio"]

    df_pca = pd.DataFrame(
        {"PC1": pca_components[:, 0], "PC2": pca_components[:, 1], "Cluster": labels}
    )

    fig = px.scatter(
        df_pca,
        x="PC1",
        y="PC2",
        color="Cluster",
        title=f"PCA Visualization (Explained Variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%})",
        labels={"Cluster": "Cluster"},
        color_continuous_scale="Viridis",
    )

    fig.update_layout(height=600, width=900)
    return fig


def create_silhouette_plot(silhouette_data: Dict) -> go.Figure:
    """Create silhouette plot for cluster quality"""
    fig = go.Figure()

    y_lower = 10
    cluster_silhouettes = silhouette_data["cluster_silhouettes"]

    colors = px.colors.qualitative.Plotly

    for cluster_id in sorted(cluster_silhouettes.keys()):
        cluster_vals = cluster_silhouettes[cluster_id]
        y_upper = y_lower + len(cluster_vals)

        color = colors[cluster_id % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=cluster_vals,
                y=list(range(y_lower, y_upper)),
                fill="tozerox",
                name=f"Cluster {cluster_id}",
                line=dict(color=color),
                mode="lines",
            )
        )

        y_lower = y_upper + 10

    # Add average silhouette score line
    avg_score = silhouette_data["avg_silhouette"]
    fig.add_vline(
        x=avg_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {avg_score:.3f}",
    )

    fig.update_layout(
        title="Silhouette Analysis",
        xaxis_title="Silhouette Coefficient",
        yaxis_title="Cluster",
        height=600,
        width=900,
        showlegend=True,
    )

    return fig


def create_cluster_characteristics_plot(characteristics: Dict) -> go.Figure:
    """Create plots for cluster characteristics"""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Cluster Sizes",
            "Intra-cluster Distances",
            "Feature Importance",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
    )

    n_clusters = len(characteristics["cluster_sizes"])
    clusters = list(range(n_clusters))

    # Cluster sizes
    fig.add_trace(
        go.Bar(
            x=clusters,
            y=characteristics["cluster_sizes"],
            name="Size",
            marker_color="lightblue",
        ),
        row=1,
        col=1,
    )

    # Intra-cluster distances
    fig.add_trace(
        go.Bar(
            x=clusters,
            y=characteristics["intra_distances"],
            name="Distance",
            marker_color="lightcoral",
        ),
        row=1,
        col=2,
    )

    # Feature importance
    features = ["Income", "Spending", "Age"]
    fig.add_trace(
        go.Bar(
            x=features,
            y=characteristics["feature_importance"],
            name="Importance",
            marker_color="lightgreen",
        ),
        row=1,
        col=3,
    )

    fig.update_xaxes(title_text="Cluster", row=1, col=1)
    fig.update_xaxes(title_text="Cluster", row=1, col=2)
    fig.update_xaxes(title_text="Feature", row=1, col=3)

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Avg Distance", row=1, col=2)
    fig.update_yaxes(title_text="Std Dev", row=1, col=3)

    fig.update_layout(
        height=400, showlegend=False, title_text="Cluster Characteristics"
    )
    return fig


def create_efficiency_plot(efficiency_data: Dict) -> go.Figure:
    """Create efficiency analysis plots"""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Computation Time vs Clusters", "Iterations to Converge"),
    )

    # Time vs clusters
    fig.add_trace(
        go.Scatter(
            x=efficiency_data["k"],
            y=efficiency_data["time_per_run"],
            mode="lines+markers",
            marker=dict(size=10, color="blue"),
            name="Time",
        ),
        row=1,
        col=1,
    )

    # Iterations vs clusters
    fig.add_trace(
        go.Scatter(
            x=efficiency_data["k"],
            y=efficiency_data["iterations"],
            mode="lines+markers",
            marker=dict(size=10, color="red"),
            name="Iterations",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Number of Clusters", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters", row=1, col=2)

    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Iterations", row=1, col=2)

    fig.update_layout(
        height=400, showlegend=False, title_text="Computational Efficiency"
    )
    return fig


def create_business_segments_plot(business_data: Dict) -> go.Figure:
    """Create business segment visualization"""
    profiles = business_data["cluster_profiles"]
    interpretations = business_data["segment_interpretations"]

    clusters = sorted(profiles.keys())

    # Create bubble chart
    data = []
    for cluster in clusters:
        profile = profiles[cluster]
        interp = interpretations[cluster]

        data.append(
            {
                "Cluster": cluster,
                "Income": profile["income_mean"],
                "Spending": profile["spending_mean"],
                "Age": profile["age_mean"],
                "Size": profile["size"],
                "Segment": interp["segment_type"],
            }
        )

    df_business = pd.DataFrame(data)

    fig = px.scatter(
        df_business,
        x="Income",
        y="Spending",
        size="Size",
        color="Segment",
        hover_data=["Cluster", "Age", "Size"],
        title="Business Segments Overview",
        labels={"Income": "Avg Annual Income (k$)", "Spending": "Avg Spending Score"},
        text="Cluster",
    )

    fig.update_traces(textposition="top center")
    fig.update_layout(height=600, width=900)

    return fig


def create_cluster_profiles_table(business_data: Dict) -> go.Figure:
    """Create table showing cluster profiles"""
    profiles = business_data["cluster_profiles"]
    interpretations = business_data["segment_interpretations"]

    clusters = sorted(profiles.keys())

    header_values = [
        "Cluster",
        "Segment Type",
        "Size",
        "Avg Age",
        "Avg Income ($k)",
        "Avg Spending Score",
    ]
    cell_values = [
        clusters,
        [interpretations[c]["segment_type"] for c in clusters],
        [profiles[c]["size"] for c in clusters],
        [f"{profiles[c]['age_mean']:.1f}" for c in clusters],
        [f"{profiles[c]['income_mean']:.1f}" for c in clusters],
        [f"{profiles[c]['spending_mean']:.1f}" for c in clusters],
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_values,
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=12, color="black"),
                ),
                cells=dict(
                    values=cell_values,
                    fill_color="lavender",
                    align="left",
                    font=dict(size=11),
                ),
            )
        ]
    )

    fig.update_layout(title="Cluster Profiles Summary", height=300)

    return fig


def create_statistical_validation_plot(statistical_data: Dict) -> go.Figure:
    """Create visualization of statistical validation results"""
    feature_tests = statistical_data["feature_tests"]

    features = list(feature_tests.keys())
    f_stats = [feature_tests[f]["f_statistic"] for f in features]
    p_values = [feature_tests[f]["p_value"] for f in features]
    significant = [feature_tests[f]["significant"] for f in features]

    colors = ["green" if sig else "red" for sig in significant]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("F-Statistics (ANOVA)", "P-Values"),
    )

    # F-statistics
    fig.add_trace(
        go.Bar(x=features, y=f_stats, marker_color=colors, name="F-stat"), row=1, col=1
    )

    # P-values with significance line
    fig.add_trace(
        go.Bar(x=features, y=p_values, marker_color=colors, name="P-value"),
        row=1,
        col=2,
    )

    fig.add_hline(
        y=0.05,
        line_dash="dash",
        line_color="red",
        row=1,
        col=2,
        annotation_text="α=0.05",
    )

    fig.update_xaxes(title_text="Features", row=1, col=1)
    fig.update_xaxes(title_text="Features", row=1, col=2)

    fig.update_yaxes(title_text="F-Statistic", row=1, col=1)
    fig.update_yaxes(title_text="P-Value", row=1, col=2)

    fig.update_layout(
        height=400, showlegend=False, title_text="Statistical Validation (ANOVA)"
    )
    return fig


def create_stability_plot(stability_data: Dict) -> go.Figure:
    """Create stability analysis plot"""
    scores = stability_data["consistency_scores"]
    iterations = list(range(1, len(scores) + 1))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=scores,
            mode="lines+markers",
            marker=dict(size=8, color="blue"),
            line=dict(width=2),
            name="Consistency Score",
        )
    )

    # Add mean line
    mean_score = stability_data["mean_consistency"]
    fig.add_hline(
        y=mean_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_score:.3f}",
    )

    fig.update_layout(
        title="Cluster Stability Analysis",
        xaxis_title="Run Number",
        yaxis_title="Adjusted Rand Index",
        height=400,
        width=800,
    )

    return fig
