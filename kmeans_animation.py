"""
K-Means clustering animation module with step-by-step visualization
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple


class KMeansAnimator:
    """Class to create animated K-means clustering visualizations"""

    def __init__(
        self,
        X: np.ndarray,
        n_clusters: int = 5,
        max_iter: int = 20,
        random_state: int = 42,
    ):
        self.X = X
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(random_state)

        self.history = []
        self.metrics_history = []

    def initialize_centroids(self) -> np.ndarray:
        """Initialize centroids using k-means++ algorithm"""
        n_samples = self.X.shape[0]
        centroids = []

        # Choose first centroid randomly
        first_idx = np.random.randint(n_samples)
        centroids.append(self.X[first_idx])

        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Calculate distances to nearest centroid
            distances = np.array(
                [min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in self.X]
            )
            # Choose next centroid with probability proportional to distance squared
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.random()
            for idx, cum_prob in enumerate(cumulative_probs):
                if r < cum_prob:
                    centroids.append(self.X[idx])
                    break

        return np.array(centroids)

    def assign_clusters(self, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid"""
        distances = cdist(self.X, centroids, "euclidean")
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids(self, labels: np.ndarray) -> np.ndarray:
        """Calculate new centroids as mean of assigned points"""
        centroids = np.array(
            [self.X[labels == k].mean(axis=0) for k in range(self.n_clusters)]
        )
        return centroids

    def calculate_inertia(self, centroids: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares"""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = self.X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia

    def calculate_metrics(self, labels: np.ndarray, centroids: np.ndarray) -> Dict:
        """Calculate clustering metrics for current state"""
        inertia = self.calculate_inertia(centroids, labels)

        # Only calculate if we have at least 2 clusters with points
        if len(np.unique(labels)) >= 2:
            silhouette = silhouette_score(self.X, labels)
            davies_bouldin = davies_bouldin_score(self.X, labels)
            calinski_harabasz = calinski_harabasz_score(self.X, labels)
        else:
            silhouette = 0
            davies_bouldin = 0
            calinski_harabasz = 0

        return {
            "inertia": float(inertia),
            "silhouette": float(silhouette),
            "davies_bouldin": float(davies_bouldin),
            "calinski_harabasz": float(calinski_harabasz),
        }

    def fit(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run K-means algorithm and store history"""
        # Initialize centroids
        centroids = self.initialize_centroids()
        labels = self.assign_clusters(centroids)

        # Store initial state
        metrics = self.calculate_metrics(labels, centroids)
        self.history.append(
            {
                "iteration": 0,
                "centroids": centroids.copy(),
                "labels": labels.copy(),
                "step": "initialization",
            }
        )
        self.metrics_history.append({"iteration": 0, **metrics})

        # Iterate
        for iteration in range(1, self.max_iter + 1):
            # Update centroids
            old_centroids = centroids.copy()
            centroids = self.update_centroids(labels)

            # Store centroid update
            self.history.append(
                {
                    "iteration": iteration,
                    "centroids": centroids.copy(),
                    "labels": labels.copy(),
                    "step": "update_centroids",
                }
            )

            # Assign new clusters
            labels = self.assign_clusters(centroids)

            # Calculate metrics
            metrics = self.calculate_metrics(labels, centroids)
            self.metrics_history.append({"iteration": iteration, **metrics})

            # Store cluster assignment
            self.history.append(
                {
                    "iteration": iteration,
                    "centroids": centroids.copy(),
                    "labels": labels.copy(),
                    "step": "assign_clusters",
                }
            )

            # Check convergence
            if np.allclose(old_centroids, centroids):
                print(f"Converged at iteration {iteration}")
                break

        return labels, centroids

    def create_2d_animation(
        self, feature_idx: Tuple[int, int] = (0, 1), feature_names: List[str] = None
    ) -> go.Figure:
        """Create 2D animated visualization of clustering process"""
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.X.shape[1])]

        # Extract features for visualization
        X_2d = self.X[:, list(feature_idx)]

        # Create frames for animation
        frames = []

        for idx, state in enumerate(self.history):
            centroids_2d = state["centroids"][:, list(feature_idx)]
            labels = state["labels"]

            # Create scatter plot for points
            frame_data = []

            # Plot points colored by cluster
            for cluster in range(self.n_clusters):
                mask = labels == cluster
                frame_data.append(
                    go.Scatter(
                        x=X_2d[mask, 0],
                        y=X_2d[mask, 1],
                        mode="markers",
                        marker=dict(size=8, opacity=0.7),
                        name=f"Cluster {cluster}",
                        showlegend=(idx == 0),
                    )
                )

            # Plot centroids
            frame_data.append(
                go.Scatter(
                    x=centroids_2d[:, 0],
                    y=centroids_2d[:, 1],
                    mode="markers",
                    marker=dict(size=20, symbol="x", color="black", line=dict(width=2)),
                    name="Centroids",
                    showlegend=(idx == 0),
                )
            )

            # Get metrics for this frame
            metrics_idx = idx // 2 if state["step"] == "assign_clusters" else idx // 2
            if metrics_idx < len(self.metrics_history):
                metrics = self.metrics_history[metrics_idx]
                title = (
                    f"Iteration {state['iteration']} - {state['step'].replace('_', ' ').title()}<br>"
                    f"Inertia: {metrics['inertia']:.2f} | "
                    f"Silhouette: {metrics['silhouette']:.3f}"
                )
            else:
                title = f"Iteration {state['iteration']} - {state['step'].replace('_', ' ').title()}"

            frames.append(
                go.Frame(
                    data=frame_data, name=str(idx), layout=go.Layout(title_text=title)
                )
            )

        # Create initial figure
        fig = go.Figure(data=frames[0].data, frames=frames)

        # Update layout
        fig.update_layout(
            title=f"K-Means Clustering Animation (k={self.n_clusters})",
            xaxis_title=feature_names[feature_idx[0]],
            yaxis_title=feature_names[feature_idx[1]],
            hovermode="closest",
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 800, "redraw": True},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                    "transition": {"duration": 300},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "steps": [
                        {
                            "label": f"Step {i}",
                            "method": "animate",
                            "args": [
                                [str(i)],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300},
                                },
                            ],
                        }
                        for i in range(len(frames))
                    ],
                    "x": 0.1,
                    "len": 0.9,
                    "xanchor": "left",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            height=700,
            width=1000,
        )

        return fig

    def create_metrics_animation(self) -> go.Figure:
        """Create animated plot showing how metrics change over iterations"""
        iterations = [m["iteration"] for m in self.metrics_history]

        fig = go.Figure()

        # Inertia (WCSS)
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=[m["inertia"] for m in self.metrics_history],
                mode="lines+markers",
                name="Inertia (WCSS)",
                yaxis="y1",
            )
        )

        # Silhouette Score
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=[m["silhouette"] for m in self.metrics_history],
                mode="lines+markers",
                name="Silhouette Score",
                yaxis="y2",
            )
        )

        # Davies-Bouldin Index
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=[m["davies_bouldin"] for m in self.metrics_history],
                mode="lines+markers",
                name="Davies-Bouldin Index",
                yaxis="y3",
            )
        )

        # Calinski-Harabasz Score
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=[m["calinski_harabasz"] for m in self.metrics_history],
                mode="lines+markers",
                name="Calinski-Harabasz Score",
                yaxis="y4",
            )
        )

        # Update layout with multiple y-axes
        fig.update_layout(
            title="Clustering Metrics Evolution",
            xaxis=dict(title="Iteration"),
            yaxis=dict(
                title=dict(text="Inertia", font=dict(color="blue")),
                tickfont=dict(color="blue"),
            ),
            yaxis2=dict(
                title=dict(text="Silhouette", font=dict(color="red")),
                tickfont=dict(color="red"),
                anchor="x",
                overlaying="y",
                side="right",
            ),
            yaxis3=dict(
                title=dict(text="Davies-Bouldin", font=dict(color="green")),
                tickfont=dict(color="green"),
                anchor="free",
                overlaying="y",
                side="left",
                position=0.05,
            ),
            yaxis4=dict(
                title=dict(text="Calinski-Harabasz", font=dict(color="purple")),
                tickfont=dict(color="purple"),
                anchor="free",
                overlaying="y",
                side="right",
                position=0.95,
            ),
            height=600,
            width=1200,
            hovermode="x unified",
        )

        return fig

    def get_metrics_summary(self) -> Dict:
        """Get summary of metrics throughout iterations"""
        if not self.metrics_history:
            return {}

        final_metrics = self.metrics_history[-1]

        return {
            "final_iteration": final_metrics["iteration"],
            "final_inertia": final_metrics["inertia"],
            "final_silhouette": final_metrics["silhouette"],
            "final_davies_bouldin": final_metrics["davies_bouldin"],
            "final_calinski_harabasz": final_metrics["calinski_harabasz"],
            "convergence_iterations": len(self.metrics_history),
            "metrics_history": self.metrics_history,
        }


def create_kmeans_animation(
    X: np.ndarray,
    n_clusters: int = 5,
    feature_pairs: List[Tuple[int, int]] = None,
    feature_names: List[str] = None,
) -> Dict:
    """
    Create K-means animation and return all visualizations

    Args:
        X: Standardized feature matrix
        n_clusters: Number of clusters
        feature_pairs: List of feature index pairs to visualize
        feature_names: List of feature names

    Returns:
        Dictionary containing animation figures and metrics
    """
    if feature_pairs is None:
        # Default: visualize all pairs
        feature_pairs = [(0, 1), (0, 2), (1, 2)]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    animator = KMeansAnimator(X, n_clusters=n_clusters)
    labels, centroids = animator.fit()

    # Create animations for each feature pair
    animations = {}
    for idx, (f1, f2) in enumerate(feature_pairs):
        fig = animator.create_2d_animation(
            feature_idx=(f1, f2), feature_names=feature_names
        )
        animations[f"animation_{f1}_{f2}"] = fig

    # Create metrics animation
    metrics_fig = animator.create_metrics_animation()

    # Get metrics summary
    metrics_summary = animator.get_metrics_summary()

    return {
        "animations": animations,
        "metrics_animation": metrics_fig,
        "metrics_summary": metrics_summary,
        "final_labels": labels.tolist(),
        "final_centroids": centroids.tolist(),
        "history_length": len(animator.history),
    }
