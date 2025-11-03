"""
Data loading and clustering analysis module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    adjusted_rand_score,
)
from scipy.stats import f_oneway, chi2_contingency
from scipy.spatial.distance import cdist
import time
from typing import Dict, Tuple


def load_data(filepath: str = "Mall_Customers.csv") -> pd.DataFrame:
    """Load and return the customer dataset"""
    df = pd.read_csv(filepath)
    return df


def get_basic_info(df: pd.DataFrame) -> Dict:
    """Get basic information about the dataset"""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "head": df.head().to_dict("records"),
        "describe": df.describe().to_dict(),
        "gender_distribution": df["Gender"].value_counts().to_dict()
        if "Gender" in df.columns
        else {},
    }


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Prepare and standardize features for clustering"""
    X = df[["Annual Income (k$)", "Spending Score (1-100)", "Age"]].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return X_std, scaler


def elbow_method(X: np.ndarray, max_k: int = 15) -> Dict:
    """Calculate metrics for elbow method"""
    wcss = []
    silhouette_scores = []
    ch_scores = []
    db_scores = []

    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))

    # Find optimal k
    optimal_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
    optimal_k_ch = list(k_range)[np.argmax(ch_scores)]
    optimal_k_db = list(k_range)[np.argmin(db_scores)]

    return {
        "k_range": list(k_range),
        "wcss": wcss,
        "silhouette_scores": silhouette_scores,
        "ch_scores": ch_scores,
        "db_scores": db_scores,
        "optimal_k_silhouette": optimal_k_silhouette,
        "optimal_k_ch": optimal_k_ch,
        "optimal_k_db": optimal_k_db,
    }


def perform_clustering(X: np.ndarray, n_clusters: int = 5) -> Tuple[KMeans, np.ndarray]:
    """Perform K-means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


def evaluate_clustering_quality(X: np.ndarray, labels: np.ndarray) -> Dict:
    """Evaluate clustering quality using multiple metrics"""
    silhouette_avg = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)

    # Interpretation
    if silhouette_avg > 0.7:
        interpretation = "Excellent clustering structure"
    elif silhouette_avg > 0.5:
        interpretation = "Reasonable clustering structure"
    elif silhouette_avg > 0.25:
        interpretation = "Weak clustering structure"
    else:
        interpretation = "No substantial clustering structure"

    return {
        "silhouette_score": float(silhouette_avg),
        "calinski_harabasz_score": float(ch_score),
        "davies_bouldin_score": float(db_score),
        "interpretation": interpretation,
    }


def evaluate_cluster_stability(
    X: np.ndarray, n_clusters: int = 5, n_iterations: int = 10
) -> Dict:
    """Test stability of clusters by running K-means multiple times"""
    consistency_scores = []
    base_labels = None

    for i in range(n_iterations):
        kmeans_temp = KMeans(
            n_clusters=n_clusters, init="k-means++", random_state=i, n_init=10
        )
        labels_temp = kmeans_temp.fit_predict(X)

        if base_labels is None:
            base_labels = labels_temp
            consistency = 1.0
        else:
            consistency = adjusted_rand_score(base_labels, labels_temp)

        consistency_scores.append(consistency)

    return {
        "consistency_scores": consistency_scores,
        "mean_consistency": float(np.mean(consistency_scores)),
        "std_consistency": float(np.std(consistency_scores)),
        "n_iterations": n_iterations,
    }


def test_kmeans_efficiency(
    X: np.ndarray, k_range: range = range(2, 11), n_runs: int = 5
) -> Dict:
    """Test computational efficiency of K-means"""
    results = {"k": [], "time_per_run": [], "iterations": []}

    for k in k_range:
        run_times = []
        iterations_list = []

        for run in range(n_runs):
            start_time = time.time()
            kmeans = KMeans(n_clusters=k, init="k-means++", random_state=run, n_init=1)
            kmeans.fit(X)
            end_time = time.time()

            run_times.append(end_time - start_time)
            iterations_list.append(kmeans.n_iter_)

        results["k"].append(k)
        results["time_per_run"].append(np.mean(run_times))
        results["iterations"].append(np.mean(iterations_list))

    return results


def get_silhouette_analysis(X: np.ndarray, labels: np.ndarray) -> Dict:
    """Get silhouette analysis data for visualization"""
    silhouette_vals = silhouette_samples(X, labels)

    result = {
        "cluster_silhouettes": {},
        "avg_silhouette": float(silhouette_score(X, labels)),
    }

    for i in range(len(np.unique(labels))):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals = np.sort(cluster_silhouette_vals)
        result["cluster_silhouettes"][int(i)] = cluster_silhouette_vals.tolist()

    return result


def get_cluster_characteristics(
    X: np.ndarray, labels: np.ndarray, kmeans: KMeans
) -> Dict:
    """Get characteristics of each cluster"""
    centroids = kmeans.cluster_centers_

    # Cluster sizes
    cluster_sizes = [int(np.sum(labels == i)) for i in range(len(centroids))]

    # Intra-cluster distances
    intra_distances = []
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            distances = cdist(cluster_points, [centroids[i]], "euclidean")
            intra_distances.append(float(np.mean(distances)))
        else:
            intra_distances.append(0.0)

    # Feature importance
    feature_importance = np.std(centroids, axis=0).tolist()

    return {
        "cluster_sizes": cluster_sizes,
        "intra_distances": intra_distances,
        "feature_importance": feature_importance,
        "centroids": centroids.tolist(),
        "n_iterations": int(kmeans.n_iter_),
        "inertia": float(kmeans.inertia_),
    }


def statistical_cluster_validation(
    df: pd.DataFrame, cluster_column: str = "Cluster"
) -> Dict:
    """Perform statistical tests to validate cluster differences"""
    clusters = df[cluster_column].unique()
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

    results = {"feature_tests": {}, "gender_test": None}

    # ANOVA test for continuous variables
    for feature in features:
        cluster_groups = [
            df[df[cluster_column] == cluster][feature].values for cluster in clusters
        ]
        f_stat, p_value = f_oneway(*cluster_groups)

        results["feature_tests"][feature] = {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        }

    # Chi-square test for categorical variables
    if "Gender" in df.columns:
        contingency_table = pd.crosstab(df["Gender"], df[cluster_column])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        results["gender_test"] = {
            "chi_square": float(chi2),
            "p_value": float(p_value),
            "dof": int(dof),
            "significant": bool(p_value < 0.05),
        }

    return results


def business_validation(df: pd.DataFrame, cluster_column: str = "Cluster") -> Dict:
    """Validate clusters from a business perspective"""
    results = {"cluster_profiles": {}, "segment_interpretations": {}}

    for cluster in sorted(df[cluster_column].unique()):
        cluster_data = df[df[cluster_column] == cluster]

        age_mean = cluster_data["Age"].mean()
        income_mean = cluster_data["Annual Income (k$)"].mean()
        spending_mean = cluster_data["Spending Score (1-100)"].mean()
        size = len(cluster_data)

        # Business interpretation
        if income_mean > 70 and spending_mean > 70:
            segment_type = "High Value Customers"
        elif income_mean < 40 and spending_mean > 60:
            segment_type = "Budget Enthusiasts"
        elif income_mean > 70 and spending_mean < 40:
            segment_type = "Wealthy but Conservative"
        elif income_mean < 40 and spending_mean < 40:
            segment_type = "Low Value Customers"
        else:
            segment_type = "Average Customers"

        results["cluster_profiles"][int(cluster)] = {
            "age_mean": float(age_mean),
            "age_std": float(cluster_data["Age"].std()),
            "income_mean": float(income_mean),
            "income_std": float(cluster_data["Annual Income (k$)"].std()),
            "spending_mean": float(spending_mean),
            "spending_std": float(cluster_data["Spending Score (1-100)"].std()),
            "size": int(size),
        }

        results["segment_interpretations"][int(cluster)] = {
            "segment_type": segment_type,
            "description": f"Size: {size} customers, Avg Age: {age_mean:.1f}, Income: ${income_mean:.1f}k, Spending Score: {spending_mean:.1f}",
        }

    return results


def apply_pca(X: np.ndarray) -> Tuple[np.ndarray, PCA]:
    """Apply PCA for dimensionality reduction"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def get_comprehensive_analysis(df: pd.DataFrame, n_clusters: int = 5) -> Dict:
    """Run comprehensive analysis and return all results"""
    # Prepare features
    X_std, scaler = prepare_features(df)

    # Elbow method
    elbow_results = elbow_method(X_std)

    # Perform clustering
    kmeans, labels = perform_clustering(X_std, n_clusters)

    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels

    # Quality metrics
    quality = evaluate_clustering_quality(X_std, labels)

    # Stability
    stability = evaluate_cluster_stability(X_std, n_clusters)

    # Efficiency
    efficiency = test_kmeans_efficiency(X_std)

    # Silhouette analysis
    silhouette_data = get_silhouette_analysis(X_std, labels)

    # Cluster characteristics
    characteristics = get_cluster_characteristics(X_std, labels, kmeans)

    # Statistical validation
    statistical = statistical_cluster_validation(df_clustered)

    # Business validation
    business = business_validation(df_clustered)

    # PCA
    X_pca, pca = apply_pca(X_std)

    # Inverse transform centroids for original scale
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

    return {
        "basic_info": get_basic_info(df),
        "elbow_method": elbow_results,
        "clustering": {
            "n_clusters": n_clusters,
            "labels": labels.tolist(),
            "centroids_std": kmeans.cluster_centers_.tolist(),
            "centroids_original": centroids_original.tolist(),
        },
        "quality_metrics": quality,
        "stability": stability,
        "efficiency": efficiency,
        "silhouette_analysis": silhouette_data,
        "cluster_characteristics": characteristics,
        "statistical_validation": statistical,
        "business_validation": business,
        "pca": {
            "components": X_pca.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        },
        "data": df_clustered.to_dict("records"),
    }
