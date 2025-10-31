from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time

def evaluate_clustering_quality(X, labels):
    """
    Evaluate clustering quality using multiple internal metrics
    """
    print("=== CLUSTERING QUALITY METRICS ===")
    
    # Silhouette Score (-1 to 1, higher is better)
    silhouette_avg = silhouette_score(X, labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # Calinski-Harabasz Index (higher is better)
    ch_score = calinski_harabasz_score(X, labels)
    print(f"Calinski-Harabasz Index: {ch_score:.4f}")
    
    # Davies-Bouldin Index (lower is better)
    db_score = davies_bouldin_score(X, labels)
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    
    # Interpret results
    print("\n=== INTERPRETATION ===")
    if silhouette_avg > 0.7:
        print("✓ Excellent clustering structure")
    elif silhouette_avg > 0.5:
        print("✓ Reasonable clustering structure")
    elif silhouette_avg > 0.25:
        print("○ Weak clustering structure")
    else:
        print("✗ No substantial clustering structure")

# Apply evaluation
evaluate_clustering_quality(X_std, kmeans.labels_)

def evaluate_cluster_stability(X, n_clusters=5, n_iterations=10):
    """
    Test stability of clusters by running K-means multiple times
    """
    from collections import Counter
    import numpy as np
    
    consistency_scores = []
    base_labels = None
    
    for i in range(n_iterations):
        kmeans_temp = KMeans(n_clusters=n_clusters, random_state=i)
        labels_temp = kmeans_temp.fit_predict(X)
        
        if base_labels is None:
            base_labels = labels_temp
            consistency = 1.0
        else:
            # Calculate consistency with previous run
            consistency = adjusted_rand_score(base_labels, labels_temp)
        
        consistency_scores.append(consistency)
    
    print(f"Cluster Stability Analysis ({n_iterations} runs):")
    print(f"Average Consistency Score: {np.mean(consistency_scores):.4f}")
    print(f"Consistency Std Dev: {np.std(consistency_scores):.4f}")
    
    return consistency_scores

stability_scores = evaluate_cluster_stability(X_std)

def find_optimal_clusters(X, max_k=15):
    """
    Enhanced elbow method with multiple metrics
    """
    wcss = []  # Within-cluster sum of squares
    silhouette_scores = []
    ch_scores = []  # Calinski-Harabasz scores
    db_scores = []  # Davies-Bouldin scores
    
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))
    
    # Plot multiple metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Elbow curve
    ax1.plot(k_range, wcss, 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('WCSS')
    ax1.set_title('Elbow Method')
    ax1.grid(True)
    
    # Silhouette scores
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True)
    
    # Calinski-Harabasz
    ax3.plot(k_range, ch_scores, 'go-')
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Calinski-Harabasz Score')
    ax3.set_title('Variance Ratio Criterion')
    ax3.grid(True)
    
    # Davies-Bouldin
    ax4.plot(k_range, db_scores, 'mo-')
    ax4.set_xlabel('Number of Clusters')
    ax4.set_ylabel('Davies-Bouldin Score')
    ax4.set_title('Davies-Bouldin Index (lower is better)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal k based on multiple criteria
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    optimal_k_ch = k_range[np.argmax(ch_scores)]
    optimal_k_db = k_range[np.argmin(db_scores)]
    
    print(f"Optimal K by Silhouette: {optimal_k_silhouette}")
    print(f"Optimal K by Calinski-Harabasz: {optimal_k_ch}")
    print(f"Optimal K by Davies-Bouldin: {optimal_k_db}")
    
    return {
        'k_range': list(k_range),
        'wcss': wcss,
        'silhouette_scores': silhouette_scores,
        'ch_scores': ch_scores,
        'db_scores': db_scores
    }

# Run enhanced analysis
metrics = find_optimal_clusters(X_std)

def test_kmeans_efficiency(X, k_range=range(2, 11), n_runs=5):
    """
    Test computational efficiency of K-means
    """
    results = {
        'k': [],
        'time_per_run': [],
        'iterations': [],
        'convergence_time': []
    }
    
    for k in k_range:
        run_times = []
        iterations_list = []
        
        for run in range(n_runs):
            start_time = time.time()
            kmeans = KMeans(n_clusters=k, random_state=run, n_init=1)
            kmeans.fit(X)
            end_time = time.time()
            
            run_times.append(end_time - start_time)
            iterations_list.append(kmeans.n_iter_)
        
        results['k'].append(k)
        results['time_per_run'].append(np.mean(run_times))
        results['iterations'].append(np.mean(iterations_list))
        results['convergence_time'].append(np.mean(run_times))
    
    # Plot efficiency results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time vs Clusters
    ax1.plot(results['k'], results['time_per_run'], 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Average Time (seconds)')
    ax1.set_title('Computational Time vs Number of Clusters')
    ax1.grid(True)
    
    # Iterations vs Clusters
    ax2.plot(results['k'], results['iterations'], 'ro-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Average Iterations to Converge')
    ax2.set_title('Convergence Speed vs Number of Clusters')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Test efficiency
efficiency_results = test_kmeans_efficiency(X_std)

def visualize_cluster_quality(X, labels, kmeans_model):
    """
    Create comprehensive visualizations for cluster quality assessment
    """
    from scipy.spatial.distance import cdist
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Silhouette Analysis
    plt.subplot(2, 3, 1)
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X, labels)
    
    y_lower = 10
    for i in range(len(np.unique(labels))):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        y_upper = y_lower + cluster_silhouette_vals.shape[0]
        
        color = plt.cm.nipy_spectral(float(i) / len(np.unique(labels)))
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        y_lower = y_upper + 10
    
    plt.axvline(x=silhouette_score(X, labels), color="red", linestyle="--")
    plt.title('Silhouette Plot for Clusters')
    plt.xlabel('Silhouette Coefficient Values')
    plt.ylabel('Cluster Label')
    
    # 2. Cluster Size Distribution
    plt.subplot(2, 3, 2)
    cluster_sizes = np.bincount(labels)
    plt.bar(range(len(cluster_sizes)), cluster_sizes)
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Points')
    
    # 3. Intra-cluster vs Inter-cluster Distance
    plt.subplot(2, 3, 3)
    centroids = kmeans_model.cluster_centers_
    
    # Intra-cluster distances
    intra_distances = []
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            distances = cdist(cluster_points, [centroids[i]], 'euclidean')
            intra_distances.append(np.mean(distances))
    
    plt.bar(range(len(intra_distances)), intra_distances)
    plt.title('Average Intra-cluster Distances')
    plt.xlabel('Cluster')
    plt.ylabel('Mean Distance to Centroid')
    
    # 4. Feature Importance in Clustering
    plt.subplot(2, 3, 4)
    feature_importance = np.std(centroids, axis=0)
    features = ['Annual Income', 'Spending Score', 'Age']
    plt.bar(features, feature_importance)
    plt.title('Feature Importance in Clustering')
    plt.ylabel('Standard Deviation of Centroids')
    plt.xticks(rotation=45)
    
    # 5. Convergence Monitoring
    plt.subplot(2, 3, 5)
    # Simulate inertia over iterations (you might need to modify KMeans to track this)
    plt.plot(range(kmeans_model.n_iter_), [kmeans_model.inertia_] * kmeans_model.n_iter_, 'bo-')
    plt.title('Convergence Pattern')
    plt.xlabel('Iteration')
    plt.ylabel('Inertia')
    
    plt.tight_layout()
    plt.show()

# Create quality visualizations
visualize_cluster_quality(X_std, kmeans.labels_, kmeans)

def statistical_cluster_validation(df, cluster_column='Cluster'):
    """
    Perform statistical tests to validate cluster differences
    """
    from scipy.stats import f_oneway, chi2_contingency
    import pandas as pd
    
    clusters = df[cluster_column].unique()
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    print("=== STATISTICAL SIGNIFICANCE TESTING ===")
    
    # ANOVA test for continuous variables
    for feature in features:
        cluster_groups = [df[df[cluster_column] == cluster][feature] for cluster in clusters]
        f_stat, p_value = f_oneway(*cluster_groups)
        
        print(f"\n{feature}:")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  ✓ Significant differences between clusters")
        else:
            print(f"  ✗ No significant differences between clusters")
    
    # Chi-square test for categorical variables
    if 'Gender' in df.columns:
        contingency_table = pd.crosstab(df['Gender'], df[cluster_column])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"\nGender Distribution:")
        print(f"  Chi-square: {chi2:.4f}")
        print(f"  P-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  ✓ Gender distribution differs significantly between clusters")
        else:
            print(f"  ✗ Gender distribution is similar across clusters")

# Run statistical validation
statistical_cluster_validation(df, 'Cluster')

def business_validation(df, cluster_column='Cluster'):
    """
    Validate clusters from a business perspective
    """
    cluster_profiles = df.groupby(cluster_column).agg({
        'Age': ['mean', 'std'],
        'Annual Income (k$)': ['mean', 'std'],
        'Spending Score (1-100)': ['mean', 'std'],
        'CustomerID': 'count'
    }).round(2)
    
    print("=== BUSINESS VALIDATION ===")
    print("Cluster Profiles:")
    print(cluster_profiles)
    
    # Check for meaningful business segments
    print("\n=== SEGMENT INTERPRETABILITY ===")
    for cluster in sorted(df[cluster_column].unique()):
        cluster_data = df[df[cluster_column] == cluster]
        
        age_mean = cluster_data['Age'].mean()
        income_mean = cluster_data['Annual Income (k$)'].mean()
        spending_mean = cluster_data['Spending Score (1-100)'].mean()
        
        # Simple business interpretation
        segment_type = ""
        if income_mean > 70 and spending_mean > 70:
            segment_type = "High Value Customers"
        elif income_mean < 40 and spending_mean > 60:
            segment_type = "Budget Enthusiasts"
        elif income_mean > 70 and spending_mean < 40:
            segment_type = "Wealthy but Conservative"
        else:
            segment_type = "Average Customers"
        
        print(f"Cluster {cluster}: {segment_type}")
        print(f"  Size: {len(cluster_data)} customers")
        print(f"  Avg Age: {age_mean:.1f}, Income: ${income_mean:.1f}k, Spending Score: {spending_mean:.1f}")

# Business validation
business_validation(df)

def run_comprehensive_kmeans_test(X, df, true_k=5):
    """
    Run all tests in one comprehensive function
    """
    print("🔍 COMPREHENSIVE K-MEANS EVALUATION")
    print("=" * 50)
    
    # 1. Find optimal clusters
    print("\n1. OPTIMAL CLUSTER DETERMINATION")
    metrics = find_optimal_clusters(X)
    
    # 2. Fit final model
    kmeans_final = KMeans(n_clusters=true_k, random_state=42)
    labels_final = kmeans_final.fit_predict(X)
    df['Cluster'] = labels_final
    
    # 3. Evaluate quality
    print("\n2. CLUSTERING QUALITY ASSESSMENT")
    evaluate_clustering_quality(X, labels_final)
    
    # 4. Test stability
    print("\n3. CLUSTER STABILITY ANALYSIS")
    stability_scores = evaluate_cluster_stability(X, true_k)
    
    # 5. Efficiency testing
    print("\n4. COMPUTATIONAL EFFICIENCY")
    efficiency_results = test_kmeans_efficiency(X)
    
    # 6. Statistical validation
    print("\n5. STATISTICAL VALIDATION")
    statistical_cluster_validation(df)
    
    # 7. Business validation
    print("\n6. BUSINESS VALIDATION")
    business_validation(df)
    
    # 8. Quality visualization
    print("\n7. QUALITY VISUALIZATION")
    visualize_cluster_quality(X, labels_final, kmeans_final)
    
    return kmeans_final, df

# Run complete test suite
final_model, segmented_df = run_comprehensive_kmeans_test(X_std, df)

