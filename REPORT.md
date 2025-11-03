# Customer Segmentation

This document provides a detailed explanation of all concepts, algorithms, and statistical methods used in this customer segmentation project.

---

## Table of Contents

1. [K-Means Clustering Algorithm](#1-k-means-clustering-algorithm)
2. [Distance Metrics](#2-distance-metrics)
3. [Feature Standardization](#3-feature-standardization)
4. [Elbow Method](#4-elbow-method)
5. [Cluster Quality Metrics](#5-cluster-quality-metrics)
6. [Statistical Validation](#6-statistical-validation)
7. [Dimensionality Reduction (PCA)](#7-dimensionality-reduction-pca)
8. [Cluster Stability Analysis](#8-cluster-stability-analysis)

---

## 1. K-Means Clustering Algorithm

### Overview
K-Means is an unsupervised machine learning algorithm that partitions a dataset into K distinct, non-overlapping clusters. Each data point belongs to the cluster with the nearest mean (centroid).

### Mathematical Formulation

#### Objective Function
The algorithm minimizes the Within-Cluster Sum of Squares (WCSS), also called inertia:

$$J = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2$$

Where:
- $J$ = Total inertia (objective to minimize)
- $K$ = Number of clusters
- $C_i$ = Set of points in cluster $i$
- $x$ = Data point
- $\mu_i$ = Centroid of cluster $i$
- $\|x - \mu_i\|$ = Euclidean distance between point and centroid

#### Algorithm Steps

1. **Initialization (K-Means++)**
   
   Instead of random initialization, K-Means++ chooses initial centroids strategically:
   
   - Select first centroid $\mu_1$ uniformly at random from data points
   - For each subsequent centroid $\mu_k$, select data point $x$ with probability:
   
   $$P(x) = \frac{D(x)^2}{\sum_{x' \in X} D(x')^2}$$
   
   Where $D(x)$ is the distance from $x$ to the nearest already-chosen centroid.

2. **Assignment Step**
   
   Assign each point $x_j$ to the nearest centroid:
   
   $$C_i^{(t)} = \{x_j : \|x_j - \mu_i^{(t)}\| \leq \|x_j - \mu_{i'}^{(t)}\| \text{ for all } i' = 1,\ldots,K\}$$
   
   Where $t$ is the current iteration.

3. **Update Step**
   
   Recalculate centroids as the mean of all points in each cluster:
   
   $$\mu_i^{(t+1)} = \frac{1}{|C_i^{(t)}|} \sum_{x \in C_i^{(t)}} x$$

4. **Convergence Check**
   
   Repeat steps 2-3 until:
   - Centroids don't change: $\|\mu_i^{(t+1)} - \mu_i^{(t)}\| < \epsilon$ for all $i$
   - Maximum iterations reached
   - Assignment labels don't change

### Computational Complexity

- Time Complexity: $O(n \cdot K \cdot d \cdot I)$
  - $n$ = number of data points
  - $K$ = number of clusters
  - $d$ = number of dimensions
  - $I$ = number of iterations until convergence
  
- Space Complexity: $O(n \cdot d + K \cdot d)$

---

## 2. Distance Metrics

### Euclidean Distance

The most commonly used distance metric in K-Means:

$$d(x, y) = \sqrt{\sum_{i=1}^{d} (x_i - y_i)^2}$$

Or in vector notation:

$$d(x, y) = \|x - y\|_2 = \sqrt{(x - y)^T(x - y)}$$

**Properties:**
- Symmetric: $d(x, y) = d(y, x)$
- Non-negative: $d(x, y) \geq 0$
- Identity: $d(x, x) = 0$
- Triangle inequality: $d(x, z) \leq d(x, y) + d(y, z)$

### Manhattan Distance (L1 Norm)

$$d(x, y) = \sum_{i=1}^{d} |x_i - y_i|$$

### Squared Euclidean Distance

Used in K-Means to avoid square root computation:

$$d^2(x, y) = \sum_{i=1}^{d} (x_i - y_i)^2$$

---

## 3. Feature Standardization

### Z-Score Normalization (Standardization)

Transform features to have zero mean and unit variance:

$$x_{standardized} = \frac{x - \mu}{\sigma}$$

Where:
- $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$ (mean)
- $\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}$ (standard deviation)

**Why Standardization:**
1. K-Means uses Euclidean distance, which is sensitive to feature scales
2. Features with larger ranges dominate the distance calculation
3. Standardization ensures all features contribute equally

---

## 4. Elbow Method

### Within-Cluster Sum of Squares (WCSS)

For each value of $K$, calculate WCSS:

$$WCSS(K) = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2$$

### Elbow Point Detection

The elbow point is where adding more clusters provides diminishing returns. Mathematically, we look for the point where the rate of WCSS decrease changes significantly.

**Rate of Change:**

$$\Delta WCSS_K = WCSS_{K-1} - WCSS_K$$

**Second Derivative (Curvature):**

$$\Delta^2 WCSS_K = \Delta WCSS_K - \Delta WCSS_{K+1}$$

The elbow is typically where $\Delta^2 WCSS_K$ is maximum (highest curvature).

---

## 5. Cluster Quality Metrics

### 5.1 Silhouette Score

Measures how similar a point is to its own cluster compared to other clusters.

**For a single point $i$:**

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = average distance from point $i$ to all other points in the same cluster (cohesion)
- $b(i)$ = minimum average distance from point $i$ to points in other clusters (separation)

**Average Silhouette Score:**

$$\text{Silhouette} = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

**Interpretation:**
- $s(i)$ close to +1: Point is well-matched to its cluster
- $s(i)$ close to 0: Point is on the border between clusters
- $s(i)$ close to -1: Point may be assigned to wrong cluster

**Range:** $[-1, 1]$, where higher is better

### 5.2 Calinski-Harabasz Index (Variance Ratio Criterion)

Ratio of between-cluster variance to within-cluster variance:

$$CH = \frac{SS_B / (K-1)}{SS_W / (n-K)}$$

Where:

**Between-Cluster Sum of Squares:**

$$SS_B = \sum_{i=1}^{K} n_i \|\mu_i - \mu\|^2$$

- $n_i$ = number of points in cluster $i$
- $\mu_i$ = centroid of cluster $i$
- $\mu$ = global centroid (mean of all data)

**Within-Cluster Sum of Squares:**

$$SS_W = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2$$

**Interpretation:**
- Higher values indicate better-defined clusters
- Measures how well-separated clusters are relative to how compact they are
- No fixed threshold; compare across different $K$ values

### 5.3 Davies-Bouldin Index

Measures average similarity between each cluster and its most similar cluster:

$$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} R_{ij}$$

Where:

$$R_{ij} = \frac{S_i + S_j}{M_{ij}}$$

- $S_i$ = average distance of points in cluster $i$ to centroid $\mu_i$
- $M_{ij}$ = distance between centroids $\mu_i$ and $\mu_j$

**Interpretation:**
- Lower values indicate better clustering
- Values closer to 0 indicate better separation
- $DB \geq 0$, where 0 is perfect (no overlap)

### Mathematical Relationships

**Cluster Compactness (Within-Cluster Variance):**

$$\sigma_i^2 = \frac{1}{n_i} \sum_{x \in C_i} \|x - \mu_i\|^2$$

**Cluster Separation (Between-Cluster Distance):**

$$d_{ij} = \|\mu_i - \mu_j\|$$

**Optimal Clustering Goal:**
- Minimize within-cluster variance: $\min \sum_{i=1}^{K} \sigma_i^2$
- Maximize between-cluster separation: $\max \min_{i \neq j} d_{ij}$

---

## 6. Statistical Validation

### 6.1 Analysis of Variance (ANOVA)

Tests whether cluster means differ significantly across features.

**Null Hypothesis:** $H_0: \mu_1 = \mu_2 = \cdots = \mu_K$ (all cluster means are equal)

**F-Statistic:**

$$F = \frac{MS_B}{MS_W} = \frac{SS_B / (K-1)}{SS_W / (n-K)}$$

Where:
- $MS_B$ = Mean Square Between groups
- $MS_W$ = Mean Square Within groups
- $SS_B$ = Sum of Squares Between clusters
- $SS_W$ = Sum of Squares Within clusters

**Calculation Details:**

**Total Sum of Squares:**

$$SS_T = \sum_{i=1}^{n} (x_i - \bar{x})^2$$

**Between-Cluster Sum of Squares:**

$$SS_B = \sum_{j=1}^{K} n_j (\bar{x}_j - \bar{x})^2$$

**Within-Cluster Sum of Squares:**

$$SS_W = SS_T - SS_B = \sum_{j=1}^{K} \sum_{i \in C_j} (x_i - \bar{x}_j)^2$$

**Degrees of Freedom:**
- Between: $df_B = K - 1$
- Within: $df_W = n - K$
- Total: $df_T = n - 1$

**P-value:** Probability of observing F-statistic this extreme under $H_0$

$$p = P(F_{df_B, df_W} \geq F_{observed})$$

**Interpretation:**
- If $p < \alpha$ (typically 0.05): Reject $H_0$, clusters differ significantly
- Large F-statistic with small p-value indicates strong cluster separation

### 6.2 Chi-Square Test (for Categorical Variables)

Tests independence between cluster assignment and categorical features (e.g., Gender).

**Test Statistic:**

$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Where:
- $O_{ij}$ = Observed frequency in cell $(i,j)$
- $E_{ij}$ = Expected frequency: $E_{ij} = \frac{n_{i \cdot} \cdot n_{\cdot j}}{n}$
- $r$ = number of rows (categories)
- $c$ = number of columns (clusters)

**Degrees of Freedom:** $df = (r-1)(c-1)$

**Interpretation:**
- Large $\chi^2$ with small p-value: Strong association between variable and clusters
- If $p < 0.05$: Clusters differ significantly in categorical distribution

---

## 7. Dimensionality Reduction (PCA)

### Principal Component Analysis

PCA finds orthogonal directions of maximum variance in high-dimensional data.

**Steps:**

1. **Center the Data:**
   
   $$X_{centered} = X - \bar{X}$$
   
   Where $\bar{X}$ is the mean of all features.

2. **Compute Covariance Matrix:**
   
   $$\Sigma = \frac{1}{n-1} X_{centered}^T X_{centered}$$
   
   $\Sigma$ is a $d \times d$ symmetric matrix where:
   - $\Sigma_{ij} = \text{Cov}(X_i, X_j)$
   - Diagonal elements: variances
   - Off-diagonal elements: covariances

3. **Eigenvalue Decomposition:**
   
   $$\Sigma v_i = \lambda_i v_i$$
   
   Where:
   - $v_i$ = eigenvectors (principal components)
   - $\lambda_i$ = eigenvalues (variance explained)
   
   Sort eigenvalues: $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$

4. **Project Data:**
   
   Select top $k$ eigenvectors to form projection matrix $V_k$:
   
   $$Z = X_{centered} V_k$$
   
   Where $Z$ is the $n \times k$ projected data matrix.

### Variance Explained

**Proportion of Variance Explained by PC $i$:**

$$\frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

**Cumulative Variance Explained by first $k$ PCs:**

$$\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

### Reconstruction Error

Original data can be approximately reconstructed:

$$\hat{X} = Z V_k^T + \bar{X}$$

**Reconstruction Error:**

$$E = \sum_{j=k+1}^{d} \lambda_j$$

**Why PCA for Visualization:**
- Projects high-dimensional data to 2D or 3D
- Preserves maximum variance (information)
- Makes cluster patterns visible to human eye
- First 2-3 PCs typically capture 70-90% of variance

---

## 8. Cluster Stability Analysis

### Adjusted Rand Index (ARI)

Measures similarity between two clustering assignments, adjusted for chance.

**Formula:**

$$ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}{\frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}$$

Where:
- $n_{ij}$ = number of points in cluster $i$ of first clustering and cluster $j$ of second
- $a_i$ = number of points in cluster $i$ of first clustering
- $b_j$ = number of points in cluster $j$ of second clustering
- $n$ = total number of points

**Simplified Form:**

$$ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}$$

Where:
- $RI$ = Rand Index (proportion of agreeing pairs)
- $E[RI]$ = Expected Rand Index under random labeling

**Properties:**
- Range: $[-1, 1]$
- $ARI = 1$: Perfect agreement
- $ARI = 0$: Random labeling
- $ARI < 0$: Worse than random

**Interpretation:**
- $ARI > 0.8$: Highly stable clustering
- $0.6 < ARI < 0.8$: Moderately stable
- $ARI < 0.6$: Unstable clustering

### Stability Testing Procedure

1. Run K-Means multiple times with different random seeds
2. Compare each run's labels to baseline run using ARI
3. Calculate mean and standard deviation of ARI scores:

$$\text{Mean ARI} = \frac{1}{m} \sum_{i=1}^{m} ARI_i$$

$$\text{Std ARI} = \sqrt{\frac{1}{m-1} \sum_{i=1}^{m} (ARI_i - \overline{ARI})^2}$$

Where $m$ is the number of runs.

**High stability:** High mean ARI with low standard deviation

---

## 9. Performance Metrics

### Convergence Criteria

**Centroid Change:**

$$\Delta\mu = \max_{i=1,\ldots,K} \|\mu_i^{(t+1)} - \mu_i^{(t)}\|$$

Converged if: $\Delta\mu < \epsilon$ (e.g., $\epsilon = 10^{-4}$)

**Inertia Change:**

$$\Delta J = |J^{(t)} - J^{(t+1)}|$$

Converged if: $\Delta J / J^{(t)} < \epsilon$

### Computational Efficiency Metrics

**Iterations to Convergence:** Number of assignment-update cycles until convergence

**Time Complexity per Iteration:**
- Distance calculation: $O(n \cdot K \cdot d)$
- Centroid update: $O(n \cdot d)$
- Total: $O(n \cdot K \cdot d)$

**Memory Usage:**
- Data storage: $O(n \cdot d)$
- Centroid storage: $O(K \cdot d)$
- Distance matrix (if cached): $O(n \cdot K)$

---

## 10. Business Metrics

### Cluster Size Balance

**Coefficient of Variation:**

$$CV = \frac{\sigma_n}{\mu_n} = \frac{\sqrt{\frac{1}{K}\sum_{i=1}^{K}(n_i - \bar{n})^2}}{\frac{1}{K}\sum_{i=1}^{K}n_i}$$

Where:
- $n_i$ = size of cluster $i$
- $\bar{n}$ = average cluster size

**Interpretation:**
- $CV < 0.3$: Well-balanced clusters
- $0.3 < CV < 0.6$: Moderate imbalance
- $CV > 0.6$: Highly imbalanced (some clusters too small/large)

### Cluster Characterization

**Feature Mean Deviation:**

For feature $f$ in cluster $i$:

$$\delta_{if} = \frac{\mu_{if} - \mu_f}{\sigma_f}$$

Where:
- $\mu_{if}$ = mean of feature $f$ in cluster $i$
- $\mu_f$ = global mean of feature $f$
- $\sigma_f$ = global standard deviation of feature $f$

**Interpretation:**
- $|\delta_{if}| > 1$: Cluster significantly differs from overall population
- $|\delta_{if}| > 2$: Very strong characteristic of cluster

---

## 11. Summary of Key Formulas

### Distance and Similarity
| Metric | Formula |
|--------|---------|
| Euclidean Distance | $d(x,y) = \sqrt{\sum_i (x_i - y_i)^2}$ |
| WCSS (Inertia) | $J = \sum_{i=1}^K \sum_{x \in C_i} \|x - \mu_i\|^2$ |

### Cluster Quality
| Metric | Formula | Range | Optimal |
|--------|---------|-------|---------|
| Silhouette Score | $s = \frac{b-a}{\max(a,b)}$ | [-1, 1] | → 1 |
| Calinski-Harabasz | $CH = \frac{SS_B/(K-1)}{SS_W/(n-K)}$ | [0, ∞) | → ∞ |
| Davies-Bouldin | $DB = \frac{1}{K}\sum_i \max_{j \neq i} \frac{S_i + S_j}{M_{ij}}$ | [0, ∞) | → 0 |

### Statistical Tests
| Test | Statistic | Usage |
|------|-----------|-------|
| ANOVA | $F = \frac{MS_B}{MS_W}$ | Continuous features |
| Chi-Square | $\chi^2 = \sum \frac{(O-E)^2}{E}$ | Categorical features |

### Stability
| Metric | Formula | Range |
|--------|---------|-------|
| Adjusted Rand Index | $ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}$ | [-1, 1] |

---

## 12. Interpretations and Thresholds

### Quality Thresholds

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| Silhouette Score | > 0.7 | 0.5-0.7 | 0.25-0.5 | < 0.25 |
| Davies-Bouldin | < 0.5 | 0.5-1.0 | 1.0-1.5 | > 1.5 |
| ARI (Stability) | > 0.9 | 0.75-0.9 | 0.6-0.75 | < 0.6 |

### Statistical Significance

- **ANOVA p-value < 0.05:** Clusters differ significantly
- **Chi-square p-value < 0.05:** Association with categorical variable
- **High F-statistic:** Strong separation between clusters

---

## 13. Practical Considerations

### Choosing K (Number of Clusters)

Multiple methods should agree:
1. **Elbow Method:** Look for "elbow" in WCSS plot
2. **Silhouette Analysis:** Choose K with highest average silhouette
3. **Calinski-Harabasz:** Choose K with highest CH index
4. **Davies-Bouldin:** Choose K with lowest DB index
5. **Domain Knowledge:** Business-meaningful number of segments

### Convergence Guarantees

K-Means has these properties:
- **Monotonic decrease:** $J^{(t+1)} \leq J^{(t)}$ (inertia never increases)
- **Local optimum:** Converges to local minimum, not necessarily global
- **Finite convergence:** Algorithm terminates in finite steps
- **Sensitive to initialization:** Different starting points → different results

### When K-Means Works Well

✅ **Good for:**
- Spherical/globular clusters
- Similar-sized clusters
- Well-separated clusters
- Numerical features

❌ **Poor for:**
- Non-convex shapes (e.g., crescents, rings)
- Vastly different cluster sizes
- Different cluster densities
- Hierarchical structures

---

## 14. References and Further Reading

### Key Papers
1. MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
2. Arthur, D. & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding"
3. Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation"

### Textbooks
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). "An Introduction to Statistical Learning"
- Murphy, K. P. (2012). "Machine Learning: A Probabilistic Perspective"

### Online Resources
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/clustering.html
- Stanford CS229 Lecture Notes: http://cs229.stanford.edu/notes/
- MIT OpenCourseWare: Machine Learning courses

---

## Appendix: Implementation Notes

### Numerical Stability

**Avoiding numerical issues:**
1. Use squared distances to avoid square root operations
2. Subtract mean before computing variance (two-pass algorithm)
3. Check for empty clusters and reinitialize if needed
4. Use tolerance threshold for floating-point comparisons

### Optimization Techniques

1. **Early Stopping:** Terminate if assignments don't change
2. **Mini-batch K-Means:** Use random subsets for large datasets
3. **Vectorization:** Use matrix operations instead of loops
4. **Caching:** Store distance calculations when possible
5. **Parallel Processing:** Assign points to clusters in parallel

### Handling Edge Cases

- **Empty clusters:** Reinitialize centroid from largest cluster
- **Identical centroids:** Add small random noise
- **Single-point clusters:** Consider merging or adjusting K
- **Outliers:** Consider outlier detection before clustering

---