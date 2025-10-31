# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load and explore the dataset
df = pd.read_csv('Mall_Customers.csv')
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())

# Basic data exploration
print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Distribution analysis
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.distplot(df['Age'])
plt.title('Distribution of Age')

plt.subplot(1, 3, 2)
sns.distplot(df['Annual Income (k$)'])
plt.title('Distribution of Annual Income')

plt.subplot(1, 3, 3)
sns.distplot(df['Spending Score (1-100)'])
plt.title('Distribution of Spending Score')

plt.tight_layout()
plt.show()

# Gender distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.show()

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Find the optimal number of clusters using elbow method
wcss = []  # Within-Cluster Sum of Squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_std)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Based on the elbow method, choose optimal K (typically 5 for this dataset)
optimal_clusters = 5

# Apply K-means with optimal clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
kmeans.fit(X_std)

# Add cluster labels to dataframe
df['Cluster'] = kmeans.labels_

# Analyze cluster characteristics
cluster_summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean', 
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'})

print("Cluster Summary:")
print(cluster_summary)

# 2D Visualization: Income vs Spending Score
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Annual Income (k$)'], 
                     df['Spending Score (1-100)'], 
                     c=df['Cluster'], 
                     cmap='viridis', 
                     s=60, 
                     alpha=0.7)

plt.colorbar(scatter)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments: Income vs Spending Score')
plt.grid(True, alpha=0.3)

# Add cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], 
           marker='X', s=200, c='red', label='Centroids')

plt.legend()
plt.show()

# 3D Visualization (optional)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['Annual Income (k$)'], 
                    df['Spending Score (1-100)'], 
                    df['Age'], 
                    c=df['Cluster'], 
                    cmap='viridis', 
                    s=60)

ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_zlabel('Age')
ax.set_title('3D Customer Segmentation')
plt.colorbar(scatter)
plt.show()

# Apply PCA for better visualization if you have more features
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', s=60)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Customer Segments (PCA Visualization)')
plt.colorbar(scatter)
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Save the segmented data
df.to_csv('segmented_customers.csv', index=False)

# Create segment profiles
segment_profiles = df.groupby('Cluster').agg({
    'Age': ['mean', 'std'],
    'Annual Income (k$)': ['mean', 'std'],
    'Spending Score (1-100)': ['mean', 'std'],
    'Gender': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
}).round(2)

print("Detailed Segment Profiles:")
print(segment_profiles)