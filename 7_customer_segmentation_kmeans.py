import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_wine

# Load the Wine dataset
wine = load_wine()
X = wine.data
feature_names = wine.feature_names

# Create a DataFrame for easier handling
data = pd.DataFrame(X, columns=feature_names)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to determine the optimal number of clusters using the elbow method
def elbow_method(X, max_k):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

# Determine the optimal number of clusters
elbow_method(X_scaled, 10)

# Based on the elbow method, let's choose the optimal number of clusters
n_clusters = 3  # You may need to adjust this based on the elbow plot

# Perform k-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
data['Cluster'] = cluster_labels

# Calculate silhouette score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Visualize the clusters (using the first two features)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Wine Segments')
plt.colorbar(scatter)
plt.show()

# Analyze cluster characteristics
cluster_means = data.groupby('Cluster').mean()
print("\nCluster Characteristics:")
print(cluster_means)

# Function to assign a new sample to a cluster
def assign_cluster(sample_data):
    # Ensure sample_data has the same features used for clustering
    scaled_sample = scaler.transform(sample_data.reshape(1, -1))
    cluster = kmeans.predict(scaled_sample)
    return cluster[0]

# Example usage
new_sample = np.array([13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820])
assigned_cluster = assign_cluster(new_sample)
print(f"\nNew sample assigned to cluster: {assigned_cluster}")

# Visualize feature importance
feature_importance = np.abs(kmeans.cluster_centers_).mean(axis=0)
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(feature_names)[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Feature Importance for Clustering')
plt.tight_layout()
plt.show()