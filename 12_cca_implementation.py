import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features1 = 5
n_features2 = 4

# Create two sets of features
X1, _ = make_classification(n_samples=n_samples, n_features=n_features1, n_informative=3, n_redundant=2, n_classes=1, n_clusters_per_class=1)
X2, _ = make_classification(n_samples=n_samples, n_features=n_features2, n_informative=2, n_redundant=2, n_classes=1, n_clusters_per_class=1)

# Standardize the features
scaler1 = StandardScaler()
scaler2 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)
X2_scaled = scaler2.fit_transform(X2)

# Perform CCA
n_components = min(n_features1, n_features2)
cca = CCA(n_components=n_components)
X1_cca, X2_cca = cca.fit_transform(X1_scaled, X2_scaled)

# Calculate correlations between canonical variates
correlations = [np.corrcoef(X1_cca[:, i], X2_cca[:, i])[0, 1] for i in range(n_components)]

# Print correlations
print("Correlations between canonical variates:")
for i, corr in enumerate(correlations):
    print(f"Component {i+1}: {corr:.4f}")

# Visualize correlations
plt.figure(figsize=(10, 6))
plt.bar(range(1, n_components + 1), correlations)
plt.xlabel('Canonical Component')
plt.ylabel('Correlation')
plt.title('Correlations between Canonical Variates')
plt.xticks(range(1, n_components + 1))
plt.show()

# Visualize the first two canonical variates
plt.figure(figsize=(10, 6))
plt.scatter(X1_cca[:, 0], X2_cca[:, 0], alpha=0.5)
plt.xlabel('First Canonical Variate (X1)')
plt.ylabel('First Canonical Variate (X2)')
plt.title('First Canonical Variates')
plt.show()

# Function to plot loadings
def plot_loadings(loadings, feature_names, title):
    plt.figure(figsize=(12, 6))
    for i in range(loadings.shape[1]):
        plt.bar(feature_names, loadings[:, i], alpha=0.5, label=f'CV{i+1}')
    plt.xlabel('Features')
    plt.ylabel('Loading')
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Plot loadings for X1
X1_loadings = cca.x_loadings_
X1_feature_names = [f'X1_F{i+1}' for i in range(n_features1)]
plot_loadings(X1_loadings, X1_feature_names, 'CCA Loadings for X1')

# Plot loadings for X2
X2_loadings = cca.y_loadings_
X2_feature_names = [f'X2_F{i+1}' for i in range(n_features2)]
plot_loadings(X2_loadings, X2_feature_names, 'CCA Loadings for X2')

# Function to predict new data
def predict_cca(X1_new, X2_new):
    X1_new_scaled = scaler1.transform(X1_new)
    X2_new_scaled = scaler2.transform(X2_new)
    X1_new_cca, X2_new_cca = cca.transform(X1_new_scaled, X2_new_scaled)
    return X1_new_cca, X2_new_cca

# Example usage of prediction function
X1_new = np.random.rand(5, n_features1)
X2_new = np.random.rand(5, n_features2)
X1_new_cca, X2_new_cca = predict_cca(X1_new, X2_new)

print("\nPredicted CCA components for new data:")
print("X1 CCA components:")
print(X1_new_cca)
print("\nX2 CCA components:")
print(X2_new_cca)