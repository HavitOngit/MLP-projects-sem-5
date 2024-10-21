import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Create a DataFrame for easier handling
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.tight_layout()
plt.show()

# Calculate cumulative explained variance ratio
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

# Plot cumulative explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), 
         cumulative_explained_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.legend()
plt.tight_layout()
plt.show()

# Determine the number of components for 95% explained variance
n_components = np.argmax(cumulative_explained_variance_ratio >= 0.95) + 1
print(f"Number of components for 95% explained variance: {n_components}")

# Apply PCA with the determined number of components
pca = PCA(n_components=n_components)
X_pca_reduced = pca.fit_transform(X_scaled)

# Visualize the reduced features
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Iris Dataset - PCA Reduced Features')
plt.colorbar(scatter, label='Target Class')
plt.tight_layout()
plt.show()

# Print feature importance (component loadings)
feature_importance = np.abs(pca.components_)
for i, component in enumerate(feature_importance):
    print(f"\nPrincipal Component {i+1} - Feature Importance:")
    for name, importance in zip(feature_names, component):
        print(f"{name}: {importance:.4f}")

# Function to project new data onto reduced PCA space
def project_new_data(new_data):
    scaled_data = scaler.transform(new_data)
    return pca.transform(scaled_data)

# Example usage
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example Iris sample
projected_sample = project_new_data(new_sample)
print("\nProjected new sample:")
print(projected_sample)