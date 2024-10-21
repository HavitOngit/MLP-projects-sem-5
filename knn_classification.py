import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for easier data handling
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['target_names'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
k = 5  # number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Function to plot decision boundaries
def plot_decision_boundary(X, y, model, scaler):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(f'KNN Decision Boundary (k={k})')
    plt.show()

# Plot decision boundary for the first two features
plot_decision_boundary(X[:, :2], y, knn, scaler)

# Function to find optimal k
def find_optimal_k(X_train, y_train, X_test, y_test, max_k):
    k_values = range(1, max_k + 1)
    accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. k for KNN')
    plt.show()
    
    optimal_k = k_values[accuracies.index(max(accuracies))]
    print(f"Optimal k: {optimal_k}")
    return optimal_k

# Find optimal k
optimal_k = find_optimal_k(X_train_scaled, y_train, X_test_scaled, y_test, 20)

# Train and evaluate model with optimal k
optimal_knn = KNeighborsClassifier(n_neighbors=optimal_k)
optimal_knn.fit(X_train_scaled, y_train)
optimal_y_pred = optimal_knn.predict(X_test_scaled)
optimal_accuracy = accuracy_score(y_test, optimal_y_pred)
print(f"Accuracy with optimal k: {optimal_accuracy:.4f}")

# Function to predict new samples
def predict_new_samples(model, scaler, samples):
    samples_scaled = scaler.transform(samples)
    predictions = model.predict(samples_scaled)
    return [iris.target_names[pred] for pred in predictions]

# Example usage of prediction function
new_samples = np.array([[5.1, 3.5, 1.4, 0.2],  # Likely Setosa
                        [6.3, 3.3, 6.0, 2.5],  # Likely Virginica
                        [5.5, 2.5, 4.0, 1.3]])  # Likely Versicolor

predictions = predict_new_samples(optimal_knn, scaler, new_samples)
print("\nPredictions for new samples:")
for sample, prediction in zip(new_samples, predictions):
    print(f"Sample {sample}: Predicted as {prediction}")