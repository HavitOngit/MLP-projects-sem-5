import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Create a DataFrame for easier data handling
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Perform Grid Search for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("\nBest parameters found by Grid Search:")
print(grid_search.best_params_)

# Train the SVM model with best parameters
best_svm = grid_search.best_estimator_
best_svm.fit(X_train_scaled, y_train)

# Make predictions with the best model
y_pred_best = best_svm.predict(X_test_scaled)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"\nAccuracy with best parameters: {accuracy_best:.4f}")

print("\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred_best, target_names=cancer.target_names))

# Function to visualize decision boundary
def plot_decision_boundary(X, y, model, scaler):
    # Use PCA to reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Plot the decision boundary
    Z = model.predict(scaler.transform(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])))
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    
    # Plot the training points
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('SVM Decision Boundary (PCA-reduced feature space)')
    plt.colorbar(scatter)
    plt.show()

# Visualize decision boundary
plot_decision_boundary(X, y, best_svm, scaler)

# Function to predict new samples
def predict_new_samples(model, scaler, samples):
    samples_scaled = scaler.transform(samples)
    predictions = model.predict(samples_scaled)
    return [cancer.target_names[pred] for pred in predictions]

# Example usage of prediction function
new_samples = np.array([[15.0, 12.0, 95.0, 680.0, 0.09684, 0.08324, 0.0508, 0.02125, 0.1777, 0.06386,
                         0.2872, 0.8313, 2.085, 23.97, 0.005682, 0.01628, 0.02029, 0.006541, 0.01428, 0.002426,
                         16.51, 17.02, 111.1, 826.8, 0.1257, 0.1997, 0.2315, 0.08423, 0.2589, 0.08032],
                        [13.17, 18.66, 85.98, 534.6, 0.1158, 0.1231, 0.1226, 0.0734, 0.2128, 0.06777,
                         0.2871, 0.8937, 1.897, 24.25, 0.006532, 0.02336, 0.02905, 0.01215, 0.01743, 0.003643,
                         15.67, 24.64, 102.0, 744.9, 0.1557, 0.2845, 0.3285, 0.1741, 0.3358, 0.09215]])

predictions = predict_new_samples(best_svm, scaler, new_samples)
print("\nPredictions for new samples:")
for i, prediction in enumerate(predictions):
    print(f"Sample {i+1}: Predicted as {prediction}")