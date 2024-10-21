import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_boston, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# 1. Simple Linear Regression
print("1. Simple Linear Regression")

# Generate sample data
np.random.seed(42)
X_simple = np.random.rand(100, 1)
y_simple = 2 + 3 * X_simple + np.random.randn(100, 1) * 0.1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

# Create and train the model
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

# Make predictions
y_pred_simple = model_simple.predict(X_test)

# Evaluate the model
mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print(f"Mean Squared Error: {mse_simple:.4f}")
print(f"R2 Score: {r2_simple:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_simple, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# 2. Multiple Linear Regression
print("\n2. Multiple Linear Regression")

# Load the Boston Housing dataset
boston = load_boston()
X_multi = boston.data
y_multi = boston.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Create and train the model
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Make predictions
y_pred_multi = model_multi.predict(X_test)

# Evaluate the model
mse_multi = mean_squared_error(y_test, y_pred_multi)
r2_multi = r2_score(y_test, y_pred_multi)

print(f"Mean Squared Error: {mse_multi:.4f}")
print(f"R2 Score: {r2_multi:.4f}")

# Visualize feature importance
feature_importance = abs(model_multi.coef_)
feature_names = boston.feature_names

plt.figure(figsize=(12, 6))
plt.bar(feature_names, feature_importance)
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.title('Feature Importance in Multiple Linear Regression')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. Logistic Regression
print("\n3. Logistic Regression")

# Load the Iris dataset
iris = load_iris()
X_log = iris.data
y_log = iris.target

# Use only two classes for binary classification
X_log = X_log[y_log != 2]
y_log = y_log[y_log != 2]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model_log = LogisticRegression(random_state=42)
model_log.fit(X_train_scaled, y_train)

# Make predictions
y_pred_log = model_log.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_log)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log, target_names=iris.target_names[:2]))

# Visualize decision boundary
def plot_decision_boundary(X, y, model, scaler):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())]))
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

plot_decision_boundary(X_log[:, :2], y_log, model_log, scaler)