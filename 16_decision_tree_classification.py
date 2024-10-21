import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# Make predictions
y_pred = dt.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title("Decision Tree for Iris Dataset")
plt.show()

# Feature importance
feature_importance = dt.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(iris.feature_names)[sorted_idx])
plt.title('Feature Importance')
plt.xlabel('Gini Importance')
plt.tight_layout()
plt.show()

# Function to predict new samples
def predict_new_samples(model, scaler, samples):
    samples_scaled = scaler.transform(samples)
    predictions = model.predict(samples_scaled)
    return [iris.target_names[pred] for pred in predictions]

# Example usage of prediction function
new_samples = np.array([[5.1, 3.5, 1.4, 0.2],  # Likely Setosa
                        [6.3, 3.3, 6.0, 2.5],  # Likely Virginica
                        [5.5, 2.5, 4.0, 1.3]])  # Likely Versicolor

predictions = predict_new_samples(dt, scaler, new_samples)
print("\nPredictions for new samples:")
for sample, prediction in zip(new_samples, predictions):
    print(f"Sample {sample}: Predicted as {prediction}")

# Pruning the tree
def plot_pruning_results(X_train, X_test, y_train, y_test):
    ccp_alphas = []
    impurities = []

    for ccp_alpha in np.linspace(0, 0.05, 100):
        dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        dt.fit(X_train, y_train)
        ccp_alphas.append(ccp_alpha)
        impurities.append(dt.tree_.impurity.sum())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ccp_alphas, impurities)
    ax.set_xlabel("Effective alpha")
    ax.set_ylabel("Total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.show()

    train_scores = []
    test_scores = []

    for ccp_alpha in ccp_alphas:
        dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        dt.fit(X_train, y_train)
        train_scores.append(dt.score(X_train, y_train))
        test_scores.append(dt.score(X_test, y_test))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()

    return ccp_alphas, test_scores

# Plot pruning results
ccp_alphas, test_scores = plot_pruning_results(X_train_scaled, X_test_scaled, y_train, y_test)

# Find the optimal alpha
optimal_alpha = ccp_alphas[np.argmax(test_scores)]
print(f"\nOptimal alpha: {optimal_alpha:.4f}")

# Train the pruned tree
dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha)
dt_pruned.fit(X_train_scaled, y_train)

# Evaluate the pruned tree
y_pred_pruned = dt_pruned.predict(X_test_scaled)
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
print(f"Accuracy of pruned tree: {accuracy_pruned:.4f}")

# Visualize the pruned tree
plt.figure(figsize=(20,10))
plot_tree(dt_pruned, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title("Pruned Decision Tree for Iris Dataset")
plt.show()