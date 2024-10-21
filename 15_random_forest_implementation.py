import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Create a DataFrame for easier data handling
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Perform Randomized Search for hyperparameter tuning
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_distributions=param_dist,
                               n_iter=100, cv=3, verbose=1, random_state=42, n_jobs=-1)
rf_random.fit(X_train_scaled, y_train)

print("\nBest parameters found by Randomized Search:")
print(rf_random.best_params_)

# Train the Random Forest model with best parameters
best_rf = rf_random.best_estimator_
best_rf.fit(X_train_scaled, y_train)

# Make predictions with the best model
y_pred_best = best_rf.predict(X_test_scaled)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"\nAccuracy with best parameters: {accuracy_best:.4f}")

print("\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred_best, target_names=wine.target_names))

# Feature importance
feature_importance = best_rf.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.barh(pos, feature_importance[sorted_idx], align='center')
ax1.set_yticks(pos)
ax1.set_yticklabels(np.array(wine.feature_names)[sorted_idx])
ax1.set_title('Feature Importance (MDI)')

# Permutation Importance
result = permutation_importance(best_rf, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()
ax2.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(wine.feature_names)[sorted_idx])
ax2.set_title("Permutation Importance (test set)")

fig.tight_layout()
plt.show()

# Function to visualize trees in the forest
def plot_tree(tree, feature_names, class_names):
    from sklearn.tree import plot_tree
    plt.figure(figsize=(20,10))
    plot_tree(tree, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
    plt.show()

# Visualize the first tree in the forest
plot_tree(best_rf.estimators_[0], wine.feature_names, wine.target_names)

# Function to predict new samples
def predict_new_samples(model, scaler, samples):
    samples_scaled = scaler.transform(samples)
    predictions = model.predict(samples_scaled)
    return [wine.target_names[pred] for pred in predictions]

# Example usage of prediction function
new_samples = np.array([[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820],
                        [12.37, 1.07, 2.1, 18.5, 88, 3.52, 3.75, 0.24, 1.95, 4.5, 1.04, 2.77, 660]])

predictions = predict_new_samples(best_rf, scaler, new_samples)
print("\nPredictions for new samples:")
for i, prediction in enumerate(predictions):
    print(f"Sample {i+1}: Predicted as {prediction}")