import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Create a DataFrame for easier handling
df = pd.DataFrame(X, columns=boston.feature_names)
df['PRICE'] = y

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the regression model
model = LinearRegression()

# Perform k-Fold Cross-Validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Lists to store results
mse_scores = []
r2_scores = []
fold_indices = []

# Perform cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X_scaled), 1):
    # Split the data
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Store results
    mse_scores.append(mse)
    r2_scores.append(r2)
    fold_indices.append(fold)
    
    print(f"Fold {fold}: MSE = {mse:.4f}, R2 = {r2:.4f}")

# Calculate average scores
avg_mse = np.mean(mse_scores)
avg_r2 = np.mean(r2_scores)

print(f"\nAverage MSE: {avg_mse:.4f}")
print(f"Average R2: {avg_r2:.4f}")

# Visualize the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(fold_indices, mse_scores)
plt.axhline(y=avg_mse, color='r', linestyle='--', label='Average MSE')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('MSE Scores Across Folds')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(fold_indices, r2_scores)
plt.axhline(y=avg_r2, color='r', linestyle='--', label='Average R2')
plt.xlabel('Fold')
plt.ylabel('R2 Score')
plt.title('R2 Scores Across Folds')
plt.legend()

plt.tight_layout()
plt.show()

# Function to make predictions on new data
def predict_new_data(new_data):
    scaled_data = scaler.transform(new_data)
    return model.predict(scaled_data)

# Example usage
new_sample = np.array([[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.0900, 1, 296.0, 15.3, 396.90, 4.98]])
prediction = predict_new_data(new_sample)
print(f"\nPredicted price for new sample: ${prediction[0]:.2f}")

# Feature importance
feature_importance = abs(model.coef_)
feature_names = boston.feature_names

plt.figure(figsize=(12, 6))
plt.bar(feature_names, feature_importance)
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.title('Feature Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()