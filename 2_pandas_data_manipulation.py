import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Data cleaning
df.dropna(inplace=True)  # Remove rows with missing values
df = pd.get_dummies(df, columns=['categorical_column'])  # One-hot encode categorical variables

# Feature selection
X = df.drop('target_column', axis=1)
y = df['target_column']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Extract insights
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
print(feature_importance.sort_values('importance', ascending=False).head(10))