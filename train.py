
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=========================================")
print("   Model Training Pipeline    ")
print("=========================================")
print("[1/3] Loading dataset...")

# Load dataset
try:
    df = pd.read_csv('dataset/ai4i2020.csv')
    print(f"Dataset loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'dataset/ai4i2020.csv' not found. Please ensure the dataset exists.")
    exit(1)

# 2. Preprocessing & Feature Engineering

print("[2/3] Preprocessing and Feature Engineering...")

# Encode 'Type' column to match app.py logic: H->0, L->1, M->2
type_mapping = {'H': 0, 'L': 1, 'M': 2}
df['Type'] = df['Type'].map(type_mapping)

# Calculate derived features
# Temperature Difference
df['temperature_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']

# Mechanical Power [W]
# Formula: Torque * (Rot Speed * 2 * pi / 60)
df['Mechanical Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi / 60

# Select Features and Target
# Features must match those expected by app.py
feature_cols = [
    'Type', 
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]', 
    'temperature_difference', 
    'Mechanical Power [W]'
]

X = df[feature_cols]
y = df['Machine failure']

print(f"Features selected: {feature_cols}")
print(f"Target distribution:\n{y.value_counts()}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
print("[3/3] Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.4f}")

# 4. Save Model
model_path = 'models/model.joblib'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
print("=========================================")
