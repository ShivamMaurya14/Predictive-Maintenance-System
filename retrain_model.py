
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Generating synthetic dataset based on notebook description...")

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 10000

# 1. Generate core features based on df.describe() stats and notebook description

# Type: L (60%), M (30%), H (10%)
types = np.random.choice(['L', 'M', 'H'], size=n_samples, p=[0.6, 0.3, 0.1])

# Air temperature [K]: mean ~300, std ~2
air_temp = np.random.normal(300, 2, n_samples)

# Process temperature [K]: mean ~310, std ~1.5
# Notebook says: Air temp + 10 + noise
process_temp = air_temp + 10 + np.random.normal(0, 1, n_samples)

# Rotational speed [rpm]: mean ~1538, std ~179
rot_speed = np.random.normal(1539, 179, n_samples)

# Torque [Nm]: mean ~40, std ~10
torque = np.random.normal(40, 10, n_samples)

# Tool wear [min]: mean ~108, std ~63
tool_wear = np.random.uniform(0, 253, n_samples) # describe shows min 0 max 253, uniform is a reasonable approx for wear over time

# Create DataFrame
df = pd.DataFrame({
    'Type': types,
    'Air temperature [K]': air_temp,
    'Process temperature [K]': process_temp,
    'Rotational speed [rpm]': rot_speed,
    'Torque [Nm]': torque,
    'Tool wear [min]': tool_wear
})

# 2. Feature Engineering

# Temperature Difference
df['temperature_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']

# Mechanical Power [W]
# Formula: Torque * (Rot Speed * 2 * pi / 60)
df['Mechanical Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi / 60

# 3. Generate Target (Machine Failure) based on rules in Cell 53

# Initialize failures
df['Machine failure'] = 0

# Rule 1: Tool Wear Failure (TWF)
# Notebook: replaced between 200-240. 
# Simplified: if tool wear > 200 and random chance
twf_mask = (df['Tool wear [min]'] > 200) & (np.random.random(n_samples) < 0.05) # Arbitrary prob based on ~120 times count

# Rule 2: Heat Dissipation Failure (HDF)
# diff < 8.6 and speed < 1380
hdf_mask = (df['temperature_difference'] < 8.6) & (df['Rotational speed [rpm]'] < 1380)

# Rule 3: Power Failure (PWF)
# Power < 3500 or > 9000
pwf_mask = (df['Mechanical Power [W]'] < 3500) | (df['Mechanical Power [W]'] > 9000)

# Rule 4: Overstrain Failure (OSF)
# Tool wear * Torque > 11000 (L), 12000 (M), 13000 (H)
osf_thresholds = {'L': 11000, 'M': 12000, 'H': 13000}
strain = df['Tool wear [min]'] * df['Torque [Nm]']
osf_mask = pd.Series(False, index=df.index)
for t_type, threshold in osf_thresholds.items():
    osf_mask |= ((df['Type'] == t_type) & (strain > threshold))

# Rule 5: Random Failure (RNF)
# 0.1% chance
rnf_mask = np.random.random(n_samples) < 0.001

# Combine failures
df.loc[twf_mask | hdf_mask | pwf_mask | osf_mask | rnf_mask, 'Machine failure'] = 1

print(f"Dataset generated. Failure count: {df['Machine failure'].sum()}")

# 4. Preprocessing for Training

# Encode Type
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])
print(f"Type mapping classes: {le.classes_}") 
# Expected: H->0, L->1, M->2 (alphabetical) which aligns with app.py logic

# Features and Target
X = df[['Type', 'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
        'temperature_difference', 'Mechanical Power [W]']]
y = df['Machine failure']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.4f}")

# 6. Save Model
joblib.dump(model, 'model.joblib')
print("Model saved to model.joblib")
