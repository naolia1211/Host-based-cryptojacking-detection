import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print("=== Loading Kaggle dataset ===")
base_dir = Path(__file__).parent
normal_path = base_dir / "archive" / "final-normal-data-set.csv"
anormal_path = base_dir / "archive" / "final-anormal-data-set.csv"

df_normal = pd.read_csv(normal_path, low_memory=False)
df_anormal = pd.read_csv(anormal_path, low_memory=False)

df_normal['Label'] = 1
df_anormal['Label'] = -1

df = pd.concat([df_normal, df_anormal], ignore_index=True)

df = df.fillna(df.mean(numeric_only=True))
numeric_cols = df.select_dtypes(include=['number']).columns
df = df[numeric_cols]

X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================== Isolation Forest ======================
print("Training Isolation Forest model...")

iso_forest = IsolationForest(
    n_estimators=150,
    contamination=0.15,
    max_samples=0.8,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_scaled)

pred = iso_forest.predict(X_test_scaled)

pred_mapped = np.where(pred == 1, 1, 0)
y_test_mapped = np.where(y_test == 1, 1, 0)

acc = accuracy_score(y_test_mapped, pred_mapped) * 100
prec = precision_score(y_test_mapped, pred_mapped, pos_label=0) * 100
rec = recall_score(y_test_mapped, pred_mapped, pos_label=0) * 100
f1 = f1_score(y_test_mapped, pred_mapped, pos_label=0) * 100

print("\n=== ISOLATION FOREST PERFORMANCE ===")
print(f"Accuracy   : {acc:.2f}%")
print(f"Precision  : {prec:.2f}%")
print(f"Recall     : {rec:.2f}%")
print(f"F1-score   : {f1:.2f}%")

joblib.dump(iso_forest, models_dir / "isolation_forest_model.pkl")
joblib.dump(scaler, models_dir / "scaler.pkl")

print("\nModel and scaler saved successfully.")
print(f"   - {models_dir}/isolation_forest_model.pkl")
print(f"   - {models_dir}/scaler.pkl")