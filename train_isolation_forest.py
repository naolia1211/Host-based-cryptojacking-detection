import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ====================== PATHS ======================
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_crypto_hijacking.csv"

# ====================== LOAD TRAIN ONLY ======================
print("Loading train set...")
train_df     = pd.read_csv(TRAIN_PATH, low_memory=False)
FEATURE_COLS = [c for c in train_df.columns if c != "label"]

X_train = train_df[FEATURE_COLS].values.astype(np.float32)
y_train = train_df["label"].values

# ====================== SCALE ======================
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ====================== TRAIN ======================
n_anormal     = (y_train == 1).sum()
contamination = round(n_anormal / len(y_train), 4)

print(f"Contamination: {contamination:.4f} | Training Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=150,
    contamination=contamination,
    max_samples=0.8,
    random_state=42,
    n_jobs=-1,
)
iso_forest.fit(X_train_scaled)

# ====================== TRAIN METRICS ======================
raw_pred    = iso_forest.predict(X_train_scaled)
pred_train  = np.where(raw_pred == -1, 1, 0)

acc  = accuracy_score(y_train, pred_train) * 100
prec = precision_score(y_train, pred_train, pos_label=1, zero_division=0) * 100
rec  = recall_score(y_train, pred_train,    pos_label=1, zero_division=0) * 100
f1   = f1_score(y_train, pred_train,        pos_label=1, zero_division=0) * 100

print(f"\n[Train]")
print(f"  Accuracy  : {acc:.2f}%")
print(f"  Precision : {prec:.2f}%")
print(f"  Recall    : {rec:.2f}%")
print(f"  F1-Score  : {f1:.2f}%")

# ====================== SAVE ======================
joblib.dump(iso_forest, MODELS_DIR / "isolation_forest_model.pkl")
joblib.dump(scaler,     MODELS_DIR / "scaler_iso.pkl")
print("\nModels saved.")