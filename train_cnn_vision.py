import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import joblib

# ====================== PATHS ======================
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_SIZE = 64   # 8x8 image
EPOCHS      = 50
LR          = 0.0008

# ====================== LOAD TRAIN ======================
print("Loading train set...")
train_df     = pd.read_csv(DATA_DIR / "train_crypto_hijacking.csv", low_memory=False)
FEATURE_COLS = [c for c in train_df.columns if c != "label"]

X_train_raw = train_df[FEATURE_COLS].values.astype(np.float32)
y_train_raw = train_df["label"].values.astype(np.float32)

# Fill NaN
means = np.nanmean(X_train_raw, axis=0)
mask  = np.isnan(X_train_raw)
X_train_raw[mask] = np.take(means, np.where(mask)[1])

# Pad to TARGET_SIZE
if X_train_raw.shape[1] < TARGET_SIZE:
    X_train_raw = np.pad(X_train_raw, ((0, 0), (0, TARGET_SIZE - X_train_raw.shape[1])))
else:
    X_train_raw = X_train_raw[:, :TARGET_SIZE]

# ====================== SCALE ======================
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_train_raw)

X_train = torch.FloatTensor(X_scaled.reshape(-1, 1, 8, 8))
y_train = torch.FloatTensor(y_train_raw).unsqueeze(1)

# Class weight thay cho SMOTE
n_neg = (y_train_raw == 0).sum()
n_pos = (y_train_raw == 1).sum()
pos_weight = torch.tensor([n_neg / n_pos * 0.4])  # giảm bias về class anormal
print(f"Class weight (pos/neg): {pos_weight.item():.3f}")

# ====================== MODEL (nhỏ hơn) ======================
class VisionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

model     = VisionCNN()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)

# ====================== TRAIN ======================
print(f"Training CNN ({EPOCHS} epochs)...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {loss.item():.4f}")

# ====================== TRAIN METRICS ======================
model.eval()
with torch.no_grad():
    pred_train = (torch.sigmoid(model(X_train)).squeeze().numpy() > 0.5).astype(int)

y_true = y_train_raw.astype(int)
acc  = accuracy_score(y_true, pred_train) * 100
prec = precision_score(y_true, pred_train, pos_label=1, zero_division=0) * 100
rec  = recall_score(y_true, pred_train,    pos_label=1, zero_division=0) * 100
f1   = f1_score(y_true, pred_train,        pos_label=1, zero_division=0) * 100

print(f"\n[Train]")
print(f"  Accuracy  : {acc:.2f}%")
print(f"  Precision : {prec:.2f}%")
print(f"  Recall    : {rec:.2f}%")
print(f"  F1-Score  : {f1:.2f}%")

# ====================== SAVE ======================
torch.save(model.state_dict(), MODELS_DIR / "cnn_vision_model.pth")
joblib.dump(scaler, MODELS_DIR / "scaler_cnn.pkl")
print("\nModels saved.")