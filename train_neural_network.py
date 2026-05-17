import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

# ====================== CONFIG ======================
DATA_DIR = Path('data')
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 512
NUM_EPOCHS = 50          # Giảm epochs
LEARNING_RATE = 0.0003
PATIENCE = 10
DROPOUT = 0.6
# ===================================================

print("=" * 70)
print("TRAINING CRYPTOJACKING NEURAL NETWORK - ANTI-OVERFITTING")
print("=" * 70)

# Load + Fill NaN
train_df = pd.read_csv(DATA_DIR / 'train_crypto_hijacking.csv', low_memory=False)
val_df   = pd.read_csv(DATA_DIR / 'val_crypto_hijacking.csv', low_memory=False)
test_df  = pd.read_csv(DATA_DIR / 'test_crypto_hijacking.csv', low_memory=False)

feature_cols = [col for col in train_df.columns if col not in ['label', 'timestamp']]

train_df[feature_cols] = train_df[feature_cols].fillna(train_df[feature_cols].median())
val_df[feature_cols]   = val_df[feature_cols].fillna(train_df[feature_cols].median())
test_df[feature_cols]  = test_df[feature_cols].fillna(train_df[feature_cols].median())

X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df['label'].values

X_val = val_df[feature_cols].values.astype(np.float32)
y_val = val_df['label'].values

X_test = test_df[feature_cols].values.astype(np.float32)
y_test = test_df['label'].values

# SMOTE only on train
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Scale
scaler = StandardScaler()
X_train_bal = scaler.fit_transform(X_train_bal)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')

# Tensor
X_train_t = torch.FloatTensor(X_train_bal)
y_train_t = torch.FloatTensor(y_train_bal).unsqueeze(1)
X_val_t   = torch.FloatTensor(X_val)
y_val_t   = torch.FloatTensor(y_val).unsqueeze(1)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)

# ====================== MODEL (Stronger Regularization) ======================
class CryptoJackingModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 96),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        return self.net(x)


model = CryptoJackingModel(X_train_t.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ====================== Training ======================
print("\nStarting training...")

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()

    if epoch % 5 == 0 or epoch == NUM_EPOCHS-1:
        print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

if best_model_state:
    model.load_state_dict(best_model_state)

# Save
torch.save(model.state_dict(), MODEL_DIR / "cryptojacking_neural_network.pth")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

# ====================== Evaluation ======================
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    pred = torch.sigmoid(outputs).round().squeeze().numpy()
    y_true = y_test_t.squeeze().numpy()

print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE ON TEST SET")
print("="*60)
print(classification_report(y_true, pred, target_names=['Normal', 'Crypto Hijacking'], digits=4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, pred))