import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from pathlib import Path
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
df_anormal['Label'] = 0
df = pd.concat([df_normal, df_anormal], ignore_index=True)

df = df.fillna(df.mean(numeric_only=True))
numeric_cols = df.select_dtypes(include=['number']).columns
df = df[numeric_cols]

X = df.drop('Label', axis=1).values.astype(np.float32)
y = df['Label'].values.astype(np.float32)

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bal)

# ==================== Train / Validation / Test Split ====================
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y_bal, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

X_train = torch.FloatTensor(X_train)
X_val   = torch.FloatTensor(X_val)
X_test  = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
y_val   = torch.FloatTensor(y_val).unsqueeze(1)
y_test  = torch.FloatTensor(y_test).unsqueeze(1)

# ====================== Model Definition ======================
class CryptoJackingModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.55),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


model = CryptoJackingModel(X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)

# ====================== Training with Early Stopping ======================
print("Training model with early stopping...")

best_val_loss = float('inf')
patience = 15
patience_counter = 0
best_model_state = None

for epoch in range(100):
    # Training phase
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()

    if epoch % 10 == 0 or epoch == 99:
        print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# ====================== Evaluation ======================
print("\nEvaluating on test set...")

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    pred = torch.sigmoid(outputs).round().squeeze().numpy()
    y_true = y_test.squeeze().numpy()

acc = accuracy_score(y_true, pred) * 100
prec = precision_score(y_true, pred) * 100
rec = recall_score(y_true, pred) * 100
f1 = f1_score(y_true, pred) * 100

print("\n=== MODEL PERFORMANCE ===")
print(f"Accuracy   : {acc:.2f}%")
print(f"Precision  : {prec:.2f}%")
print(f"Recall     : {rec:.2f}%")
print(f"F1-score   : {f1:.2f}%")

# ====================== Save Model ======================
torch.save(model.state_dict(), models_dir / "cryptojacking_model.pth")
joblib.dump(scaler, models_dir / "scaler.pkl")

print("\nModel and scaler saved successfully.")
print(f"   - {models_dir}/cryptojacking_model.pth")
print(f"   - {models_dir}/scaler.pkl")