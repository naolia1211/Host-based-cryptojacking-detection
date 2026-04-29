import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from pathlib import Path
import joblib

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print("=" * 70)
print("CRYPTOJACKING DETECTION - VISION-BASED CNN MODEL")
print("Inspired by Paper 2: A Holistic Intelligent CryptoJacking Malware Detection System")
print("=" * 70)
print("Dataset: Kaggle Cryptojacking (final-normal-data-set + final-anormal-data-set)\n")

base_dir = Path(__file__).parent
normal_path = base_dir / "archive" / "final-normal-data-set.csv"
anormal_path = base_dir / "archive" / "final-anormal-data-set.csv"

df_normal = pd.read_csv(normal_path, low_memory=False)
df_anormal = pd.read_csv(anormal_path, low_memory=False)

df_normal['Label'] = 1
df_anormal['Label'] = 0
df = pd.concat([df_normal, df_anormal], ignore_index=True)

df = df.fillna(df.mean(numeric_only=True))
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

print(f"Number of numeric features detected: {len(numeric_cols)}")

# Pad to 64 features (8x8 image)
X = df[numeric_cols].values.astype(np.float32)
target_size = 64
if X.shape[1] < target_size:
    pad_width = target_size - X.shape[1]
    X = np.pad(X, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    print(f"Padding {pad_width} zero columns to create 8x8 grayscale image representation")

y = df['Label'].values.astype(np.float32)

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bal)

# Reshape to 8x8 image
X_image = X_scaled.reshape(-1, 1, 8, 8)

X_train, X_test, y_train, y_test = train_test_split(X_image, y_bal, test_size=0.3, random_state=42)

X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
y_test  = torch.FloatTensor(y_test).unsqueeze(1)

# ====================== MODEL ARCHITECTURE ======================
class VisionCryptoJackingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

model = VisionCryptoJackingCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n=== TRAINING PHASE ===")
for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0 or epoch == 29:
        print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f}")

# ====================== EVALUATION ======================
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    pred = torch.sigmoid(outputs).round().squeeze().numpy()
    y_true = y_test.squeeze().numpy()

acc = accuracy_score(y_true, pred) * 100
prec = precision_score(y_true, pred) * 100
rec = recall_score(y_true, pred) * 100
f1 = f1_score(y_true, pred) * 100

print("\n=== FINAL EVALUATION RESULTS ===")
print(f"Accuracy   : {acc:.2f}%")
print(f"Precision  : {prec:.2f}%")
print(f"Recall     : {rec:.2f}%")
print(f"F1-score   : {f1:.2f}%")
print("\nModel training completed successfully.")

torch.save(model.state_dict(), models_dir / "cnn_vision_model.pth")
joblib.dump(scaler, models_dir / "scaler.pkl")

print("\nModel files have been saved:")
print(f"   • {models_dir.absolute()}/cnn_vision_model.pth")
print(f"   • {models_dir.absolute()}/scaler.pkl")