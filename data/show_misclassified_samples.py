import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path

# ====================== LOAD MODEL & DATA ======================
MODEL_PATH = Path("models/cryptojacking_neural_network.pth")
SCALER_PATH = Path("models/scaler.pkl")
TEST_PATH = Path("data/test_crypto_hijacking.csv")

# Load model
class CryptoJackingModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 96), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(96, 48), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(48, 24), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(24, 1)
        )
    def forward(self, x):
        return self.net(x)

model = CryptoJackingModel(58)  # điều chỉnh nếu input_size khác
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

scaler = joblib.load(SCALER_PATH)

# Load test data
test_df = pd.read_csv(TEST_PATH)
feature_cols = [col for col in test_df.columns if col not in ['label', 'timestamp']]

X_test = test_df[feature_cols].values.astype(np.float32)
X_test_scaled = scaler.transform(X_test)
y_true = test_df['label'].values

# ====================== DỰ ĐOÁN ======================
with torch.no_grad():
    outputs = model(torch.FloatTensor(X_test_scaled))
    probs = torch.sigmoid(outputs).squeeze().numpy()
    y_pred = (probs >= 0.5).astype(int)

# ====================== TÌM MẪU SAI ======================
misclassified = test_df.copy()
misclassified['Probability'] = probs
misclassified['Predicted'] = y_pred
misclassified['Correct'] = y_true == y_pred

# False Negative: Thực tế là Cryptojacking (1) nhưng model đoán Normal (0)
fn = misclassified[(y_true == 1) & (y_pred == 0)]

# False Positive: Thực tế là Normal (0) nhưng model đoán Cryptojacking (1)
fp = misclassified[(y_true == 0) & (y_pred == 1)]

print("="*80)
print("🔍 CÁC MẪU BỊ PHÁT HIỆN SAI")
print("="*80)

print(f"\n False Negative (Miss Cryptojacking): {len(fn)} mẫu")
if not fn.empty:
    print(fn[['Probability'] + feature_cols[:8]].round(4))  # show vài cột quan trọng

print(f"\n  False Positive (Báo động nhầm): {len(fp)} mẫu")
if not fp.empty:
    print(fp[['Probability'] + feature_cols[:8]].round(4))

# Lưu ra file để dễ copy vào slide
misclassified[misclassified['Correct'] == False].to_csv('misclassified_samples.csv', index=False)
print("\n Đã lưu chi tiết tất cả mẫu sai vào: misclassified_samples.csv")