import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="CryptoJacking Detector", layout="wide", page_icon="🛡️")

st.title("🛡️ CryptoJacking Detection System")
st.markdown("**Advanced Malware Analysis Course** — Host-based Cryptojacking Detection")

st.info("""
This application detects host-based cryptojacking malware using system metrics (CPU, Memory, Load, etc.).
Model: Neural Network (CryptoJackingModel from Paper 1)
""")

# ====================== LOAD NEURAL NETWORK ======================
@st.cache_resource
def load_model():
    models_dir = Path("models")
    scaler = joblib.load(models_dir / "scaler.pkl")
    
    state_dict = torch.load(models_dir / "cryptojacking_model.pth", map_location=torch.device('cpu'))
    input_size = state_dict['net.0.weight'].shape[1]
    
    class CryptoJackingNN(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.45),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x)
    
    model = CryptoJackingNN(input_size)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler, input_size

model, scaler, input_size = load_model()

st.subheader("📤 Upload System Metrics CSV")
uploaded_file = st.file_uploader("Choose CSV file (normal, complete, or any system metrics file)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.write("**Data Preview**", df.head())
    
    # Prepare features
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    X = df[numeric_cols].values.astype(np.float32)
    
    # Match scaler features
    expected = scaler.n_features_in_
    if X.shape[1] > expected:
        X = X[:, :expected]
    elif X.shape[1] < expected:
        pad = expected - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad)), mode='constant')
    
    X_scaled = scaler.transform(X)
    
    # Predict
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_scaled))
        probs = torch.sigmoid(outputs).squeeze().numpy()
    
    preds = (probs > 0.5).astype(int)
    
    # Summary
    num_crypto = (preds == 0).sum()
    num_normal = (preds == 1).sum()
    total = len(preds)
    percent_crypto = (num_crypto / total * 100) if total > 0 else 0
    
    st.subheader("📊 Detection Summary")
    col1, col2 = st.columns(2)
    col1.metric("🔴 Cryptojacking Detected", f"{num_crypto} samples ({percent_crypto:.1f}%)")
    col2.metric("🟢 Normal", f"{num_normal} samples ({100 - percent_crypto:.1f}%)")
    
    # Detailed table
    with st.expander("Show Detailed Prediction Table", expanded=False):
        df_result = df.copy()
        df_result['Prediction'] = np.where(preds == 1, "🟢 Normal", "🔴 Cryptojacking")
        df_result['Probability'] = probs.round(4)
        st.dataframe(df_result[['Prediction', 'Probability']], use_container_width=True)
    
    # Ground truth comparison
    if 'Label' in df.columns:
        st.subheader("📊 Comparison with Ground Truth")
        y_true = df['Label'].values
        y_pred = preds
        acc = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, pos_label=0) * 100
        rec = recall_score(y_true, y_pred, pos_label=0) * 100
        f1 = f1_score(y_true, y_pred, pos_label=0) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.2f}%")
        col2.metric("Precision (Cryptojacking)", f"{prec:.2f}%")
        col3.metric("Recall (Cryptojacking)", f"{rec:.2f}%")
        col4.metric("F1-score", f"{f1:.2f}%")

st.caption("Demo for Advanced Malware Mechanisms Course | Neural Network implemented from Paper 1")