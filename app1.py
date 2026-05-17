import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="CryptoJacking CNN Detector", layout="wide")

st.title("CryptoJacking Detection — Vision-based CNN")
st.caption("Advanced Malware Analysis Course — Host-based Cryptojacking Detection")
st.divider()

# ====================== LOAD MODEL ======================
TARGET_SIZE = 64  # 8x8 image

@st.cache_resource
def load_model():
    models_dir = Path("models")
    scaler = joblib.load(models_dir / "scaler_cnn.pkl")

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

    model = VisionCNN()
    model.load_state_dict(torch.load(models_dir / "cnn_vision_model.pth", map_location="cpu"))
    model.eval()
    return model, scaler

try:
    model, scaler = load_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ====================== UPLOAD ======================
st.subheader("Upload Dataset")
st.markdown("Upload validation set (`val_crypto_hijacking.csv`) or test set (`test_crypto_hijacking.csv`).")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# ====================== INFERENCE ======================
if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)

    with st.expander("Data preview", expanded=False):
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"{df.shape[0]:,} rows x {df.shape[1]} columns")

    label_col = "label" if "label" in df.columns else None
    y_true    = df[label_col].values.astype(int) if label_col else None

    feature_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)

    # Fill NaN
    means = np.nanmean(X, axis=0)
    mask  = np.isnan(X)
    X[mask] = np.take(means, np.where(mask)[1])

    # Pad/trim to TARGET_SIZE
    if X.shape[1] < TARGET_SIZE:
        X = np.pad(X, ((0, 0), (0, TARGET_SIZE - X.shape[1])))
    else:
        X = X[:, :TARGET_SIZE]

    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled.reshape(-1, 1, 8, 8))

    with torch.no_grad():
        probs = torch.sigmoid(model(X_tensor)).squeeze().numpy()

    preds = (probs > 0.5).astype(int)

    # ====================== SUMMARY ======================
    num_anormal = int((preds == 1).sum())
    num_normal  = int((preds == 0).sum())
    total       = len(preds)
    pct_anormal = num_anormal / total * 100 if total > 0 else 0

    st.subheader("Detection Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cryptojacking (Anormal)", f"{num_anormal:,} ({pct_anormal:.1f}%)")
    c2.metric("Normal",                  f"{num_normal:,} ({100 - pct_anormal:.1f}%)")
    c3.metric("Total Samples",           f"{total:,}")

    with st.expander("Detailed prediction table", expanded=False):
        df_result = df.copy()
        df_result["Prediction"] = np.where(preds == 1, "Cryptojacking", "Normal")
        df_result["Probability"] = probs.round(4)
        show_cols = ["Prediction", "Probability"] + (["label"] if label_col else [])
        st.dataframe(df_result[show_cols], use_container_width=True)

    # ====================== METRICS ======================
    if y_true is not None:
        st.subheader("Evaluation Metrics")

        acc  = accuracy_score(y_true, preds) * 100
        prec = precision_score(y_true, preds, pos_label=1, zero_division=0) * 100
        rec  = recall_score(y_true, preds,    pos_label=1, zero_division=0) * 100
        f1   = f1_score(y_true, preds,        pos_label=1, zero_division=0) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",            f"{acc:.2f}%")
        m2.metric("Precision (Anormal)", f"{prec:.2f}%")
        m3.metric("Recall (Anormal)",    f"{rec:.2f}%")
        m4.metric("F1-Score",            f"{f1:.2f}%")

        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, preds)

        col_cm, _ = st.columns([1, 2])
        with col_cm:
            fig, ax = plt.subplots(figsize=(3, 2.2))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anormal"],
                yticklabels=["Normal", "Anormal"],
                ax=ax, annot_kws={"size": 11}, cbar=False,
            )
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Actual", fontsize=9)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("No 'label' column found — skipping ground truth comparison.")

st.divider()
st.caption("Vision-based CNN | Advanced Malware Mechanisms Course")