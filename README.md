# Host-based Cryptojacking Detection

**Đồ án môn NT2202 - Cơ chế hoạt động mã độc nâng cao**

Hệ thống phát hiện mã độc khai thác tiền mã hóa (Cryptojacking) dựa trên host sử dụng system metrics.

## Các mô hình đã implement
- **Neural Network (CryptoJackingModel)** — theo Paper 1 (Sanda et al.)
- **Isolation Forest** — baseline từ Paper 1
- **Vision-based CNN** — lấy ý tưởng từ Paper 2 (Almurshid et al.)

## Kết quả nổi bật
- Neural Network: **Accuracy 99.95%**, Recall **100%**
- Xây dựng Streamlit Application để demo thực tế
- Demo mã độc XMRig (full load & stealth mode) trên Ubuntu VM

## Công nghệ sử dụng
- Python, PyTorch, scikit-learn, pandas, imbalanced-learn
- Streamlit (Web UI)

