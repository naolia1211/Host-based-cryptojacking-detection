import pandas as pd
import numpy as np
from pathlib import Path

print("=== Generating Test Datasets for Cryptojacking Demo ===")
print("Sử dụng cấu trúc 58 features từ dataset Kaggle\n")

# Đọc file mẫu để lấy đúng danh sách cột numeric
sample_path = Path("archive/final-normal-data-set.csv")
if not sample_path.exists():
    print("❌ Không tìm thấy file archive/final-normal-data-set.csv")
    print("Vui lòng copy file final-normal-data-set.csv vào thư mục archive/")
    exit()

df_sample = pd.read_csv(sample_path, low_memory=False)
numeric_cols = df_sample.select_dtypes(include=['number']).columns.tolist()

print(f"✅ Sử dụng đúng {len(numeric_cols)} numeric features từ dataset Kaggle")

# Tạo 5 file test
for i in range(1, 6):
    n_samples = 4000  # tổng 4000 dòng mỗi file
    
    # Normal data (thấp)
    normal = pd.DataFrame()
    for col in numeric_cols:
        if "cpu_user" in col or "cpu_total" in col:
            normal[col] = np.random.normal(10, 5, n_samples)
        elif "cpu_system" in col:
            normal[col] = np.random.normal(3, 2, n_samples)
        elif "mem_percent" in col or "mem_used" in col:
            normal[col] = np.random.normal(45, 12, n_samples)
        elif "load" in col:
            normal[col] = np.random.normal(0.8, 0.4, n_samples)
        elif "disk" in col:
            normal[col] = np.random.normal(1200000, 600000, n_samples)
        else:
            normal[col] = np.random.normal(0, 10, n_samples)
    
    normal['Label'] = 1
    
    # Cryptojacking data (cao)
    crypto = pd.DataFrame()
    for col in numeric_cols:
        if "cpu_user" in col or "cpu_total" in col:
            crypto[col] = np.random.normal(78, 12, n_samples)
        elif "cpu_system" in col:
            crypto[col] = np.random.normal(18, 7, n_samples)
        elif "mem_percent" in col or "mem_used" in col:
            crypto[col] = np.random.normal(82, 9, n_samples)
        elif "load" in col:
            crypto[col] = np.random.normal(5.5, 1.8, n_samples)
        elif "disk" in col:
            crypto[col] = np.random.normal(4500000, 1500000, n_samples)
        else:
            crypto[col] = np.random.normal(30, 20, n_samples)
    
    crypto['Label'] = 0
    
    # Kết hợp và shuffle
    df_test = pd.concat([normal, crypto], ignore_index=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    
    # Lưu file
    filename = f"test_metrics_{i}.csv"
    df_test.to_csv(filename, index=False)
    
    print(f"✅ Tạo thành công {filename} | {len(df_test)} samples | Cryptojacking ~50%")

print("\n🎉 Hoàn tất! Bạn có 5 file test_metrics_1.csv đến test_metrics_5.csv")
print("   Khuyến nghị: Upload file test_metrics_*.csv vào Streamlit app để demo.")