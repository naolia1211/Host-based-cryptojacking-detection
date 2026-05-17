import pandas as pd
import numpy as np
from pathlib import Path

print("=== Generating Hard Test Datasets ===")
print("Creating more challenging test sets with realistic overlap and noise.\n")

# Load sample file
sample_path = Path("archive/final-normal-data-set.csv")
if not sample_path.exists():
    print("Error: archive/final-normal-data-set.csv not found.")
    print("Please place the file in the archive folder.")
    exit()

df_sample = pd.read_csv(sample_path, low_memory=False)
numeric_cols = df_sample.select_dtypes(include=['number']).columns.tolist()

means = df_sample[numeric_cols].mean()
stds = df_sample[numeric_cols].std()

print(f"Using {len(numeric_cols)} features with statistics from original data.")

# Generate 5 test files
for i in range(1, 6):
    n_samples = 4000
    
    # Normal data
    normal = pd.DataFrame()
    for col in numeric_cols:
        if col in means.index:
            normal[col] = np.random.normal(means[col] * 0.9, stds[col] * 1.2, n_samples)
        else:
            normal[col] = np.random.normal(0, 10, n_samples)
    
    normal['Label'] = 1
    
    # Cryptojacking data - harder version
    crypto = pd.DataFrame()
    for col in numeric_cols:
        if "cpu_user" in col or "cpu_total" in col or "cpu" in col:
            crypto[col] = np.random.normal(means[col] * 2.8 + 12, stds[col] * 2.1, n_samples)
        elif "cpu_system" in col:
            crypto[col] = np.random.normal(means[col] * 3.2 + 8, stds[col] * 2.3, n_samples)
        elif "mem" in col or "memory" in col:
            crypto[col] = np.random.normal(means[col] * 1.75, stds[col] * 1.9, n_samples)
        elif "load" in col:
            crypto[col] = np.random.normal(means[col] * 4.2, stds[col] * 2.4, n_samples)
        elif "disk" in col or "io" in col:
            crypto[col] = np.random.normal(means[col] * 2.9, stds[col] * 2.6, n_samples)
        else:
            crypto[col] = np.random.normal(means[col] * 2.0, stds[col] * 2.0, n_samples)
    
    crypto['Label'] = 0
    
    # Add noise - FIXED VERSION
    numeric_data = crypto.drop(columns=['Label']).values          # shape: (4000, n_features)
    std_per_feature = numeric_data.std(axis=0)                    # shape: (n_features,)
    
    noise = np.random.normal(0, 0.25, numeric_data.shape)
    noisy_numeric = numeric_data + noise * std_per_feature
    
    # Put back to DataFrame
    crypto_numeric_df = pd.DataFrame(noisy_numeric, columns=numeric_cols)
    crypto = pd.concat([crypto_numeric_df, crypto[['Label']].reset_index(drop=True)], axis=1)
    
    # Combine normal + crypto
    df_test = pd.concat([normal, crypto], ignore_index=True)
    df_test = df_test.sample(frac=1, random_state=42 + i).reset_index(drop=True)
    
    # Save
    filename = f"test_metrics_hard_{i}.csv"
    df_test.to_csv(filename, index=False)
    
    print(f"Generated {filename} | {len(df_test)} samples")

print("\nCompleted! 5 hard test files generated successfully.")
print("Files: test_metrics_hard_1.csv → test_metrics_hard_5.csv")