import pandas as pd

# Load clinical data
clinical_df = pd.read_csv("D:/Dataset/merged_clinical_data.csv")

# Load signal features
signal_df = pd.read_csv("D:/Dataset/signal_features.csv")

# Merge on 'Record'
final_df = pd.merge(clinical_df, signal_df, on="Record", how="inner")

# Save final dataset
final_df.to_csv("D:/Dataset/final_multimodal_dataset.csv", index=False)

print("Final multimodal dataset saved as final_multimodal_dataset.csv")
