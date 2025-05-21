# Imported required libraries
import pandas as pd

# To load clinical data
clinical_df = pd.read_csv("D:/Dataset/merged_clinical_data.csv")

# To load signal features
signal_df = pd.read_csv("D:/Dataset/signal_features.csv")

# To merge the record
final_df = pd.merge(clinical_df, signal_df, on="Record", how="inner")

# To save final dataset
final_df.to_csv("D:/Dataset/final_multimodal_dataset.csv", index=False)

print("Final multimodal dataset saved as final_multimodal_dataset.csv")
