import pandas as pd

# Step 1: Load your CSV files
age_sex = pd.read_csv(r"D:\Dataset\age-sex.csv")
ahi_data = pd.read_csv(r"D:\Dataset\ahi_results_with_severity.csv")

# Step 2: Merge them using 'Record' column
merged_data = pd.merge(age_sex, ahi_data, on="Record", how="inner")

# Step 3: Save the result
merged_data.to_csv("D:/Dataset/merged_clinical_data.csv", index=False)

print("Merged clinical data saved as merged_clinical_data.csv")
