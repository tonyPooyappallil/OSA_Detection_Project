# Imported required libraries
import pandas as pd

# To load CSV files
age_sex = pd.read_csv(r"D:\Dataset\age-sex.csv")
ahi_data = pd.read_csv(r"D:\Dataset\ahi_results_with_severity.csv")

# To merge them using record column
merged_data = pd.merge(age_sex, ahi_data, on="Record", how="inner")

# To save the result
merged_data.to_csv("D:/Dataset/merged_clinical_data.csv", index=False)

print("Merged clinical data saved as merged_clinical_data.csv")
