# Imported required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# To load patient prediction data from CSV file into a DataFrame
df = pd.read_csv("D:/Dataset/patient_predictions.csv")

# To assign number to each sleep apnea severity level for easier processing
severity_labels = {
    "No OSA": 0,
    "Mild OSA": 1,
    "Moderate OSA": 2,
    "Severe OSA": 3
}
reverse_labels = {v: k for k, v in severity_labels.items()}

# Created if condition to convert severity values from numbers back to labels if needed then get all unique severity types in order
if df["Severity"].dtype != 'object':
    df["Severity"] = df["Severity"].map(reverse_labels)
if df["Predicted_Severity"].dtype != 'object':
    df["Predicted_Severity"] = df["Predicted_Severity"].map(reverse_labels)

labels = sorted(df["Severity"].unique())

# To create confusion matrix
cm = confusion_matrix(df["Severity"], df["Predicted_Severity"], labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix: Actual vs Predicted Severity")
plt.xlabel("Predicted Severity")
plt.ylabel("Actual Severity")
plt.tight_layout()
plt.show()

# To count how many patients fall into each severity level
actual_counts = df["Severity"].value_counts().sort_index()
predicted_counts = df["Predicted_Severity"].value_counts().sort_index()
combined_df = pd.DataFrame({
    "Actual": actual_counts,
    "Predicted": predicted_counts
}).fillna(0)

# To create bar chart
combined_df.plot(kind="bar", figsize=(10, 6))
plt.title("Severity Distribution: Actual vs Predicted")
plt.xlabel("Severity Level")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
