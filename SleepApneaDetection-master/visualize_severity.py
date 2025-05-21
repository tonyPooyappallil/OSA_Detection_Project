# Imported required libraries
import pandas as pd
import matplotlib.pyplot as plt

# To load prediction results from a CSV file
df = pd.read_csv("D:/Dataset/patient_predictions.csv")

# To create number codes for each OSA severity level 
severity_labels = {
    "No OSA": 0,
    "Mild OSA": 1,
    "Moderate OSA": 2,
    "Severe OSA": 3
}
reverse_labels = {v: k for k, v in severity_labels.items()}

# Created if condition for severity is stored as numbers then convert it back to readable labels like Mild OSA
if df["Severity"].dtype != 'object':
    df["Severity"] = df["Severity"].map(reverse_labels)

# Created Bar Chart for Actual Severity
df["Severity"].value_counts().sort_index().plot(kind="bar", color="lightgreen")
plt.title("Actual Severity Distribution")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Created Pie Chart for Actual Severity
df["Severity"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Actual Severity Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()
