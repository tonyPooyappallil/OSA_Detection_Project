import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
df = pd.read_csv("D:/Dataset/patient_predictions.csv")

# Normalize and convert sex values
df["Sex"] = df["Sex"].astype(str).str.upper()  # convert m/f to M/F
df["Sex"] = df["Sex"].map({"M": "Male", "F": "Female"})

# Drop any rows that didn't map correctly
df = df.dropna(subset=["Sex"])

# Pie Chart
sex_counts = df["Sex"].value_counts()
sex_counts.plot(kind="pie", autopct="%1.1f%%", startangle=140, labels=sex_counts.index, colors=["lightblue", "lightpink"])
plt.title("Sex Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Bar Chart
sex_counts.plot(kind="bar", color=["lightblue", "lightpink"])
plt.title("Number of Male vs Female Patients")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
