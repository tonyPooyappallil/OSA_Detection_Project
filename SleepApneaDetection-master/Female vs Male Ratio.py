# Imported required libraries
import pandas as pd
import matplotlib.pyplot as plt

# To load predictions file
df = pd.read_csv("D:/Dataset/patient_predictions.csv")

# To mormalize and convert into sex values
df["Sex"] = df["Sex"].astype(str).str.upper()  
df["Sex"] = df["Sex"].map({"M": "Male", "F": "Female"})

# To drop any rows that didn't map correctly
df = df.dropna(subset=["Sex"])

# To create pie chart
sex_counts = df["Sex"].value_counts()
sex_counts.plot(kind="pie", autopct="%1.1f%%", startangle=140, labels=sex_counts.index, colors=["lightblue", "lightpink"])
plt.title("Sex Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# To create bar Chart
sex_counts.plot(kind="bar", color=["lightblue", "lightpink"])
plt.title("Number of Male vs Female Patients")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
