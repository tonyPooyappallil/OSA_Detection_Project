import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
df = pd.read_csv("D:/Dataset/final_multimodal_dataset.csv")

# Step 2: Encode categorical features
sex_encoder = LabelEncoder()
df['Sex'] = sex_encoder.fit_transform(df['Sex'])

severity_encoder = LabelEncoder()
df['Severity'] = severity_encoder.fit_transform(df['Severity'])

# Step 3: Prepare features and labels
X = df.drop(columns=["Record", "Severity"])
y = df["Severity"]

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train Logistic Regression with more iterations
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

print(f"Accuracy: {accuracy} %")
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 8: Save predictions
results_df = df.loc[y_test.index].copy()
results_df["Predicted_Severity"] = y_pred

# Decode labels for readability
results_df["Sex"] = sex_encoder.inverse_transform(results_df["Sex"])
results_df["Severity"] = severity_encoder.inverse_transform(results_df["Severity"])
results_df["Predicted_Severity"] = severity_encoder.inverse_transform(results_df["Predicted_Severity"])

results_df.to_csv("D:/Dataset/patient_predictions.csv", index=False)
print("Saved predictions to patient_predictions.csv")
