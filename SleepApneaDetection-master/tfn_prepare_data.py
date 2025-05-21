# Imported required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Created function to load data, clean the sex column and turn Sex and Severity into numbers for machine learning
def prepare_data(path):
    df = pd.read_csv(path)

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(str).str.upper().replace("M", "M").replace("F", "F")

    sex_encoder = LabelEncoder()
    df["Sex"] = sex_encoder.fit_transform(df["Sex"])

    severity_encoder = LabelEncoder()
    df["Severity"] = severity_encoder.fit_transform(df["Severity"])

    clinical_features = df[["Age", "Sex", "AHI"]].values
    signal_features = df[["Signal_Mean", "Signal_Std", "Signal_Max", "Signal_Min"]].values
    target = df["Severity"].values

    return clinical_features, signal_features, target, sex_encoder, severity_encoder
