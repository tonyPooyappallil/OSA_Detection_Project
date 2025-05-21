# Imported required libraries
import os
import scipy.io
import pandas as pd
import numpy as np

# To set the main folder location and create an empty list to store signal features
base_path = r"D:\Dataset\training"
signal_features = []

# Created for loop through each patient folder
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    mat_file_path = os.path.join(folder_path, f"{folder_name}.mat")
    
    if os.path.exists(mat_file_path):
        try:
            # To load .mat file
            mat = scipy.io.loadmat(mat_file_path)

            # To find the correct key containing signal data 
            signal_key = [k for k in mat.keys() if not k.startswith('__')][0]
            signal_data = mat[signal_key].flatten()

            # To extract basic features
            features = {
                'Record': folder_name,
                'Signal_Mean': np.mean(signal_data),
                'Signal_Std': np.std(signal_data),
                'Signal_Max': np.max(signal_data),
                'Signal_Min': np.min(signal_data),
            }

            signal_features.append(features)
        except Exception as e:
            print(f"Failed to process {folder_name}: {e}")

# To save features to CSV
signal_df = pd.DataFrame(signal_features)
signal_df.to_csv("D:/Dataset/signal_features.csv", index=False)
print("Signal features saved to signal_features.csv")
