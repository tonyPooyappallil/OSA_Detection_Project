import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

# Only generate images for these key signals
TARGET_SIGNALS = ['AIRFLOW', 'SaO2', 'ECG']

# Path to the training data folder (one step back)
training_root = os.path.abspath(os.path.join(os.getcwd(), '..', 'training'))

# Output directory
output_dir = os.path.join(os.getcwd(), 'signal_images')
os.makedirs(output_dir, exist_ok=True)

# Function to process a single folder
def process_folder(folder):
    folder_path = os.path.join(training_root, folder)
    if os.path.isdir(folder_path):
        mat_file = os.path.join(folder_path, f"{folder}.mat")
        if os.path.exists(mat_file):
            try:
                start_time = time.time()  # Time the processing of this folder
                
                mat_data = scipy.io.loadmat(mat_file)
                
                signal_data = mat_data.get('val')  # Checking for 'val' key
                print(f"Keys in {mat_file}: {mat_data.keys()}")
                print(f"Contents of 'val': {signal_data}")
                
                if signal_data is None:
                    print(f"Skipping {folder} (missing signal data)")
                    return
                
                # Assuming signal_data is structured such that we can extract and plot it
                signal_names = ['AIRFLOW', 'SaO2', 'ECG']  # Update with actual signal names if different

                for i, signal_name in enumerate(signal_names):
                    if signal_name in TARGET_SIGNALS:
                        data = signal_data[:, i]  # Get the data for the current signal
                        plt.figure(figsize=(8, 2))
                        plt.plot(data, linewidth=0.5)
                        plt.title(f"{folder} - {signal_name}")
                        plt.axis('off')
                        image_name = f"{folder}_{signal_name}.png"
                        image_path = os.path.join(output_dir, image_name)
                        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        print(f"Saved: {image_path}")
                
                end_time = time.time()  # Time the end of processing this folder
                print(f"Processed {folder} in {end_time - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error processing {mat_file}: {e}")

# Main function for parallel processing
def main():
    start_time = time.time()  # Overall time measurement
    
    # Get list of folders
    folders = [f for f in os.listdir(training_root) if os.path.isdir(os.path.join(training_root, f))]
    
    # Use ProcessPoolExecutor to run the processing in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:  # Using 4 workers to match your 4-core CPU
        executor.map(process_folder, folders)

    end_time = time.time()  # Overall time measurement end
    print(f"Total time taken for processing: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
