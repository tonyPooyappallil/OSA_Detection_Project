import wfdb
import pandas as pd
import h5py
import numpy as np
import re
import os

# Define AHI severity thresholds
def categorize_osa(ahi):
    """Categorizes the OSA severity based on AHI value."""
    if ahi < 5:
        return "No OSA"
    elif 5 <= ahi < 15:
        return "Mild OSA"
    elif 15 <= ahi < 30:
        return "Moderate OSA"
    else:
        return "Severe OSA"

def calculate_ahi(record_path):
    """Calculates AHI for a single record."""
    try:
        record_name = os.path.basename(record_path)
        annotation_file = os.path.join(record_path, record_name)
        mat_file = os.path.join(record_path, f"{record_name}-arousal.mat")

        # 1. Load annotations using wfdb
        annotation = wfdb.rdann(annotation_file, 'arousal')
        df_events = pd.DataFrame({
            'Sample': annotation.sample,
            'Symbol': annotation.symbol,
            'Description': annotation.aux_note
        })

        # --- Get Unique Descriptions ---
        unique_descriptions = df_events['Description'].unique()
        print(f"\nUnique Descriptions for {record_name}:")
        for desc in unique_descriptions:
            print(f"  - {desc}")
        # --- End Unique Descriptions ---

        # 2. Define apnea/hypopnea patterns
        apnea_hypopnea_patterns = [
            r"\(?resp_centralapnea\)?",
            r"\(?resp_hypopnea\)?",
            r"\(?resp_obstructiveapnea\)?",
            r"\(?resp_mixedapnea\)?",
            r"mixed apnea"
        ]
        compiled_patterns = [re.compile(pattern) for pattern in apnea_hypopnea_patterns]

        # 3. Load sleep stages from .mat file
        sleep_stages_dict = {
            'wake': 0,
            'nonrem1': 1,
            'nonrem2': 2,
            'nonrem3': 3,
            'rem': 4,
            'undefined': -1
        }

        sleep_stages = None  # Initialize sleep_stages

        with h5py.File(mat_file, 'r') as f:
            try:
                stage_arrays = []
                for stage, stage_code in sleep_stages_dict.items():
                    stage_data = np.array(f['data']['sleep_stages'][stage])
                    stage_arrays.append(stage_data)

                sleep_stages_matrix = np.concatenate(stage_arrays, axis=0)

                # --- Validate sleep stages matrix shape ---
                print(f"Sleep stages matrix shape for {record_name}: {sleep_stages_matrix.shape}")
                expected_num_stages = 6
                if sleep_stages_matrix.shape[0] != expected_num_stages:
                    print(f"  Warning: Expected {expected_num_stages} sleep stages, but got {sleep_stages_matrix.shape[0]} for {record_name}")

                sleep_stage_indices = np.argmax(sleep_stages_matrix, axis=0)
                sleep_stages = np.where(np.all(sleep_stages_matrix == 0, axis=0), sleep_stages_dict['undefined'], sleep_stage_indices)
                sample_rate = 200
            except KeyError:
                print(f"  Warning: 'data/sleep_stages' not found in {mat_file}")
                return None, {}
            except Exception as e:
                print(f"  Error loading sleep stages from {mat_file}: {e}")
                return None, {}

        if sleep_stages is None:
            return None, {}

        # 4. Calculate total sleep time (exclude wake and undefined)
        sleep_samples = np.sum((sleep_stages != 0) & (sleep_stages != -1))
        total_sleep_time_seconds = sleep_samples / sample_rate
        total_sleep_time_hours = total_sleep_time_seconds / 3600.0

        # 5. Count apnea/hypopnea events (case-insensitive match)
        apnea_hypopnea_count = 0
        for description in df_events['Description']:
            for pattern in compiled_patterns:
                if pattern.search(description.lower()):  # Case-insensitive match
                    apnea_hypopnea_count += 1
                    break

        # 6. Calculate AHI
        ahi = apnea_hypopnea_count / total_sleep_time_hours if total_sleep_time_hours > 0 else 0.0

        return ahi, unique_descriptions

    except Exception as e:
        print(f"Error processing {record_path}: {e}")
        return None, {}


def process_all_records(base_dir, save_csv=False):
    """Processes all records in the given base directory."""
    all_results = {
        "No OSA": 0,
        "Mild OSA": 0,
        "Moderate OSA": 0,
        "Severe OSA": 0
    }

    all_ahi_results = {}

    for record_dir in os.listdir(base_dir):
        record_path = os.path.join(base_dir, record_dir)
        if os.path.isdir(record_path):
            ahi, unique_descriptions = calculate_ahi(record_path)
            if ahi is not None:
                severity = categorize_osa(ahi)
                all_results[severity] += 1  # Increment the appropriate severity category
                all_ahi_results[record_dir] = {
                    "AHI": ahi,
                    "Descriptions": unique_descriptions,
                    "Severity": severity
                }

    # Print summary results
    print("\n--- AHI Results Summary ---")
    for severity, count in all_results.items():
        print(f"{severity}: {count} records")

    # Print detailed results
    print("\n--- AHI Results for All Records ---")
    for record, data in all_ahi_results.items():
        print(f"{record}: AHI = {data['AHI']:.2f}, Severity = {data['Severity']}")

    # Optional: Save results to CSV
    if save_csv:
        output_df = pd.DataFrame([
            {"Record": rec, "AHI": data["AHI"], "Severity": data["Severity"]}
            for rec, data in all_ahi_results.items()
        ])
        output_df.to_csv("ahi_results_with_severity.csv", index=False)
        print("\nResults saved to 'ahi_results_with_severity.csv'")

# --- Main Execution ---
training_dir = r"C:\Users\anton\OneDrive\Documents\project data\training"

process_all_records(training_dir, save_csv=True)
