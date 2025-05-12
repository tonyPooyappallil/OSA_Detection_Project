import wfdb
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm

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

def get_sleep_stages_from_annotations(annotation, record_header, record_name, sleep_stages_dict):
    """
    Extracts sleep stages from WFDB annotations based on .hea file for duration,
    applying forward-filling for stages.
    """
    stage_symbol_to_code_map = {
        'W': 0, 'SLEEP-STAGE-W': 0, 'WAKE': 0,
        '1': 1, 'SLEEP-STAGE-1': 1, 'N1': 1,
        '2': 2, 'SLEEP-STAGE-2': 2, 'N2': 2,
        '3': 3, 'SLEEP-STAGE-3': 3, 'N3': 3,
        '4': 3, 'SLEEP-STAGE-4': 3, # Map N4 to N3 as per AASM current practice
        'R': 4, 'SLEEP-STAGE-R': 4, 'REM': 4,
        '?': -1, 'SLEEP-STAGE-UNKNOWN': -1, 'UNDEF': -1, 'UNKNOWN': -1
    }
    
    if not annotation.sample.size > 0:
        # print(f"  Debug (get_sleep_stages - {record_name}): No annotation samples found.") # Optional
        return np.array([sleep_stages_dict['undefined']]) 

    fs = record_header.fs
    sig_len = record_header.sig_len
    expected_epoch_duration_seconds = 30.0
    expected_epoch_duration_samples = int(expected_epoch_duration_seconds * fs)

    if expected_epoch_duration_samples == 0:
        print(f"  Error (get_sleep_stages - {record_name}): Epoch duration in samples is 0 (fs={fs}).")
        return np.array([sleep_stages_dict['undefined']])

    num_epochs_total = sig_len // expected_epoch_duration_samples
    
    # print(f"  Debug (get_sleep_stages - {record_name}):") # Optional
    # print(f"    Signal length (samples): {sig_len}, Sampling frequency (Hz): {fs}")
    # print(f"    Calculated epoch duration (samples): {expected_epoch_duration_samples}")
    # print(f"    Total number of 30s epochs in recording: {num_epochs_total}")

    if num_epochs_total == 0:
        # print(f"  Warning (get_sleep_stages - {record_name}): Calculated 0 total epochs.") # Optional
        return np.array([sleep_stages_dict['undefined']])

    sleep_stages_array = np.full(num_epochs_total, sleep_stages_dict['undefined'], dtype=int)
    # print(f"    Initialized sleep_stages_array for {num_epochs_total} epochs with 'undefined' ({sleep_stages_dict['undefined']}).") # Optional

    sorted_annotations = sorted(
        zip(annotation.sample, annotation.symbol, annotation.aux_note),
        key=lambda x: x[0]
    )

    current_stage_code = sleep_stages_dict['undefined'] 
    last_updated_epoch_idx = -1

    for sample, symbol, raw_aux_note in sorted_annotations:
        aux_note = str(raw_aux_note).strip().upper()
        epoch_index = int(sample // expected_epoch_duration_samples)

        if epoch_index >= num_epochs_total:
            continue

        parsed_stage_code = None
        
        for stage_text_map_key, code in stage_symbol_to_code_map.items():
            if f"SLEEP-STAGE-{stage_text_map_key}" == aux_note or stage_text_map_key == aux_note :
                parsed_stage_code = code
                break
        if parsed_stage_code is None and symbol in stage_symbol_to_code_map: # Fallback to symbol
            parsed_stage_code = stage_symbol_to_code_map[symbol]

        if parsed_stage_code is not None:
            # Forward fill from the last updated epoch up to (but not including) the current one
            # Ensure we only fill if current_stage_code is a valid, known stage (not initial undefined if no stage seen yet)
            if last_updated_epoch_idx != -1 and epoch_index > last_updated_epoch_idx : # Ensure we are moving forward
                fill_start = last_updated_epoch_idx + 1 # Start filling from the epoch after the last one updated
                fill_end = epoch_index # Fill up to the current epoch_index
                if fill_start < num_epochs_total:
                    sleep_stages_array[fill_start:min(fill_end, num_epochs_total)] = current_stage_code
            
            # Update current epoch and current_stage_code
            if epoch_index < num_epochs_total:
                sleep_stages_array[epoch_index] = parsed_stage_code
                current_stage_code = parsed_stage_code # This becomes the stage to forward-fill next
                last_updated_epoch_idx = epoch_index

    # After loop, if there are remaining epochs, fill them with the last known stage
    if last_updated_epoch_idx != -1 and last_updated_epoch_idx < num_epochs_total - 1:
        fill_start = last_updated_epoch_idx + 1
        sleep_stages_array[fill_start:] = current_stage_code # Fill remaining with the last valid stage
    
    defined_stages_count = np.sum(sleep_stages_array != sleep_stages_dict['undefined'])
    # if defined_stages_count == 0 : # Optional warning
        # print(f"  Warning (get_sleep_stages - {record_name}): No AASM sleep stages could be parsed/filled. TST might be 0.")
    # else:
        # print(f"    {record_name}: After forward-filling, {defined_stages_count} epochs have defined stages out of {num_epochs_total} total epochs.")
    
    return sleep_stages_array


def calculate_ahi(record_path):
    """Calculates AHI for a single record."""
    record_name = os.path.basename(record_path)
    annotation_file_base = os.path.join(record_path, record_name) 

    unique_event_descriptions_counted = set() 

    sleep_stages_dict = {
        'wake': 0, 'nonrem1': 1, 'nonrem2': 2, 'nonrem3': 3, 'rem': 4, 'undefined': -1
    }

    try:
        # 1. Load record header for signal length and fs
        try:
            record_header = wfdb.rdheader(annotation_file_base)
        except Exception as e_header:
            print(f"Error reading header for {record_name}: {e_header}")
            return None, set()

        # 2. Load event annotations using wfdb
        annotation = wfdb.rdann(annotation_file_base, 'arousal')
        
        # 3. Define apnea/hypopnea patterns
        apnea_hypopnea_patterns = [
            r"resp_centralapnea", 
            r"resp_hypopnea",
            r"resp_obstructiveapnea",
            r"resp_mixedapnea",
            r"mixed apnea" 
        ]
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in apnea_hypopnea_patterns]

        # 4. Get Sleep Stages from .arousal file using header info
        sleep_stages_array = get_sleep_stages_from_annotations(annotation, record_header, record_name, sleep_stages_dict)
        
        # 5. Calculate total sleep time
        epoch_duration_seconds = 30.0
        sleep_epochs_count = np.sum((sleep_stages_array != sleep_stages_dict['wake']) & 
                                    (sleep_stages_array != sleep_stages_dict['undefined']))
        
        total_sleep_time_seconds = sleep_epochs_count * epoch_duration_seconds
        total_sleep_time_hours = total_sleep_time_seconds / 3600.0

        if total_sleep_time_hours <= 0:
            # print(f"  Warning: Total sleep time is zero or negative for {record_name} ({total_sleep_time_hours:.2f} hrs from {sleep_epochs_count} sleep epochs). AHI will be 0 or undefined.")
            return 0.0, unique_event_descriptions_counted

        # 6. Count apnea/hypopnea events (Span-based logic)
        apnea_hypopnea_count = 0
        i = 0
        n_ann = len(annotation.sample)
        while i < n_ann:
            current_description = str(annotation.aux_note[i]).strip()
            
            if current_description.startswith('('):
                tag_content = current_description[1:] 
                is_target_event_start = False
                for pattern in compiled_patterns:
                    if pattern.fullmatch(tag_content): 
                        is_target_event_start = True
                        break
                
                if is_target_event_start:
                    expected_end_tag = tag_content + ')' 
                    found_end = False
                    for j in range(i + 1, n_ann):
                        end_description = str(annotation.aux_note[j]).strip()
                        if end_description == expected_end_tag:
                            apnea_hypopnea_count += 1
                            unique_event_descriptions_counted.add(current_description) 
                            i = j 
                            found_end = True
                            break
            i += 1

        # 7. Calculate AHI
        ahi = apnea_hypopnea_count / total_sleep_time_hours if total_sleep_time_hours > 0 else 0.0
        return ahi, unique_event_descriptions_counted

    except FileNotFoundError as fnf_e:
        print(f"Error: File not found during processing of {record_name}. Details: {fnf_e}")
        return None, set()
    except Exception as e:
        print(f"General error processing {record_path} ({record_name}): {e}")
        import traceback
        traceback.print_exc() 
        return None, set()


def process_all_records(base_dir, save_csv=False):
    """Processes all records in the given base directory."""
    all_results_summary = {
        "No OSA": 0, "Mild OSA": 0, "Moderate OSA": 0, "Severe OSA": 0
    }
    all_ahi_details = [] 
    overall_unique_matched_event_descriptions = set()

    record_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for record_dir_name in tqdm(record_folders, desc="Processing Records for AHI"):
        record_full_path = os.path.join(base_dir, record_dir_name)
        
        ahi, unique_descriptions_for_record = calculate_ahi(record_full_path)
        
        if ahi is not None:
            severity = categorize_osa(ahi)
            all_results_summary[severity] += 1
            all_ahi_details.append({
                "Record": record_dir_name,
                "AHI": ahi,
                "Severity": severity
            })
            overall_unique_matched_event_descriptions.update(unique_descriptions_for_record)
        else:
            print(f"Could not calculate AHI for {record_dir_name}.")

    print("\n--- AHI Results Summary ---")
    for severity, count in all_results_summary.items():
        print(f"{severity}: {count} records")

    print("\n--- Overall Unique Apnea/Hypopnea START Event Descriptions Counted: ---")
    if overall_unique_matched_event_descriptions:
        for idx, desc in enumerate(sorted(list(overall_unique_matched_event_descriptions))):
            print(f"  [{idx}] - '{desc}'")
    else:
        print("  No apnea/hypopnea event start descriptions were counted.")


    if save_csv and all_ahi_details:
        output_df = pd.DataFrame(all_ahi_details)
        # Save CSV in the base_dir (e.g., selected_records208)
        csv_path_in_base = os.path.join(base_dir, "calculated_ahi_results_with_severity.csv")

        output_df.to_csv(csv_path_in_base, index=False)
        print(f"\nResults saved to '{csv_path_in_base}'")
    elif save_csv:
        print("\nNo AHI details to save to CSV.")

# --- Main Execution ---
# IMPORTANT: Update this path to your actual data directory
# This should be the parent directory containing individual record folders (e.g., tr03-0005, tr03-0007, etc.)
training_dir = r"C:\Users\anton\OneDrive\Documents\project data\selected_records" # Example for Colab/Linux

if __name__ == "__main__": # Ensure this block runs only when script is executed directly
    # If running locally on Windows, you might use a path like:
    # training_dir = r"C:\Users\anton\OneDrive\Documents\project data\selected_records208"
    
    # Check if the directory exists before processing
    if os.path.exists(training_dir) and os.path.isdir(training_dir):
        process_all_records(training_dir, save_csv=True)
    else:
        print(f"Error: The specified directory does not exist or is not a directory: {training_dir}")
        print("Please update the 'training_dir' variable with the correct path to your data.")

