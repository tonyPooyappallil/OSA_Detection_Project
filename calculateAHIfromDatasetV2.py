# Imported libraries to handle medical signal data
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm

# To define the AHI score into a simple sleep apnea severity label
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
# To reads sleep info from a sleep record and convert into numbered stages 
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
    # This condition created for if there no sleep data is found then it just return "undefined".
    if not annotation.sample.size > 0:
        # print(f"  Debug (get_sleep_stages - {record_name}): No annotation samples found.") # Optional
        return np.array([sleep_stages_dict['undefined']]) 
    # To find out how many data points are in 30 seconds
    fs = record_header.fs
    sig_len = record_header.sig_len
    expected_epoch_duration_seconds = 30.0
    expected_epoch_duration_samples = int(expected_epoch_duration_seconds * fs)
  
    # Created condition if 30 seconds has 0 data points then it will show error and return "undefined", otherwise count how many 30-second blocks there are.
    if expected_epoch_duration_samples == 0:
        print(f"  Error (get_sleep_stages - {record_name}): Epoch duration in samples is 0 (fs={fs}).")
        return np.array([sleep_stages_dict['undefined']])

    num_epochs_total = sig_len // expected_epoch_duration_samples
    
    # print(f"  Debug (get_sleep_stages - {record_name}):") # Optional
    # print(f"    Signal length (samples): {sig_len}, Sampling frequency (Hz): {fs}")
    # print(f"    Calculated epoch duration (samples): {expected_epoch_duration_samples}")
    # print(f"    Total number of 30s epochs in recording: {num_epochs_total}")
    
    # Created condition if there are no 30-second blocks then it will return "undefined".
    if num_epochs_total == 0:
        # print(f"  Warning (get_sleep_stages - {record_name}): Calculated 0 total epochs.") # Optional
        return np.array([sleep_stages_dict['undefined']])
    # Created an array with "undefined" for each 30-second block, then sort the annotations by time.
    sleep_stages_array = np.full(num_epochs_total, sleep_stages_dict['undefined'], dtype=int)
    # print(f"    Initialized sleep_stages_array for {num_epochs_total} epochs with 'undefined' ({sleep_stages_dict['undefined']}).") # Optional

    sorted_annotations = sorted(
        zip(annotation.sample, annotation.symbol, annotation.aux_note),
        key=lambda x: x[0]
    )

    current_stage_code = sleep_stages_dict['undefined'] 
    last_updated_epoch_idx = -1
   
    # Created for loop to go through each annotation and clean the note to find which 30-sec block it belongs to.
    for sample, symbol, raw_aux_note in sorted_annotations:
        aux_note = str(raw_aux_note).strip().upper()
        epoch_index = int(sample // expected_epoch_duration_samples)

        if epoch_index >= num_epochs_total:
            continue

        parsed_stage_code = None
        # Created for loop to match the note or symbol to a known sleep stage code
        for stage_text_map_key, code in stage_symbol_to_code_map.items():
            if f"SLEEP-STAGE-{stage_text_map_key}" == aux_note or stage_text_map_key == aux_note :
                parsed_stage_code = code
                break
        if parsed_stage_code is None and symbol in stage_symbol_to_code_map: # Fallback to symbol
            parsed_stage_code = stage_symbol_to_code_map[symbol]
        # Created if condition for a new stage is found then fill in any gaps between the last stage and now with the previous stage.
        if parsed_stage_code is not None:
            if last_updated_epoch_idx != -1 and epoch_index > last_updated_epoch_idx : 
                fill_start = last_updated_epoch_idx + 1 
                fill_end = epoch_index 
                if fill_start < num_epochs_total:
                    sleep_stages_array[fill_start:min(fill_end, num_epochs_total)] = current_stage_code
            
            # Created if condition to save the current stage in the correct spot and remember it for filling future gaps
            if epoch_index < num_epochs_total:
                sleep_stages_array[epoch_index] = parsed_stage_code
                current_stage_code = parsed_stage_code # This becomes the stage to forward-fill next
                last_updated_epoch_idx = epoch_index

    # Created if condition to fill the rest of the sleep stages using last known stage.
    if last_updated_epoch_idx != -1 and last_updated_epoch_idx < num_epochs_total - 1:
        fill_start = last_updated_epoch_idx + 1
        sleep_stages_array[fill_start:] = current_stage_code 
        
    defined_stages_count = np.sum(sleep_stages_array != sleep_stages_dict['undefined'])
    # if defined_stages_count == 0 : # Optional warning
        # print(f"  Warning (get_sleep_stages - {record_name}): No AASM sleep stages could be parsed/filled. TST might be 0.")
    # else:
        # print(f"    {record_name}: After forward-filling, {defined_stages_count} epochs have defined stages out of {num_epochs_total} total epochs.")
    
    return sleep_stages_array

# Craeted function to set up the info needed to find the file, count events, and define sleep stage codes
def calculate_ahi(record_path):
    """Calculates AHI for a single record."""
    record_name = os.path.basename(record_path)
    annotation_file_base = os.path.join(record_path, record_name) 

    unique_event_descriptions_counted = set() 

    sleep_stages_dict = {
        'wake': 0, 'nonrem1': 1, 'nonrem2': 2, 'nonrem3': 3, 'rem': 4, 'undefined': -1
    }

    try:
        # 1. To load record header for signal length and fs
        try:
            record_header = wfdb.rdheader(annotation_file_base)
        except Exception as e_header:
            print(f"Error reading header for {record_name}: {e_header}")
            return None, set()

        # 2. To load event annotations using wfdb
        annotation = wfdb.rdann(annotation_file_base, 'arousal')
        
        # 3. To define apnea/hypopnea patterns
        apnea_hypopnea_patterns = [
            r"resp_centralapnea", 
            r"resp_hypopnea",
            r"resp_obstructiveapnea",
            r"resp_mixedapnea",
            r"mixed apnea" 
        ]
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in apnea_hypopnea_patterns]

        # 4. To get sleep stages from .arousal file using header info
        sleep_stages_array = get_sleep_stages_from_annotations(annotation, record_header, record_name, sleep_stages_dict)
        
        # 5. To calculate total sleep time
        epoch_duration_seconds = 30.0
        sleep_epochs_count = np.sum((sleep_stages_array != sleep_stages_dict['wake']) & 
                                    (sleep_stages_array != sleep_stages_dict['undefined']))
        
        total_sleep_time_seconds = sleep_epochs_count * epoch_duration_seconds
        total_sleep_time_hours = total_sleep_time_seconds / 3600.0

        # Created condition if the person didn’t sleep then return AHI as 0 since there is no sleep time to measure.
        if total_sleep_time_hours <= 0:
            # print(f"  Warning: Total sleep time is zero or negative for {record_name} ({total_sleep_time_hours:.2f} hrs from {sleep_epochs_count} sleep epochs). AHI will be 0 or undefined.")
            return 0.0, unique_event_descriptions_counted

        # 6. To count apnea/hypopnea events (Span-based logic)
        apnea_hypopnea_count = 0
        i = 0
        n_ann = len(annotation.sample)
        # Created while condition to go through each event and check if it matche an apnea or hypopnea type
        while i < n_ann:
            current_description = str(annotation.aux_note[i]).strip()
            
            if current_description.startswith('('):
                tag_content = current_description[1:] 
                is_target_event_start = False
                for pattern in compiled_patterns:
                    if pattern.fullmatch(tag_content): 
                        is_target_event_start = True
                        break
               # Created if condition to count each  apnea event and then calculate AHI based on how many happened during total sleep time
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

        # 7. To calculate AHI
        ahi = apnea_hypopnea_count / total_sleep_time_hours if total_sleep_time_hours > 0 else 0.0
        return ahi, unique_event_descriptions_counted
   # To show error if something goes wrong 
    except FileNotFoundError as fnf_e:
        print(f"Error: File not found during processing of {record_name}. Details: {fnf_e}")
        return None, set()
    except Exception as e:
        print(f"General error processing {record_path} ({record_name}): {e}")
        import traceback
        traceback.print_exc() 
        return None, set()
# Created function to go through all patient folders, calculate AHI for each, and keep track of OSA results and event types
def process_all_records(base_dir, save_csv=False):
    """Processes all records in the given base directory."""
    all_results_summary = {
        "No OSA": 0, "Mild OSA": 0, "Moderate OSA": 0, "Severe OSA": 0
    }
    all_ahi_details = [] 
    overall_unique_matched_event_descriptions = set()

    record_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Created for loop for each patient folder, calculate their AHI and get the types of apnea events found
    for record_dir_name in tqdm(record_folders, desc="Processing Records for AHI"):
        record_full_path = os.path.join(base_dir, record_dir_name)
        
        ahi, unique_descriptions_for_record = calculate_ahi(record_full_path)

        # Created if condtion like if AHI was calculated then save the result with its severity and event types; otherwise, show a message
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
    # Created foor loop to show how many records had each OSA severity and list all the unique apnea event types that were found
    for severity, count in all_results_summary.items():
        print(f"{severity}: {count} records")

    print("\n--- Overall Unique Apnea/Hypopnea START Event Descriptions Counted: ---")
    if overall_unique_matched_event_descriptions:
        for idx, desc in enumerate(sorted(list(overall_unique_matched_event_descriptions))):
            print(f"  [{idx}] - '{desc}'")
    else:
        print("  No apnea/hypopnea event start descriptions were counted.")

   # Created if condition, If saving is turned on and results exist then save them to a CSV file; otherwise, show a message
    if save_csv and all_ahi_details:
        output_df = pd.DataFrame(all_ahi_details)
        csv_path_in_base = os.path.join(base_dir, "calculated_ahi_results_with_severity.csv")

        output_df.to_csv(csv_path_in_base, index=False)
        print(f"\nResults saved to '{csv_path_in_base}'")
    elif save_csv:
        print("\nNo AHI details to save to CSV.")

# Provided folder path where all the sleep record data is stored for processing.
training_dir = r"C:\Users\anton\OneDrive\Documents\project data\selected_records" 

# Created if condtion like if the data folder exists then start processing all records and save results; otherwise, show an error message
if __name__ == "__main__": 
    
    if os.path.exists(training_dir) and os.path.isdir(training_dir):
        process_all_records(training_dir, save_csv=True)
    else:
        print(f"Error: The specified directory does not exist or is not a directory: {training_dir}")
        print("Please update the 'training_dir' variable with the correct path to your data.")

