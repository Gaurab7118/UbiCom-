import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# ==========================================
# 1. DIRECTORY CONFIGURATION
# ==========================================
# Pointing to your local D: drive folders
DATASET_DIR = r"D:\UbiCom Data\Data\gaurab_upload\data"
OUTPUT_DIR = r"D:\UbiCom Data\Data\gaurab_upload\Cleaned_25Hz_Dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_LABELS = [
    'original_label:LYING_DOWN',
    'original_label:SITTING',
    'original_label:STANDING_IN_PLACE',
    'original_label:STANDING_AND_MOVING',
    'original_label:WALKING',
    'original_label:RUNNING',
    'original_label:BICYCLING'
]

# ==========================================
# 2. HELPER FUNCTION: RESAMPLE TO 25 HZ
# ==========================================
def load_and_resample_sensor(filepath, sensor_prefix):
    """Reads a space-delimited .dat file and interpolates to strictly 25 Hz."""
    if not os.path.exists(filepath):
        return None

    try:
        # Optimization: use engine='c' for speed
        df = pd.read_csv(
            filepath, 
            sep=r'\s+', 
            header=None, 
            engine='c',
            names=['time_offset', f'{sensor_prefix}_x', f'{sensor_prefix}_y', f'{sensor_prefix}_z']
        )

        df = df.dropna().drop_duplicates(subset=['time_offset'])
        if len(df) < 2:
            return None

        min_time = df['time_offset'].min()
        max_time = df['time_offset'].max()
        
        # 0.04 seconds = 25 Hz frequency
        grid_25hz = np.arange(min_time, max_time, 0.04)

        resampled_data = {'time_offset': grid_25hz}
        for axis in ['x', 'y', 'z']:
            col_name = f'{sensor_prefix}_{axis}'
            f_interp = interp1d(df['time_offset'], df[col_name], kind='linear', fill_value='extrapolate')
            resampled_data[col_name] = f_interp(grid_25hz)

        return pd.DataFrame(resampled_data)
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# ==========================================
# 3. MAIN PROCESSING LOOP
# ==========================================
def process_dataset():
    
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found at {DATASET_DIR}")
        return

    user_folders = [f.path for f in os.scandir(DATASET_DIR) if f.is_dir()]
    print(f"Found {len(user_folders)} user folders.")

    for user_folder in user_folders:
        user_uuid = os.path.basename(user_folder)
        print(f"\n>>> Processing User: {user_uuid}")

        labels_file = os.path.join(user_folder, "labels.csv")
        acc_dir = os.path.join(user_folder, "acc")
        gyro_dir = os.path.join(user_folder, "gyro")

        if not os.path.exists(labels_file):
            print("  Skipping: No labels.csv found.")
            continue
        
        df_labels = pd.read_csv(labels_file)
        user_processed_data = []
        total_timestamps = len(df_labels)

        for i, (index, row) in enumerate(df_labels.iterrows()):
            # Print progress every 10 iterations without spamming the console
            if i % 10 == 0:
                print(f"  Progress: {i}/{total_timestamps} timestamps...", end='\r')

            timestamp = str(int(row['timestamp']))
            acc_path = os.path.join(acc_dir, f"{timestamp}.m_raw_acc.dat")
            gyro_path = os.path.join(gyro_dir, f"{timestamp}.m_proc_gyro.dat")

            df_acc = load_and_resample_sensor(acc_path, 'acc')
            df_gyro = load_and_resample_sensor(gyro_path, 'gyro')

            if df_acc is not None and df_gyro is not None:
                # Align both sensors on the exact same 25Hz time grid
                df_merged = pd.merge_asof(df_acc, df_gyro, on='time_offset', direction='nearest')
                df_merged.insert(0, 'timestamp', int(timestamp))

                # Map the challenge target classes
                for label in TARGET_LABELS:
                    label_val = row[label] if label in df_labels.columns else 0
                    clean_label_name = label.replace("original_label:", "label:")
                    df_merged[clean_label_name] = label_val
                
                user_processed_data.append(df_merged)

        # Save final output for the user
        if user_processed_data:
            print(f"\n  Finalizing CSV for {user_uuid}...")
            final_user_df = pd.concat(user_processed_data, ignore_index=True)
            output_file = os.path.join(OUTPUT_DIR, f"{user_uuid}_cleaned.csv")
            final_user_df.to_csv(output_file, index=False)
            print(f"[SUCCESS] Saved: {output_file} ({len(final_user_df)} rows)")
        else:
            print(f"\n[WARNING] No valid sensor data found for {user_uuid}.")

if __name__ == "__main__":
    process_dataset()