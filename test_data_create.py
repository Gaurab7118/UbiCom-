import pandas as pd
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Point this to a user's cleaned CSV that you DID NOT load during training
# (e.g., pick a file near the bottom of your Cleaned_25Hz_Dataset folder)
INPUT_FILE = r"D:\UbiCom Data\Data\gaurab_upload\Cleaned_25Hz_Dataset\0E6184E1-90C0-48EE-B25A-F1ECB7B9714E_cleaned.csv"
OUTPUT_FILE = "blind_test_data5.csv"

def create_blind_test_file():
    print(f"Loading {os.path.basename(INPUT_FILE)}...")
    df = pd.read_csv(INPUT_FILE)
    
    # We only keep the time and sensor columns, leaving all labels behind
    columns_to_keep = [
        'timestamp', 'time_offset', 
        'acc_x', 'acc_y', 'acc_z', 
        'gyro_x', 'gyro_y', 'gyro_z'
    ]
    
    # Create the new dataframe with just the 8 required columns
    blind_df = df[columns_to_keep]
    
    # Save it to your workspace
    blind_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n[SUCCESS] Created {OUTPUT_FILE} with {len(blind_df)} rows.")
    print("This file contains ZERO labels. It is ready for the inference script.")
    print(blind_df.head())

if __name__ == "__main__":
    create_blind_test_file()