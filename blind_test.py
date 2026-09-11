import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Point this to your blind test CSV (the one with just 8 columns)
TEST_FILE = r"D:\UbiCom Data\Data\gaurab_upload\blind_test_data5.csv" 
MODEL_PATH = r"D:\UbiCom Data\Data\gaurab_upload\model\edge_mobilenet_1d.pth"
OUTPUT_FILE = "predicted_full_format.csv"

WINDOW_SIZE = 100  # 4 seconds at 25 Hz
STEP_SIZE = 100    # No overlap (1 window = 1 prediction)
FEATURES = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
LABELS = [
    'label:LYING_DOWN', 'label:SITTING', 'label:STANDING_IN_PLACE',
    'label:STANDING_AND_MOVING', 'label:WALKING', 'label:RUNNING', 'label:BICYCLING'
]

# ==========================================
# 2. REBUILD ARCHITECTURE (Required to load weights)
# ==========================================
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu2(self.bn2(self.pointwise(self.relu1(self.bn1(self.depthwise(x))))))

class MobileNet1D(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv1d(32, 64, kernel_size=3, padding=1),
            DepthwiseSeparableConv1d(64, 128, kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv1d(128, 128, kernel_size=3, padding=1),
            DepthwiseSeparableConv1d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ==========================================
# 3. GENERATE ROW-BY-ROW PREDICTIONS
# ==========================================
def run_full_format_inference():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Cannot find {MODEL_PATH}.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNet1D(num_classes=len(LABELS)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"Loading raw sensor data from: {os.path.basename(TEST_FILE)}")
    df = pd.read_csv(TEST_FILE).fillna(0)
    
    # Initialize all 7 label columns with 0s
    for label in LABELS:
        df[label] = 0
        
    features_array = df[FEATURES].values

    print("Analyzing sensor windows and applying 0s and 1s...")
    with torch.no_grad():
        # Iterate through the data in 100-row chunks
        for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            
            # Extract the 4 seconds of sensor data
            window_x = features_array[start:end].T
            tensor_x = torch.tensor(np.array([window_x]), dtype=torch.float32).to(device)
            
            # The model predicts the activity for this block
            output = model(tensor_x)
            predicted_idx = torch.argmax(output, 1).item()
            predicted_label_name = LABELS[predicted_idx]
            
            # Set the winning label column to 1 for all 100 rows in this window
            df.loc[start:end-1, predicted_label_name] = 1

    # Save the fully formatted dataframe
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n[SUCCESS] Fully formatted data saved to '{OUTPUT_FILE}'")
    # Print a small sample of the 1s and 0s to verify
    print(df[['timestamp', 'acc_x'] + LABELS].head(5))

if __name__ == "__main__":
    run_full_format_inference()