import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Clear terminal for clean output
os.system('cls' if os.name == 'nt' else 'clear')

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Point this to the FOLDER containing all your cleaned CSVs
CLEANED_DATA_DIR = r"D:\UbiCom Data\Data\gaurab_upload\Cleaned_25Hz_Dataset"

WINDOW_SIZE = 100  # 4 seconds at 25 Hz
STEP_SIZE = 50     # 2-second overlap (50% overlap)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Limit the number of users loaded at once to prevent Windows MemoryError
# Set to None to load all files, or a number (e.g., 15) if your RAM is limited
MAX_USERS_TO_LOAD = 15 

# The 6 features and 7 targets
FEATURES = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
LABELS = [
    'label:LYING_DOWN', 'label:SITTING', 'label:STANDING_IN_PLACE',
    'label:STANDING_AND_MOVING', 'label:WALKING', 'label:RUNNING', 'label:BICYCLING'
]

# ==========================================
# 2. DATASET & SLIDING WINDOW
# ==========================================
class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sliding_windows(df, window_size, step_size):
    """Slices the continuous CSV into discrete 4-second blocks."""
    X_list, y_list = [], []
    
    features_array = df[FEATURES].values
    labels_array = df[LABELS].values
    
    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        
        # X shape: (Channels, Sequence Length) -> (6, 100)
        window_x = features_array[start:end].T  
        
        # Determine the most prominent label in this 4-second window
        window_y_matrix = labels_array[start:end]
        prominent_label_idx = np.argmax(np.sum(window_y_matrix, axis=0))
        
        X_list.append(window_x)
        y_list.append(prominent_label_idx)
        
    return X_list, y_list

# ==========================================
# 3. THE 1D-MOBILENET ARCHITECTURE
# ==========================================
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Depthwise: Applies a separate filter to each input channel
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Pointwise: Combines the outputs of the depthwise step
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class MobileNet1D(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        
        # MobileNet bottleneck blocks
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv1d(32, 64, kernel_size=3, padding=1),
            DepthwiseSeparableConv1d(64, 128, kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv1d(128, 128, kernel_size=3, padding=1),
            DepthwiseSeparableConv1d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        
        # Global Average Pooling and Classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print("Finding cleaned datasets...")
    all_csv_files = glob.glob(os.path.join(CLEANED_DATA_DIR, "*_cleaned.csv"))
    
    if len(all_csv_files) == 0:
        print(f"[ERROR] No CSV files found in {CLEANED_DATA_DIR}")
        return
        
    if MAX_USERS_TO_LOAD:
        all_csv_files = all_csv_files[:MAX_USERS_TO_LOAD]
        print(f"Limiting load to {MAX_USERS_TO_LOAD} users to save RAM...")

    all_X, all_y = [], []
    
    # Process files one by one to save RAM
    for idx, file in enumerate(all_csv_files):
        print(f"[{idx+1}/{len(all_csv_files)}] Processing {os.path.basename(file)}...")
        df = pd.read_csv(file)
        
        # Fill missing values with 0 just in case
        df.fillna(0, inplace=True)
        
        X_user, y_user = create_sliding_windows(df, WINDOW_SIZE, STEP_SIZE)
        all_X.extend(X_user)
        all_y.extend(y_user)

    # Convert everything to numpy arrays at the very end
    X = np.array(all_X)
    y = np.array(all_y)
    
    print(f"\nFinal Data shape: X={X.shape}, y={y.shape}")
    
    print("Splitting into Train and Validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(SensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining Model on: {device}")
    
    model = MobileNet1D(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            
        acc = correct / len(X_train)
        
        # Validation Loop
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_outputs = model(val_X)
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct += (val_predicted == val_y).sum().item()
        
        val_acc = val_correct / len(X_val)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(train_loader):.4f} - Train Acc: {acc:.4f} - Val Acc: {val_acc:.4f}")

    # Save the model
    save_path = "edge_mobilenet_1d.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] Model saved to '{save_path}'")

if __name__ == "__main__":
    main()