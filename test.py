# data organized as: 
# dataset_sim/
#    empty/
#    pos_1/
#    pos_2/
#    ...
#    pos_16/

#dataset_real/
#    empty/
#    pos_1/
#    pos_2/
#    ...
#    pos_16/

# sim folder contains: 
# xxx_mag.csv
# xxx_phase.csv

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from scipy.io import loadmat

# =========================
# CONFIG
# =========================
NUM_CLASSES = 17  # 16 positions + empty
BATCH_SIZE = 16
EPOCHS_SIM = 15
EPOCHS_REAL = 10
LR_SIM = 0.001
LR_REAL = 0.0003

# =========================
# LABEL MAPPING
# =========================
def get_label_from_folder(folder_name):
    if folder_name == "empty":
        return 0

    elif folder_name.startswith("pos_"):
        # Extract row and column
        coords = folder_name.split("_")[1]  # "1.1"
        row, col = coords.split(".")

        row = int(row)
        col = int(col)

        # Map (row, col) → 1–16
        label = (row - 1) * 4 + col

        return label

    else:
        raise ValueError(f"Unknown folder: {folder_name}")
# =========================
# DATASET
# =========================
class VNADataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            label = get_label_from_folder(folder)

            files = os.listdir(folder_path)

            for file in files:
                file_path = os.path.join(folder_path, file)

                # CASE 1: magnitude file (simulation)
                if file.endswith("_mag.csv"):
                    phase_path = file_path.replace("_mag.csv", "_phase.csv")

                    if os.path.exists(phase_path):
                        self.samples.append(("mag_phase", file_path, phase_path, label))

                # CASE 2: complex file (real data)
                elif file.endswith(".csv") and "_mag" not in file and "_phase" not in file:
                    self.samples.append(("complex", file_path, None, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_type, path1, path2, label = self.samples[idx]
        print(self.samples[idx])
        # =========================
        # SIMULATED (MAG + PHASE)
        # =========================
        if sample_type == "mag_phase":
            mag = np.loadtxt(path1, skiprows=1, usecols=4)
            phase = np.loadtxt(path2, skiprows=1, usecols=4)

            Re = mag * np.cos(phase)
            Im = mag * np.sin(phase)

        # =========================
        # REAL (COMPLEX)
        elif sample_type == "complex":
            
            mat = loadmat(path1)

            # CHANGE 'S21' to your actual variable name
            signal = mat['S21']

            signal = signal.squeeze()

            Re = np.real(signal)
            Im = np.imag(signal)
        # =========================
        # NORMALIZATION
        # =========================
        Re = (Re - np.mean(Re)) / (np.std(Re) + 1e-8)
        Im = (Im - np.mean(Im)) / (np.std(Im) + 1e-8)

        data = np.stack([Re, Im], axis=0)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label)

# =========================
# CNN MODEL
# =========================
class CNN1D(nn.Module):
    def __init__(self, input_length):
        super().__init__()

        self.conv = nn.Sequential( #This is the model, important to know how this works and argue for in the report. Good to experiment with for different results
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self._to_linear = None
        self._get_conv_output(input_length)

        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )

    def _get_conv_output(self, input_length):
        x = torch.randn(1, 2, input_length)
        x = self.conv(x)
        self._to_linear = x.numel()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in loader: #x is data (mag and phase) y is label (position or empty)
        outputs = model(x) #Process training data (x) in model, get predictions as return value
        loss = criterion(outputs, y) #Calculate loss value by assessing predictions (outputs) againts true label (y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

# =========================
# EVALUATION
# =========================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(): # no_grad means no gradient descent, i.e. no change made to the model weights
        for x, y in loader: #x is data (mag and phase) y is label (position or empty)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1) #Predicted is the models label guess for each sample

            total += y.size(0) #Total is total samples regardless of correct or incorrect prediction
            correct += (predicted == y).sum().item() #amount of correct prediction

    return 100 * correct / total  #Accuracy. Percentage of correct predictions

# =========================
# LOAD DATA FUNCTION
# =========================
def prepare_data(dataset_path):
    dataset = VNADataset(dataset_path)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    return dataset, train_loader, val_loader, test_loader

# =========================
# MAIN PIPELINE
# =========================

# ----------- SIMULATION TRAINING -----------
print("=== TRAINING ON SIMULATED DATA ===")

dataset_sim, train_loader, val_loader, test_loader = prepare_data("dataset_sim")

sample_input, _ = dataset_sim[0]
input_length = sample_input.shape[1]

model = CNN1D(input_length)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR_SIM)

for epoch in range(EPOCHS_SIM):
    loss = train_model(model, train_loader, optimizer, criterion)
    val_acc = evaluate(model, val_loader)

    print(f"[SIM] Epoch {epoch+1}, Loss: {loss:.4f}, Val Acc: {val_acc:.2f}%")

torch.save(model.state_dict(), "model_sim.pth")


# ----------- REAL FINE-TUNING -----------
print("\n=== FINE-TUNING ON REAL DATA ===")

dataset_real, train_loader, val_loader, test_loader = prepare_data("dataset_real")

model.load_state_dict(torch.load("model_sim.pth"))

optimizer = optim.Adam(model.parameters(), lr=LR_REAL)

for epoch in range(EPOCHS_REAL):
    loss = train_model(model, train_loader, optimizer, criterion)
    val_acc = evaluate(model, val_loader)

    print(f"[REAL] Epoch {epoch+1}, Loss: {loss:.4f}, Val Acc: {val_acc:.2f}%")


# ----------- FINAL TEST -----------
test_acc = evaluate(model, test_loader)
print(f"\nFINAL TEST ACCURACY: {test_acc:.2f}%")