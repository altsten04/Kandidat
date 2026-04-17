import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from scipy.io import loadmat

# =========================
# CONFIGURATION
# =========================
NUM_CLASSES = 17   # 16 grid positions + 1 "empty"
BATCH_SIZE = 8
EPOCHS_SIM = 10
EPOCHS_REAL = 5
LR_SIM = 0.001
LR_REAL = 0.0003

# =========================
# LABEL MAPPING
# =========================
def get_label_from_folder(folder_name):
    """
    Converts folder name into a class label.
    Example: pos_2.3 → label 7
    """
    if folder_name == "empty":
        return 0

    elif folder_name.startswith("pos_"):
        coords = folder_name.split("_")[1]  # "2.3"
        row, col = coords.split(".")

        # Map grid position → class index (1–16)
        return (int(row) - 1) * 4 + int(col)

    else:
        raise ValueError(folder_name)

# =========================
# LOAD SIGNAL FUNCTION
# =========================
def load_signal(path):
    """
    Loads CSV data.
    Handles both:
    - single-column files
    - multi-column files (takes last column)
    """
    data = np.loadtxt(path, skiprows=1)

    if data.ndim == 1:
        return data
    else:
        return data[:, -1]  # take last column (actual measurement)

# =========================
# DATASET CLASS
# =========================
class VNADataset(Dataset):
    def __init__(self, root_dir, mode="sim"):
        """
        mode = "sim" → simulated data (mag + phase files)
        mode = "real" → real data (.mat files)
        """
        self.samples = []
        self.mode = mode

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            label = get_label_from_folder(folder)

            # =========================
            # SIMULATED DATA
            # =========================
            if mode == "sim":
                for f in os.listdir(folder_path):
                    if f.endswith("_mag.csv"):
                        mag_path = os.path.join(folder_path, f)
                        phase_path = mag_path.replace("_mag.csv", "_phase.csv")

                        if os.path.exists(phase_path):
                            # One mag/phase pair = one sample
                            self.samples.append((mag_path, phase_path, label))

            # =========================
            # REAL DATA (.mat)
            # =========================
            elif mode == "real":
                for f in os.listdir(folder_path):
                    if f.endswith(".mat"):
                        self.samples.append((os.path.join(folder_path, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # =========================
        # SIMULATED DATA PROCESSING
        # =========================
        if self.mode == "sim":
            path_mag, path_phase, label = self.samples[idx]

            # Load both files as full tables
            mag_data = np.loadtxt(path_mag, delimiter=",", skiprows=1)
            phase_data = np.loadtxt(path_phase, delimiter=",", skiprows=1)

            # Take the last column = actual measurement values
            mag = mag_data[:, -1]
            phase = phase_data[:, -1]

            # Handle mismatch between empty and human files
            if mag.size == 1601 and phase.size == 1601 * 21:
                mag = np.tile(mag, 21)

            if phase.size == 1601 and mag.size == 1601 * 21:
                phase = np.tile(phase, 21)

            # Safety check
            if mag.size != phase.size:
                raise ValueError(
                    f"Magnitude and phase sizes do not match:\n"
                    f"{path_mag}\n{path_phase}\n"
                    f"mag={mag.size}, phase={phase.size}"
                )

            # Convert to complex
            complex_vector = mag * np.exp(1j * phase)

            # Ensure correct size
            if complex_vector.size != 1601 * 21:
                raise ValueError(
                    f"Expected 33621 values, got {complex_vector.size}"
                )

            # Reshape to (freq, antennas)
            data = complex_vector.reshape(1601, 21)
        # =========================
        # REAL DATA PROCESSING
        # =========================
        else:
            path, label = self.samples[idx]

            mat = loadmat(path)

            # Extract S21 (complex matrix)
            data = mat["S21"]   # shape: (1601, 21)

        # =========================
        # SPLIT INTO REAL + IMAG CHANNELS
        # =========================
        Re = np.real(data)
        Im = np.imag(data)

        # Normalize (important for ML stability)
        Re = (Re - np.mean(Re)) / (np.std(Re) + 1e-8)
        Im = (Im - np.mean(Im)) / (np.std(Im) + 1e-8)

        # Stack → shape becomes (2, freq, antennas)
        data = np.stack([Re, Im], axis=0)

        # Convert to PyTorch tensors
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label)

# =========================
# CNN MODEL (2D)
# =========================
class CNN2D(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution layers extract spatial patterns
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Fully connected layers classify position
        self.fc = nn.Sequential(
            nn.Linear(32 * 400 * 5, 64),  # depends on input size after pooling
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv(x)  # extract features
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)  # output class scores

# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in loader:
        outputs = model(x)  # model prediction

        # Compare prediction with true label
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()  # compute gradients
        optimizer.step()  # update model weights

        total_loss += loss.item()

    return total_loss

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # no learning here
        for x, y in loader:
            outputs = model(x)

            # Predicted = model's guessed class for each sample
            _, predicted = torch.max(outputs, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    # Accuracy = percentage of correct predictions
    return 100 * correct / total

# =========================
# DATA PREPARATION
# =========================
def prepare_data(path, mode):
    dataset = VNADataset(path, mode)

    # Split into train / validation / test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    return (
        dataset,
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_set, batch_size=BATCH_SIZE),
        DataLoader(test_set, batch_size=BATCH_SIZE),
    )

# =========================
# MAIN PIPELINE
# =========================

print("=== TRAIN SIM ===")

dataset_sim, train_loader, val_loader, test_loader = prepare_data("dataset_sim", "sim")

model = CNN2D()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR_SIM)

# Train on simulated data first
for epoch in range(EPOCHS_SIM):
    loss = train_model(model, train_loader, optimizer, criterion)
    acc = evaluate(model, val_loader)

    print(f"[SIM] Epoch {epoch+1}: Loss {loss:.3f}, Acc {acc:.2f}%")

# Save model
torch.save(model.state_dict(), "model_sim.pth")

print("\n=== FINETUNE REAL ===")

dataset_real, train_loader, val_loader, test_loader = prepare_data("dataset_real", "real")

# Load pretrained model
model.load_state_dict(torch.load("model_sim.pth"))

optimizer = optim.Adam(model.parameters(), lr=LR_REAL)

# Fine-tune on real data
for epoch in range(EPOCHS_REAL):
    loss = train_model(model, train_loader, optimizer, criterion)
    acc = evaluate(model, val_loader)

    print(f"[REAL] Epoch {epoch+1}: Loss {loss:.3f}, Acc {acc:.2f}%")

# Final test
print("\n=== TEST ===")
print("Accuracy:", evaluate(model, test_loader))