import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, WeightedRandomSampler


# ============================================================
# CONFIGURATION
# ============================================================

N_FREQ = 1601
N_ANTENNAS = 21
EXPECTED_SIZE = N_FREQ * N_ANTENNAS
EPS = 1e-8

BATCH_SIZE = 8
EPOCHS_MIXED = 80

LR_MIXED = 5e-4

MIX_REAL_RATIO = 0.70        # 70% real data, 30% simulated data
DISTANCE_LOSS_WEIGHT = 0.2   # how much the distance-aware loss matters

DATASET_SIM_PATH = "dataset_sim"
DATASET_REAL_PATH = "dataset_real"

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ============================================================
# LABEL FUNCTIONS
# ============================================================

def get_label_from_folder(folder_name):
    name = folder_name.lower()

    if name == "empty" or "empty" in name:
        return 0

    if name.startswith("pos_"):
        coords = name.split("_", 1)[1]
    else:
        coords = name

    coords = coords.replace(",", ".")
    parts = coords.split(".")

    if len(parts) < 2:
        raise ValueError(f"Could not read position from folder: {folder_name}")

    row = int(parts[0])
    col = int(parts[1])

    if not (1 <= row <= 4 and 1 <= col <= 4):
        raise ValueError(f"Invalid grid position: {folder_name}")

    return (row - 1) * 4 + col


def label_to_row_col(label):
    label = label - 1
    row = label // 4
    col = label % 4
    return row, col


# ============================================================
# LOAD DATA FUNCTIONS
# ============================================================

def load_csv_last_column(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    if data.ndim == 1:
        values = data
    else:
        values = data[:, -1]

    return values.astype(np.float32)


def load_sim_sample(mag_path, phase_path):
    mag = load_csv_last_column(mag_path)
    phase = load_csv_last_column(phase_path)

    if mag.size != phase.size:
        raise ValueError(
            f"Mag and phase have different sizes:\n"
            f"{mag_path}: {mag.size}\n"
            f"{phase_path}: {phase.size}"
        )

    if mag.size != EXPECTED_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_SIZE} values, got {mag.size}\n"
            f"File: {mag_path}"
        )

    complex_vector = mag * np.exp(1j * phase)
    data = complex_vector.reshape(N_FREQ, N_ANTENNAS)

    return data


def load_real_sample(path):
    mat = loadmat(path)

    if "S21" not in mat:
        raise KeyError(f"S21 not found in file: {path}")

    data = mat["S21"]

    if data.shape == (N_ANTENNAS, N_FREQ):
        data = data.T

    if data.shape != (N_FREQ, N_ANTENNAS):
        raise ValueError(
            f"Expected shape {(N_FREQ, N_ANTENNAS)}, got {data.shape}\n"
            f"File: {path}"
        )

    return data


# ============================================================
# DATASET
# ============================================================

class VNADataset(Dataset):
    def __init__(self, root_dir, mode):
        self.samples = []
        self.mode = mode
        self.normalizer = None

        if mode not in ["sim", "real"]:
            raise ValueError("mode must be 'sim' or 'real'")

        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            label = get_label_from_folder(folder)

            if mode == "sim":
                for f in sorted(os.listdir(folder_path)):
                    if f.endswith("_mag.csv"):
                        mag_path = os.path.join(folder_path, f)
                        phase_path = mag_path.replace("_mag.csv", "_phase.csv")

                        if not os.path.exists(phase_path):
                            raise FileNotFoundError(f"Missing phase file for {mag_path}")

                        self.samples.append((mag_path, phase_path, label))

            elif mode == "real":
                for f in sorted(os.listdir(folder_path)):
                    if f.endswith(".mat"):
                        mat_path = os.path.join(folder_path, f)
                        self.samples.append((mat_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {root_dir}")

    def set_normalizer(self, mean, std):
        self.normalizer = (mean, std)

    def __len__(self):
        return len(self.samples)

    def load_raw(self, idx):
        if self.mode == "sim":
            mag_path, phase_path, label = self.samples[idx]
            data = load_sim_sample(mag_path, phase_path)
        else:
            path, label = self.samples[idx]
            data = load_real_sample(path)

        real = np.real(data).astype(np.float32)
        imag = np.imag(data).astype(np.float32)

        x = np.stack([real, imag], axis=0)

        return x, label

    def __getitem__(self, idx):
        x, label = self.load_raw(idx)

        if self.normalizer is not None:
            mean, std = self.normalizer
            x = (x - mean) / (std + EPS)

        human_target = 1.0 if label > 0 else 0.0
        pos_target = label - 1 if label > 0 else -1

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(human_target, dtype=torch.float32),
            torch.tensor(pos_target, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


# ============================================================
# SPLIT AND NORMALIZATION
# ============================================================

def stratified_split(dataset, train_ratio=0.70, val_ratio=0.15, seed=42): 
    #make sure that the training, test and validate sets have a equal distrubution of samples corresponding to each label
    rng = np.random.default_rng(seed)
    label_to_indices = defaultdict(list)

    for i, sample in enumerate(dataset.samples):
        label = sample[-1]
        label_to_indices[label].append(i)

    train_idx = []
    val_idx = []
    test_idx = []

    for label, indices in label_to_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)

        n = len(indices)

        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


def compute_normalizer(dataset, indices):
    total_sum = np.zeros((2, 1, 1), dtype=np.float64)
    total_sq_sum = np.zeros((2, 1, 1), dtype=np.float64)
    total_count = 0

    for idx in indices:
        x, _ = dataset.load_raw(idx)

        total_sum += x.sum(axis=(1, 2), keepdims=True)
        total_sq_sum += (x ** 2).sum(axis=(1, 2), keepdims=True)

        total_count += x.shape[1] * x.shape[2]

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean ** 2
    std = np.sqrt(np.maximum(var, EPS))

    return mean.astype(np.float32), std.astype(np.float32)


def prepare_dataset(path, mode):
    dataset = VNADataset(path, mode)
    train_idx, val_idx, test_idx = stratified_split(dataset, seed=SEED)

    return dataset, train_idx, val_idx, test_idx


def make_mixed_train_loader(real_dataset, real_train_idx, sim_dataset, sim_train_idx):
    real_subset = Subset(real_dataset, real_train_idx)
    sim_subset = Subset(sim_dataset, sim_train_idx)

    mixed_dataset = ConcatDataset([real_subset, sim_subset])

    n_real = len(real_subset)
    n_sim = len(sim_subset)

    real_weight = MIX_REAL_RATIO / n_real
    sim_weight = (1.0 - MIX_REAL_RATIO) / n_sim

    weights = [real_weight] * n_real + [sim_weight] * n_sim

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=2 * n_real,
        replacement=True
    )

    loader = DataLoader(
        mixed_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler
    )

    return loader


# ============================================================
# MODEL
# ============================================================

class VNAHybridCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((4, 1)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4, 1)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((8, 4))
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.30)
        )

        self.human_head = nn.Linear(128, 1)
        self.position_head = nn.Linear(128, 16)

    def forward(self, x):
        x = self.features(x)
        x = self.shared(x)

        human_logit = self.human_head(x).squeeze(1)
        position_logits = self.position_head(x)

        return human_logit, position_logits


# ============================================================
# LOSS FUNCTION
# ============================================================

def position_coordinates_tensor(device):
    coords = []

    for label in range(1, 17):
        row, col = label_to_row_col(label)
        coords.append([row, col])

    return torch.tensor(coords, dtype=torch.float32, device=device)


def hybrid_loss(human_logit, position_logits, human_target, pos_target):
    loss_human_fn = nn.BCEWithLogitsLoss()
    loss_position_fn = nn.CrossEntropyLoss()
    loss_distance_fn = nn.MSELoss()

    loss_human = loss_human_fn(human_logit, human_target)

    human_mask = human_target == 1

    if human_mask.sum() > 0:
        logits_human = position_logits[human_mask]
        target_human = pos_target[human_mask]

        loss_position = loss_position_fn(logits_human, target_human)

        probs = torch.softmax(logits_human, dim=1)
        coords = position_coordinates_tensor(position_logits.device)

        pred_xy = probs @ coords
        true_xy = coords[target_human]

        loss_distance = loss_distance_fn(pred_xy, true_xy)

    else:
        loss_position = torch.tensor(0.0, device=human_logit.device)
        loss_distance = torch.tensor(0.0, device=human_logit.device)

    total_loss = loss_human + loss_position + DISTANCE_LOSS_WEIGHT * loss_distance

    return total_loss, loss_human.item(), loss_position.item(), loss_distance.item()


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def train_model(model, loader, optimizer):
    model.train()

    total_loss = 0.0
    total_human_loss = 0.0
    total_position_loss = 0.0
    total_distance_loss = 0.0

    for x, human_target, pos_target, _ in loader:
        x = x.to(DEVICE)
        human_target = human_target.to(DEVICE)
        pos_target = pos_target.to(DEVICE)

        human_logit, position_logits = model(x)

        loss, human_loss, position_loss, distance_loss = hybrid_loss(
            human_logit,
            position_logits,
            human_target,
            pos_target
        )

        optimizer.zero_grad() #gradient descent
        loss.backward() #backward propagation
        optimizer.step() #go forward one step

        total_loss += loss.item()
        total_human_loss += human_loss
        total_position_loss += position_loss
        total_distance_loss += distance_loss

    n_batches = max(len(loader), 1)

    return (
        total_loss / n_batches,
        total_human_loss / n_batches,
        total_position_loss / n_batches,
        total_distance_loss / n_batches
    )


def predict_labels(human_logit, position_logits, threshold=0.5):#make a guess 
    human_prob = torch.sigmoid(human_logit)
    is_human = human_prob >= threshold

    position_pred = torch.argmax(position_logits, dim=1) + 1

    final_pred = torch.where(
        is_human,
        position_pred,
        torch.zeros_like(position_pred)
    )

    return final_pred


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    correct_human_empty = 0
    total_human_empty = 0

    correct_position = 0
    total_human_samples = 0

    total_distance = 0
    distance_count = 0

    with torch.no_grad():
        for x, _, _, label in loader:
            x = x.to(DEVICE)
            label = label.to(DEVICE)

            human_logit, position_logits = model(x)
            pred_label = predict_labels(human_logit, position_logits)

            correct += (pred_label == label).sum().item()
            total += label.size(0)

            pred_human = pred_label > 0
            true_human = label > 0

            correct_human_empty += (pred_human == true_human).sum().item()
            total_human_empty += label.size(0)

            human_mask = label > 0

            if human_mask.sum() > 0:
                pred_pos = pred_label[human_mask]
                true_pos = label[human_mask]

                correct_position += (pred_pos == true_pos).sum().item()
                total_human_samples += human_mask.sum().item()

                for p, t in zip(pred_pos.cpu().numpy(), true_pos.cpu().numpy()):
                    if p > 0:
                        pr, pc = label_to_row_col(int(p))
                        tr, tc = label_to_row_col(int(t))

                        distance = np.sqrt((pr - tr)**2 + (pc - tc)**2)

                        total_distance += distance
                        distance_count += 1

    acc = 100 * correct / max(total, 1)
    human_empty_acc = 100 * correct_human_empty / max(total_human_empty, 1)
    position_acc = 100 * correct_position / max(total_human_samples, 1)
    avg_distance = total_distance / max(distance_count, 1)

    return acc, human_empty_acc, position_acc, avg_distance


def print_metrics(prefix, metrics):
    acc, human_empty_acc, position_acc, avg_distance = metrics

    print(
        f"{prefix} | "
        f"Acc {acc:.2f}% | "
        f"Human/Empty {human_empty_acc:.2f}% | "
        f"Position {position_acc:.2f}% | "
        f"AvgDist {avg_distance:.2f}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\n=== LOAD DATA ===")

    real_dataset, real_train_idx, real_val_idx, real_test_idx = prepare_dataset(
        DATASET_REAL_PATH,
        mode="real"
    )

    sim_dataset, sim_train_idx, sim_val_idx, sim_test_idx = prepare_dataset(
        DATASET_SIM_PATH,
        mode="sim"
    )

    # Important:
    # Use real training statistics for both real and sim.
    # This makes sim data adapt to the real-data scale instead of the opposite.
    real_mean, real_std = compute_normalizer(real_dataset, real_train_idx)

    real_dataset.set_normalizer(real_mean, real_std)
    sim_dataset.set_normalizer(real_mean, real_std)

    mixed_train_loader = make_mixed_train_loader(
        real_dataset,
        real_train_idx,
        sim_dataset,
        sim_train_idx
    )

    real_val_loader = DataLoader(
        Subset(real_dataset, real_val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    real_test_loader = DataLoader(
        Subset(real_dataset, real_test_idx),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    sim_val_loader = DataLoader(
        Subset(sim_dataset, sim_val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = VNAHybridCNN().to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR_MIXED,
        weight_decay=1e-4
    )

    best_real_val_acc = 0.0

    print("\n=== MIXED TRAINING: REAL + SIM ===")
    print(f"Mix ratio: {int(MIX_REAL_RATIO * 100)}% real / {int((1 - MIX_REAL_RATIO) * 100)}% sim")

    for epoch in range(EPOCHS_MIXED):
        loss, human_loss, position_loss, distance_loss = train_model(
            model,
            mixed_train_loader,
            optimizer
        )

        real_val_metrics = evaluate(model, real_val_loader)
        sim_val_metrics = evaluate(model, sim_val_loader)

        if real_val_metrics[0] > best_real_val_acc:
            best_real_val_acc = real_val_metrics[0]
            torch.save(model.state_dict(), "model_mixed_best_real_val.pth")

        print(
            f"[MIXED] Epoch {epoch + 1:02d} | "
            f"Loss {loss:.4f} | "
            f"HumanLoss {human_loss:.4f} | "
            f"PosLoss {position_loss:.4f} | "
            f"DistLoss {distance_loss:.4f}"
        )

        print_metrics("        Real Val", real_val_metrics)
        print_metrics("        Sim Val ", sim_val_metrics)

    print("\n=== FINAL TEST ON REAL DATA ===")

    model.load_state_dict(torch.load("model_mixed_best_real_val.pth", map_location=DEVICE))

    real_test_metrics = evaluate(model, real_test_loader)

    print_metrics("REAL Test", real_test_metrics)

    print("\nSaved best model as: model_mixed_best_real_val.pth")


if __name__ == "__main__":
    main()