# Diabetic Retinopathy Severity Grading
# Applied Deep Learning - Capstone Project 2
# Pure Linear Procedural Workflow (High Performance Hardware Optimized)

# ==============================================================================
# Step 1: Import Libraries
# ==============================================================================
import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

IMG_SIZE = 288

# Augmentations (Resizing is handled in C++ via OpenCV during loading for maximum speed)
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hardware-accelerated PyTorch Dataset (OpenCV C++ decoding + SIMD resizing)
class RetinalDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        self.df = data_frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        lbl = self.df.loc[idx, 'label']

        # Fast OpenCV C++ decode and initial resize (drastically faster than PIL on disk)
        bgr = cv2.imread(img_path)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            image = Image.fromarray(rgb)
        else:
            image = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(lbl, dtype=torch.long)


# Windows multi-worker safe execution guard
if __name__ == '__main__':
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Name    :", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ==============================================================================
    # Step 2: Load Datasets and Exploratory Data Analysis (EDA)
    # ==============================================================================
    print("\n--- Step 2: Loading Datasets ---")
    records = []

    # 1. EyePACS (Dataset_2)
    if os.path.exists("Dataset_2/trainLabels.csv"):
        df_eyepacs = pd.read_csv("Dataset_2/trainLabels.csv")
        img_map_eyepacs = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob("Dataset_2/**/*.jpeg", recursive=True)}
        for _, row in df_eyepacs.iterrows():
            img_id = str(row['image']).strip()
            if img_id in img_map_eyepacs:
                records.append({'image_path': img_map_eyepacs[img_id], 'label': int(row['level']), 'source': 'EyePACS'})

    # 2. APTOS 2019 (Dataset_3)
    if os.path.exists("Dataset_3/train.csv"):
        df_aptos = pd.read_csv("Dataset_3/train.csv")
        img_map_aptos = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob("Dataset_3/**/*.png", recursive=True)}
        for _, row in df_aptos.iterrows():
            img_id = str(row['id_code']).strip()
            if img_id in img_map_aptos:
                records.append({'image_path': img_map_aptos[img_id], 'label': int(row['diagnosis']), 'source': 'APTOS'})

    # 3. IDRiD (Dataset_4)
    idrid_train_csv = "Dataset_4/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    idrid_test_csv = "Dataset_4/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    img_map_idrid = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob("Dataset_4/**/*.jpg", recursive=True)}

    for p in [idrid_train_csv, idrid_test_csv]:
        if os.path.exists(p):
            df_idrid = pd.read_csv(p)
            df_idrid.columns = [c.strip() for c in df_idrid.columns]
            for _, row in df_idrid.iterrows():
                img_id = str(row['Image name']).strip()
                if img_id in img_map_idrid:
                    records.append({'image_path': img_map_idrid[img_id], 'label': int(row['Retinopathy grade']), 'source': 'IDRiD'})

    # Combine all datasets into a single DataFrame
    df = pd.DataFrame(records)
    print("Total Images Found :", len(df))
    print("Missing Values     :\n", df.isnull().sum())
    print("\nClass Distribution :")
    print(df['label'].value_counts().sort_index())

    # Class-Balanced Sampling (Preserve rare grades 1, 3, 4)
    TARGET_SAMPLES = 15000
    df_minority = df[df['label'].isin([1, 3, 4])]
    n_remaining = max(0, TARGET_SAMPLES - len(df_minority))
    quota_each = n_remaining // 2

    df_g0 = df[df['label'] == 0].sample(n=min(quota_each, (df['label'] == 0).sum()), random_state=42)
    df_g2 = df[df['label'] == 2].sample(n=min(quota_each, (df['label'] == 2).sum()), random_state=42)
    df_balanced = pd.concat([df_minority, df_g0, df_g2]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("\nBalanced Samples for Training:")
    for g in range(5):
        print(f"Grade {g}: {(df_balanced['label'] == g).sum():,} samples")

    # ==============================================================================
    # Step 3: Train / Validation / Test Split
    # ==============================================================================
    print("\n--- Step 3: Splitting Dataset ---")
    train_df, temp_df = train_test_split(df_balanced, test_size=0.20, random_state=42, stratify=df_balanced['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

    print("Train shape     :", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape      :", test_df.shape)

    # ==============================================================================
    # Step 4: High-Throughput PyTorch DataLoaders (Batch Size 32 + 4 Parallel Workers)
    # ==============================================================================
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        RetinalDataset(train_df, train_transforms),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        RetinalDataset(val_df, val_transforms),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        RetinalDataset(test_df, val_transforms),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin
    )
    print(f"\nDataLoader Configured: Batch Size={BATCH_SIZE}, Workers={NUM_WORKERS}, Pin Memory={use_pin}")

    # ==============================================================================
    # Step 5: Model Building
    # ==============================================================================
    print("\n--- Step 5: Building Model ---")
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

    # Replace classifier head linearly
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 5)
    )
    model = model.to(device)

    print("Backbone  : EfficientNet-B2")
    print("Classifier: Dropout(0.4) -> Linear(1408, 5)")

    # Loss Function with class weights and label smoothing
    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'].values)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Hardware Tensor Core Acceleration: Automatic Mixed Precision (AMP)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"Hardware Tensor Core Mixed Precision (AMP FP16): {'ENABLED (3x Speedup)' if use_amp else 'DISABLED'}")

    # Warmup Phase: Freeze backbone features for epochs 1-2 for rapid head convergence
    print("\n[WARMUP] Freezing backbone features for epochs 1-2 for rapid training...")
    for param in model.features.parameters():
        param.requires_grad = False

    # ==============================================================================
    # Step 6: Model Training Loop
    # ==============================================================================
    EPOCHS = 12
    MODEL_PATH = "best_retinopathy_model.pth"

    print(f"\n--- Step 6: Training Model ({EPOCHS} Epochs) ---")
    best_qwk = -1.0

    for epoch in range(EPOCHS):
        # Selective Fine-Tuning: Unfreeze high-level disease-specific blocks (Stages 6-8) at Epoch 3
        # Keeps low-level stages 0-5 frozen to prevent 4GB VRAM overflow and PCIe paging
        if epoch == 2:
            print("\n[FINE-TUNING] Unfreezing top feature blocks (Stages 6-8) for high-speed fine-tuning...")
            torch.cuda.empty_cache()
            for param in model.features[6:].parameters():
                param.requires_grad = True

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] Train", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Mixed Precision Forward Pass (Tensor Cores)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Mixed Precision Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Live batch status in console
            train_bar.set_postfix({
                'loss': f"{running_loss / total:.4f}",
                'acc': f"{(correct / total) * 100:.2f}%"
            })

        train_loss = running_loss / total
        train_acc = (correct / total) * 100

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] Val  ", leave=False)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    l = criterion(outputs, labels)

                val_loss += l.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_bar.set_postfix({'loss': f"{val_loss / len(val_targets):.4f}"})

        val_loss = val_loss / len(val_df)
        val_acc = accuracy_score(val_targets, val_preds) * 100
        val_qwk = cohen_kappa_score(val_targets, val_preds, weights='quadratic')

        scheduler.step(val_qwk)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] - Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val QWK: {val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Saved Best Model Checkpoint (QWK: {best_qwk:.4f})")

    # ==============================================================================
    # Step 7: Evaluation on Unseen Test Data
    # ==============================================================================
    print("\n--- Step 7: Final Test Evaluation ---")

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    test_preds = []
    test_targets = []

    test_bar = tqdm(test_loader, desc="Evaluating Test Set", leave=True)
    with torch.no_grad():
        for images, labels in test_bar:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)

    # Metrics
    acc = accuracy_score(test_targets, test_preds) * 100
    qwk = cohen_kappa_score(test_targets, test_preds, weights='quadratic')
    f1 = f1_score(test_targets, test_preds, average='macro')

    # Clinical Referable DR (Grade 0-1 vs Grade 2-4)
    bin_acc = accuracy_score(test_targets >= 2, test_preds >= 2) * 100

    print(f"\nFinal Test Set Results:")
    print(f"5-Class Accuracy           : {acc:.2f}%")
    print(f"Quadratic Weighted Kappa    : {qwk:.4f}")
    print(f"Macro F1-Score             : {f1:.4f}")
    print(f"Referable DR Accuracy (Bin) : {bin_acc:.2f}%")

    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_targets, test_preds))

    print("\n[COMPLETE] Student Pipeline Finished Successfully.")
