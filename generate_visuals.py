"""
================================================================================
  DIABETIC RETINOPATHY: VISUAL EVALUATION & CLINICAL METRIC SUITE
  Generates High-Resolution Plots from Trained Checkpoint:
  1. Multi-Class ROC Curves with AUC Scores (per grade + macro average)
  2. Normalized Confusion Matrix Heatmap (demonstrating off-by-one errors)
  3. Per-Grade Precision, Recall, and F1-Score Bar Chart
  4. Clinical Referable DR (Binary Screening) Confusion Matrix & Metrics
================================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)

import cv2
import glob
from PIL import Image
from torchvision import models

GRADE_NAMES = [
    "Grade 0 (No DR)",
    "Grade 1 (Mild)",
    "Grade 2 (Moderate)",
    "Grade 3 (Severe)",
    "Grade 4 (Proliferative)"
]

GRADE_SHORT = ["G0", "G1", "G2", "G3", "G4"]


class RetinalDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, transform=None):
        self.df = data_frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        lbl = self.df.loc[idx, 'label']
        bgr = cv2.imread(img_path)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (288, 288), interpolation=cv2.INTER_AREA)
            image = Image.fromarray(rgb)
        else:
            image = Image.open(img_path).convert('RGB').resize((288, 288))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(lbl, dtype=torch.long)


def build_model(arch="efficientnet_b2", num_classes=5):
    model = models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def prepare_test_dataset(samples=15000, img_size=288):
    """Reconstructs the exact identical stratified test holdout set."""
    records = []
    if os.path.exists("Dataset_2/trainLabels.csv"):
        df_eyepacs = pd.read_csv("Dataset_2/trainLabels.csv")
        img_map_eyepacs = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob("Dataset_2/**/*.jpeg", recursive=True)}
        for _, row in df_eyepacs.iterrows():
            img_id = str(row['image']).strip()
            if img_id in img_map_eyepacs:
                records.append({'image_path': img_map_eyepacs[img_id], 'label': int(row['level'])})

    if os.path.exists("Dataset_3/train.csv"):
        df_aptos = pd.read_csv("Dataset_3/train.csv")
        img_map_aptos = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob("Dataset_3/**/*.png", recursive=True)}
        for _, row in df_aptos.iterrows():
            img_id = str(row['id_code']).strip()
            if img_id in img_map_aptos:
                records.append({'image_path': img_map_aptos[img_id], 'label': int(row['diagnosis'])})

    df = pd.DataFrame(records)
    if samples > 0 and samples < len(df):
        minority_grades = [1, 3, 4]
        df_minority = df[df['label'].isin(minority_grades)]
        n_minority = len(df_minority)
        n_remaining = max(0, samples - n_minority)
        quota = n_remaining // 2

        df_grade0 = df[df['label'] == 0].sample(n=min(quota, (df['label'] == 0).sum()), random_state=42)
        df_grade2 = df[df['label'] == 2].sample(n=min(quota, (df['label'] == 2).sum()), random_state=42)
        df = pd.concat([df_minority, df_grade0, df_grade2]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    class_counts = df['label'].value_counts()
    stratify_arg = df['label'] if (class_counts.min() >= 2 and len(class_counts) > 1) else None
    _, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=stratify_arg)

    temp_counts = temp_df['label'].value_counts()
    stratify_temp = temp_df['label'] if (temp_counts.min() >= 2 and len(temp_counts) > 1) else None
    _, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=stratify_temp)

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_loader = DataLoader(
        RetinalDataset(test_df, test_transforms),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return test_loader, len(test_df)


def run_evaluation(model_path="best_retinopathy_model.pth", arch="efficientnet_b2", img_size=288, samples=15000, output_dir="evaluation_plots"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model checkpoint not found at: {model_path}")
        return

    # Load Model
    print(f"[INFO] Loading {arch} weights from: {model_path}")
    model = build_model(arch=arch, num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare Dataset
    print("[INFO] Loading unseen Test holdout set...")
    test_loader, n_test = prepare_test_dataset(samples=samples, img_size=img_size)
    print(f"[INFO] Evaluating {n_test:,} Test images with Test-Time Augmentation (TTA)...")

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for step, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)

            out_orig = F.softmax(model(images), dim=1)
            out_hflip = F.softmax(model(torch.flip(images, dims=[3])), dim=1)
            out_vflip = F.softmax(model(torch.flip(images, dims=[2])), dim=1)
            out_hvflip = F.softmax(model(torch.flip(images, dims=[2, 3])), dim=1)
            avg_probs = (out_orig + out_hflip + out_vflip + out_hvflip) / 4.0

            preds = torch.argmax(avg_probs, dim=1)

            all_probs.extend(avg_probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # --------------------------------------------------------------------------
    # METRICS COMPUTATION
    # --------------------------------------------------------------------------
    acc = accuracy_score(all_targets, all_preds) * 100
    qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    binary_targets = (all_targets >= 2).astype(int)
    binary_preds = (all_preds >= 2).astype(int)
    bin_acc = accuracy_score(binary_targets, binary_preds) * 100
    bin_sens = recall_score(binary_targets, binary_preds, zero_division=0) * 100
    bin_spec = recall_score(binary_targets == 0, binary_preds == 0, zero_division=0) * 100
    bin_f1 = f1_score(binary_targets, binary_preds, zero_division=0)

    print("\n" + "=" * 60)
    print("                EVALUATION SUMMARY RESULTS")
    print("=" * 60)
    print(f"5-Class Accuracy:                {acc:.2f}%")
    print(f"Quadratic Weighted Kappa (QWK):  {qwk:.4f}")
    print(f"Macro F1-Score:                  {macro_f1:.4f}")
    print("-" * 60)
    print(f"Referable DR Accuracy (Binary):  {bin_acc:.2f}%")
    print(f"Referable DR Sensitivity/Recall: {bin_sens:.2f}%")
    print(f"Referable DR Specificity:        {bin_spec:.2f}%")
    print(f"Referable DR F1-Score:           {bin_f1:.4f}")
    print("=" * 60 + "\n")

    # --------------------------------------------------------------------------
    # PLOT 1: MULTI-CLASS ROC CURVES & AUC
    # --------------------------------------------------------------------------
    print("[PLOT 1/4] Generating Multi-Class ROC-AUC Curves...")
    plt.figure(figsize=(9, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    auc_scores = {}

    for i in range(5):
        y_true_binary = (all_targets == i).astype(int)
        y_score = all_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        auc_scores[i] = roc_auc
        plt.plot(fpr, tpr, color=colors[i], lw=2.2, label=f"{GRADE_NAMES[i]} (AUC = {roc_auc:.3f})")

    # Macro-average AUC
    macro_auc = np.mean(list(auc_scores.values()))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guessing (AUC = 0.500)')
    plt.title(f"Multi-Class ROC Curves for Diabetic Retinopathy\nMacro-Average AUC = {macro_auc:.3f}", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p1_path = os.path.join(output_dir, "1_multiclass_roc_auc.png")
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {p1_path}")

    # --------------------------------------------------------------------------
    # PLOT 2: NORMALIZED CONFUSION MATRIX
    # --------------------------------------------------------------------------
    print("[PLOT 2/4] Generating Normalized Confusion Matrix Heatmap...")
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8.5, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=GRADE_SHORT, yticklabels=GRADE_SHORT,
                cbar_kws={'label': 'Proportion of Actual Class'}, linewidths=0.5)
    plt.title(f"Normalized Confusion Matrix (5-Class Severity)\nQuadratic Weighted Kappa (QWK) = {qwk:.4f}", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Severity Grade", fontsize=11, fontweight='bold')
    plt.ylabel("True Clinical Grade", fontsize=11, fontweight='bold')
    plt.tight_layout()
    p2_path = os.path.join(output_dir, "2_confusion_matrix_5class.png")
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {p2_path}")

    # --------------------------------------------------------------------------
    # PLOT 3: PER-CLASS PRECISION, RECALL, F1 BAR CHART
    # --------------------------------------------------------------------------
    print("[PLOT 3/4] Generating Per-Grade Metrics Bar Chart...")
    precisions = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recalls = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1s = f1_score(all_targets, all_preds, average=None, zero_division=0)

    x = np.arange(5)
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - width, precisions * 100, width, label='Precision', color='#3498db')
    r2 = ax.bar(x, recalls * 100, width, label='Recall (Sensitivity)', color='#2ecc71')
    r3 = ax.bar(x + width, f1s * 100, width, label='F1-Score', color='#e67e22')

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Grade Performance: Precision, Recall & F1-Score', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(GRADE_SHORT, fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.set_ylim(0, 105)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    autolabel(r1)
    autolabel(r2)
    autolabel(r3)

    plt.tight_layout()
    p3_path = os.path.join(output_dir, "3_per_grade_metrics_barchart.png")
    plt.savefig(p3_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {p3_path}")

    # --------------------------------------------------------------------------
    # PLOT 4: REFERABLE DR CLINICAL CONFUSION MATRIX
    # --------------------------------------------------------------------------
    print("[PLOT 4/4] Generating Referable DR Clinical Screening Heatmap...")
    bin_cm = confusion_matrix(binary_targets, binary_preds)
    bin_cm_norm = bin_cm.astype('float') / bin_cm.sum(axis=1)[:, np.newaxis]

    labels_text = [
        [f"{bin_cm[0,0]}\n({bin_cm_norm[0,0]*100:.1f}%)", f"{bin_cm[0,1]}\n({bin_cm_norm[0,1]*100:.1f}%)"],
        [f"{bin_cm[1,0]}\n({bin_cm_norm[1,0]*100:.1f}%)", f"{bin_cm[1,1]}\n({bin_cm_norm[1,1]*100:.1f}%)"]
    ]

    plt.figure(figsize=(7.5, 6))
    sns.heatmap(bin_cm_norm, annot=labels_text, fmt='', cmap='Greens',
                xticklabels=["Non-Referable\n(Grades 0-1)", "Referable\n(Grades 2-4)"],
                yticklabels=["Non-Referable\n(Grades 0-1)", "Referable\n(Grades 2-4)"],
                cbar_kws={'label': 'Proportion'}, linewidths=0.5)
    plt.title(f"Clinical Screening: Referable Diabetic Retinopathy\nAccuracy: {bin_acc:.2f}% | Sensitivity: {bin_sens:.2f}% | Specificity: {bin_spec:.2f}%",
              fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Model Screening Decision", fontsize=11, fontweight='bold')
    plt.ylabel("Actual Medical Ground Truth", fontsize=11, fontweight='bold')
    plt.tight_layout()
    p4_path = os.path.join(output_dir, "4_referable_dr_clinical_matrix.png")
    plt.savefig(p4_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {p4_path}")

    print("\n[OK] All 4 publication-quality figures successfully saved in:", os.path.abspath(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visual Evaluation Plots for DR Model")
    parser.add_argument("--model-path", type=str, default="best_retinopathy_model.pth", help="Path to trained .pth model file")
    parser.add_argument("--arch", type=str, default="efficientnet_b2", help="Model architecture")
    parser.add_argument("--img-size", type=int, default=288, help="Image resolution")
    parser.add_argument("--samples", type=int, default=15000, help="Sample count used in training split")
    parser.add_argument("--output-dir", type=str, default="evaluation_plots", help="Directory where PNG plots are saved")
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        arch=args.arch,
        img_size=args.img_size,
        samples=args.samples,
        output_dir=args.output_dir
    )
