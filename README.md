# Automated Diabetic Retinopathy Severity Grading and Clinical Triage System

An end-to-end deep learning pipeline, explainable AI evaluation suite, and clinical decision support system for multi-class diabetic retinopathy severity grading and referable screening from fundus photography.

---

## Executive Summary

Diabetic Retinopathy (DR) is the leading cause of preventable blindness among working-age adults globally, impacting over one-third of individuals diagnosed with diabetes. Due to the scarcity of retinal specialists in high-prevalence and underserved regions, manual screening pipelines encounter severe diagnostic backlogs, leading to late-stage disease detection and irreversible vision loss.

This repository provides an automated, medically validated deep learning framework developed to classify retinal fundus images into the five internationally recognized stages of diabetic retinopathy:
* **Grade 0**: No Diabetic Retinopathy (Healthy Retina)
* **Grade 1**: Mild Non-Proliferative Diabetic Retinopathy (Microaneurysms only)
* **Grade 2**: Moderate Non-Proliferative Diabetic Retinopathy (More than microaneurysms, but less than severe)
* **Grade 3**: Severe Non-Proliferative Diabetic Retinopathy (4-2-1 rule: extensive hemorrhages, venous beading, IRMA)
* **Grade 4**: Proliferative Diabetic Retinopathy (Neovascularization, vitreous/preretinal hemorrhage)

In addition to 5-class severity grading, the system evaluates binary clinical referability (Non-Referable: Grades 0-1 vs. Referable: Grades 2-4) to facilitate triage in primary care environments. The architecture is paired with Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize pathological biomarkers (hard exudates, hemorrhages, cotton-wool spots, vascular abnormalities) and an interactive clinical screening dashboard built with Streamlit.

---

## System Architecture and Engineering Highlights

### 1. High-Performance Hardware-Accelerated Data Ingestion
* **Multi-Source Dataset Standardization**: Aggregates and harmonizes diverse retinal imaging repositories:
  * EyePACS / Kaggle Diabetic Retinopathy Detection Challenge
  * APTOS 2019 Blindness Detection
  * Indian Diabetic Retinopathy Image Dataset (IDRiD)
* **OpenCV C++ Ingestion**: Utilizes SIMD-accelerated C++ image decoding via OpenCV rather than Python PIL disk operations, achieving lower I/O latency.
* **Balanced Stratified Sampling**: Addresses extreme natural prevalence imbalances (Grade 0 dominance) through stratified sampling that preserves critical minority disease classes (Grades 1, 3, 4) while sampling representative cohorts of Grades 0 and 2.
* **PyTorch Multi-Worker Streaming**: Employs parallel worker threads, page-locked memory (`pin_memory=True`), and GPU prefetching for DataLoader throughput.

### 2. Deep Convolutional Backbone and Training Dynamics
* **Backbone**: EfficientNet-B2 pretrained on ImageNet, offering strong feature extraction density at an input resolution of 288x288 pixels.
* **Custom Classifier**: High-dimensional dropout regularization ($p = 0.40$) connected linearly to a 5-unit classification projection head.
* **Mixed Precision Acceleration**: Full Automatic Mixed Precision (AMP FP16) leveraging NVIDIA Tensor Cores for memory optimization and execution speedup.
* **Two-Phase Progressive Transfer Learning**:
  * *Phase 1 (Warmup, Epochs 1-2)*: Backbone feature stages are frozen; only the classification head weights are updated to stabilize gradient trajectories.
  * *Phase 2 (Selective Fine-Tuning, Epochs 3+)*: Top disease-salient feature stages (Stages 6 to 8) are unfrozen to adapt high-level visual representations while keeping early low-level edges/textures frozen to preserve stability and conserve VRAM.
* **Loss Function**: Class-weighted Cross-Entropy Loss integrated with Label Smoothing ($\epsilon = 0.10$) to prevent model overconfidence and encourage inter-class calibration along ordinal severity boundaries.
* **Optimizer and Scheduling**: AdamW optimizer ($lr = 3 \times 10^{-4}$, weight decay $= 10^{-2}$) modulated by a `ReduceLROnPlateau` scheduler monitoring validation Quadratic Weighted Kappa.

---

## Clinical Evaluation and Metrics

Because clinical grading is inherently ordinal, standard top-1 classification accuracy alone is an insufficient metric. The evaluation framework prioritizes:

1. **Quadratic Weighted Kappa (QWK)**: Measures inter-rater agreement between model predictions and clinical consensus ground truth, penalizing distance quadratically along the severity spectrum.
2. **Multi-Class Area Under ROC Curve (AUC)**: Evaluates discriminator confidence across each individual grade as well as macro-averaged performance.
3. **Confusion Matrix Analysis**: Quantifies prediction behavior to ensure errors are predominantly off-by-one (e.g., Grade 2 predicted as Grade 1 or 3) rather than cross-category failures.
4. **Clinical Referable DR Triage**: Groups predictions into binary screening actionability (Non-Referable: Grades 0-1 vs. Referable: Grades 2-4) to measure clinical sensitivity and specificity for vision-threatening retinopathy.

### Metric Visualizations Included in Repository
The evaluation suite outputs publication-ready figures located in `evaluation_plots/`:
* `1_multiclass_roc_auc.png`: One-vs-Rest ROC curves per disease grade and macro-averaged AUC.
* `2_confusion_matrix_5class.png`: Normalized 5x5 confusion matrix demonstrating ordinal clustering.
* `3_per_grade_metrics_barchart.png`: Comparison of precision, recall, and F1-score across all 5 grades.
* `4_referable_dr_clinical_matrix.png`: Binary clinical referral screening confusion matrix.
* `gradcam_result.png`: Interpretability heatmap demonstrating visual localization on retinal lesions.

---

## Explainable AI (Grad-CAM)

Clinical deployment requires transparency. The framework incorporates Gradient-weighted Class Activation Mapping (Grad-CAM) on the final convolutional stage of EfficientNet (`model.features[-1]`). 

During inference:
1. Gradients of the predicted severity class score with respect to feature activation maps are extracted via backward hooks.
2. Feature channels are weighted by their average global pooled gradients.
3. A rectified linear combination generates a 2D spatial heatmap of the most salient lesion areas.
4. Heatmaps are bilinearly upsampled, color-mapped (JET/Inferno), and overlaid on the preprocessed fundus photograph, providing ophthalmologists with an interpretable visual audit trail.

---

## Clinical Screening Dashboard (`app.py`)

An interactive, clean white-themed clinical dashboard built with Streamlit provides a turnkey point-of-care screening interface:
* **Patient Intake and Fundus Loading**: Upload local JPEG/PNG fundus photographs or select standardized validation samples.
* **Real-Time Classification**: Instant multi-class inference generating predicted grade, confidence percentage, and full probability distribution.
* **Automated Clinical Triage Assessment**:
  * **Non-Referable**: Routine annual surveillance recommended.
  * **Referable**: Vision-threatening retinopathy suspected; urgent clinical referral to an ophthalmology specialist generated.
* **Integrated Explainability**: Displays side-by-side original fundus image, processed input, and Grad-CAM localization overlay.
* **Holistic Systemic Risk Module**: Captures patient systemic metrics (HbA1c, blood pressure, duration of diabetes) to contextualize retinal findings against nephropathy and cardiovascular risk factors.

---

## Repository Structure

```text
.
|-- app.py                                            # Streamlit clinical screening and triage application
|-- student_pipeline.py                               # High-performance PyTorch training and validation pipeline
|-- generate_visuals.py                               # Script to compute and render publication-grade evaluation metrics
|-- best_retinopathy_model.pth                        # Trained PyTorch model checkpoint (EfficientNet-B2)
|-- requirements.txt                                  # Environment dependencies
|-- .gitignore                                        # Excludes datasets, checkpoints, and cache directories
|-- evaluation_plots/                                 # High-resolution clinical evaluation figures
|   |-- 1_multiclass_roc_auc.png
|   |-- 2_confusion_matrix_5class.png
|   |-- 3_per_grade_metrics_barchart.png
|   `-- 4_referable_dr_clinical_matrix.png
`-- gradcam_result.png                                # Sample Grad-CAM lesion localization visual
```

---

## Installation and Setup

### Prerequisites
* Python 3.9 or higher
* CUDA-capable GPU recommended for model training and rapid inference (CUDA 11.8 or 12.x)

### 1. Clone the Repository
```bash
git clone https://github.com/christopherjeremy/Diabetic-Retinopathy-Severity-Grading.git
cd Diabetic-Retinopathy-Severity-Grading
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Running the Clinical Screening Dashboard
Launch the Streamlit web dashboard locally:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`. Upload a retinal fundus photograph to inspect model predictions, confidence scores, and Grad-CAM explanations.

### 2. Generating Evaluation Plots
To recalculate metrics and reproduce high-resolution evaluation figures:
```bash
python generate_visuals.py
```
Figures will be generated or updated inside the `evaluation_plots/` directory.

### 3. Model Training Pipeline
To train or fine-tune the model on localized datasets:
```bash
python student_pipeline.py
```
*Note: Datasets (EyePACS, APTOS 2019, IDRiD) must be extracted into their respective directories (`Dataset_1/`, `Dataset_2/`, `Dataset_3/`, `Dataset_4/`) as specified in `student_pipeline.py`.*

---

## Clinical Disease Grading Reference

| Grade | Clinical Designation | Defining Retinal Biomarkers | Clinical Recommendation |
| :---: | :--- | :--- | :--- |
| **0** | No DR | Absence of visible retinal abnormalities | Annual routine surveillance |
| **1** | Mild NPDR | Microaneurysms only | Follow-up examination in 6-12 months |
| **2** | Moderate NPDR | Microaneurysms, dot-and-blot hemorrhages, venous loops, hard exudates | Medical management and ophthalmology review within 3-6 months |
| **3** | Severe NPDR | Greater than 20 intraretinal hemorrhages in all 4 quadrants, venous beading in 2+ quadrants, prominent IRMA in 1+ quadrant (4-2-1 rule) | Urgent ophthalmology referral within 2-4 weeks |
| **4** | Proliferative DR | Neovascularization of retina or disc, vitreous hemorrhage, preretinal fibrosis | Immediate intervention (panretinal photocoagulation / anti-VEGF therapy) |

---

## Author and Project Information

* **Author**: Christopher Jeremy A W
* **Program**: Postgraduate Diploma in Data Analytics (PGDA) - Applied Deep Learning
* **Repository**: [https://github.com/christopherjeremy/Diabetic-Retinopathy-Severity-Grading](https://github.com/christopherjeremy/Diabetic-Retinopathy-Severity-Grading)
* **License**: MIT License (or institutional standard)

---

## Disclaimer
This software is developed as part of an academic capstone research project in applied deep learning and medical computer vision. It is intended for educational, research, and assistive screening validation purposes and is not a replacement for formal clinical evaluation by certified eye care professionals or ophthalmologists.
