# Unsupervised Classroom Acoustic Anomaly Detection using DSP and AI

**FPT University HCMC — DSP501 Digital Signal & Image Processing Final Assignment | July 2026**

---

## 📌 Executive Summary & Research Scope

This repository provides a complete, research-grade software implementation for **Unsupervised Classroom Acoustic Anomaly Detection**. The application integrates digital signal processing (DSP) techniques—including zero-phase IIR bandpass filtering, Short-Time Fourier Transform (STFT), and spectral sub-band power extraction—with unsupervised machine learning models (PyTorch Autoencoders and Isolation Forests).

The central scientific goal is to evaluate whether structured DSP preprocessing improves unsupervised acoustic anomaly detection performance and noise robustness compared to a minimal-DSP baseline, and to quantify the relative contribution of each DSP stage via an ablation study.

---

## 🎯 Research Questions & Hypotheses

### Research Questions
- **RQ1:** How does DSP preprocessing affect unsupervised classroom acoustic anomaly detection performance compared with a minimal-DSP baseline?
- **RQ2:** Which DSP component (bandpass filtering, STFT, or band-power extraction) contributes most to anomaly detection performance?
- **RQ3:** How robust are the different DSP pipelines under increasing artificial additive noise?
- **RQ4:** What acoustic characteristics are associated with false positives and false negatives?

### Hypotheses
- **H1:** The proposed full-DSP pipeline produces superior anomaly detection metrics (Accuracy, F1, ROC-AUC) and noise robustness compared to the minimal-DSP baseline.
- **H2:** Removing any individual core DSP component reduces overall classification metrics or noise resilience.

---

## 🔒 Strict Unsupervised Learning Protocol

To prevent data leakage and guarantee scientific validity:
1. The model learns the concept of "normal classroom acoustics" **ONLY** from normal training audio.
2. Abnormal dataset samples are **NEVER** used during model fitting, scaler estimation, threshold selection, or hyperparameter tuning.
3. **Split Strategy:**
   - Normal dataset: **80% Training**, **20% Evaluation**
   - Final Evaluation Set: **20% Normal Evaluation + 100% Abnormal Dataset**
4. The anomaly threshold is derived solely from the 99th percentile of normal training reconstruction scores.

---

## 🔬 Five Pipeline Variants (Ablation Architecture)

1. **`full_dsp` (Proposed):** `Raw Audio` ➔ `Resample/Mono` ➔ `Butterworth Bandpass Filter` ➔ `STFT` ➔ `Spectral Band-Power Features` ➔ `StandardScaler` ➔ `Unsupervised Autoencoder`
2. **`without_bandpass`:** `Raw Audio` ➔ `Resample/Mono` ➔ `STFT` ➔ `Spectral Band-Power Features` ➔ `StandardScaler` ➔ `Unsupervised Autoencoder`
3. **`without_bandpower`:** `Raw Audio` ➔ `Resample/Mono` ➔ `Butterworth Bandpass Filter` ➔ `STFT` ➔ `Mean Spectrum Vector` ➔ `StandardScaler` ➔ `Unsupervised Autoencoder`
4. **`without_stft`:** `Raw Audio` ➔ `Resample/Mono` ➔ `Butterworth Bandpass Filter` ➔ `Time-Domain Band Energy` ➔ `StandardScaler` ➔ `Unsupervised Autoencoder`
5. **`baseline` (Pipeline A):** `Raw Audio` ➔ `Resample/Mono` ➔ `Minimal Raw Waveform Representation` ➔ `StandardScaler` ➔ `Unsupervised Autoencoder`

---

## 📂 Dataset Directory Structure

```text
data/
├── normal/
│   ├── normal_0001.wav
│   ├── normal_0002.wav
│   └── ...
└── abnormal/
    ├── abnormal_0001.wav
    ├── abnormal_0002.wav
    └── ...
```

*Note:* If `data/` is empty upon first run, the system automatically bootstraps synthetic normal and abnormal classroom audio samples so the system functions immediately out of the box!

---

## 🚀 Quick Start Guide

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Reproducible Terminal Execution (CLI)
To run the complete experiment suite, train all 5 variants, and generate report outputs:
```bash
python run.py
```

Useful CLI arguments:
```bash
python run.py --data data --sample-rate 16000 --seed 42 --epochs 40 --threshold-percentile 99.0
```

### 3. Interactive Web Interface (Streamlit GUI)
To launch the interactive dashboard and dual pipeline live inference demo:
```bash
streamlit run app.py
```

---

## 📊 Experimental Results & Outputs

Execution automatically generates structured artifacts in `results/`:
- `summary_metrics.csv`: Quantitative metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FAR, DR) across all 5 variants.
- `dataset_manifest.csv`: Audit log of all audio files.
- `noise_robustness.csv`: Performance metrics across additive noise levels `0.01, 0.03, 0.05, 0.10`.
- `thresholds.csv`: Derived decision thresholds.
- `roc_curves.png` & `precision_recall_curves.png`: Publication figures.
- `confusion_matrix_{pipeline}.png`: Confusion matrix heatmaps.
- `ablation_comparison.png` & `noise_robustness.png`: Comparative charts.
- `error_analysis/`: Per-sample prediction records, False Positive, and False Negative diagnostic logs.

---

## 🛡️ Ethics, Privacy & AI Declaration

- **Ethics & Privacy:** Classroom audio recordings may capture human speech. All dataset recordings must be anonymized, collected with informed consent where required, and stored securely without personally identifiable metadata.
- **AI Tool Usage Declaration:** Developed using AI Studio Agent assisting in DSP pipeline engineering, Python software architecture, and visualization layout. Human engineering verification was performed on all mathematical formulas, filter frequency constraints, and unsupervised data leakage protocols.
