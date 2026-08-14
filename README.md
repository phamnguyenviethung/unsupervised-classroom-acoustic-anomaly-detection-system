# Unsupervised Classroom Acoustic Anomaly Detection using DSP and AI

**FPT University HCMC — DSP501 Digital Signal & Image Processing Final Assignment | August 2026**

---

## 📌 Executive Summary & Research Scope

This repository provides a complete, research-grade software implementation for **Unsupervised Classroom Acoustic Anomaly Detection**. The application integrates digital signal processing (DSP) techniques—including zero-phase IIR bandpass filtering, Short-Time Fourier Transform (STFT), and spectral sub-band power extraction—with unsupervised machine learning models (a PyTorch Deep Autoencoder).

The central scientific goal is to evaluate whether structured DSP preprocessing improves unsupervised acoustic anomaly detection performance and behavior under increasing additive noise compared with a minimal-DSP baseline, and to quantify the relative contribution of each DSP stage via an ablation study.

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
4. The anomaly threshold is derived solely from the 97th percentile of normal training reconstruction scores.

---

## 🔬 Five Pipeline Variants (Ablation Architecture)

1. **`full_dsp` (Proposed):** `Raw Audio` ➔ `Resample/Mono` ➔ `Butterworth Bandpass Filter` ➔ `STFT` ➔ `Spectral Band-Power Features` ➔ `StandardScaler` ➔ `Unsupervised Autoencoder`
2. **`without_bandpass`:** `Raw Audio` ➔ `Resample/Mono` ➔ `STFT` ➔ `Spectral Band-Power Features` ➔ `StandardScaler` ➔ `Unsupervised Autoencoder`
3. **`without_bandpower`:** `Raw Audio` ➔ `Resample/Mono` ➔ `Butterworth Bandpass Filter` ➔ `STFT` ➔ `Mean Spectrum Vector` ➔ `StandardScaler` ➔ `Unsupervised Autoencoder`
4. **`without_stft`:** `Raw Audio` ➔ `Resample/Mono` ➔ `Butterworth Bandpass Filter` ➔ `Time-Domain Band Energy` ➔ `StandardScaler` ➔ `Unsupervised Autoencoder`
5. **`baseline` (Pipeline A):** `Raw Audio` ➔ `Resample/Mono` ➔ `Minimal Raw Waveform Representation` ➔ `StandardScaler` ➔ `Unsupervised Autoencoder`

---

## 📊 Main Results

The reported experiments used **50 training epochs** and a **97th-percentile** reconstruction-error threshold.

The best clean-condition performance was obtained by **`without_bandpower`**, which retained the richer 513-dimensional STFT mean-spectrum representation instead of compressing the spectrum into six band-power features.

| Pipeline | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| **without_bandpower** | **0.847** | **0.950** | **0.687** | **0.797** | **0.895** | **0.912** | **0.028** |
| baseline | 0.614 | 0.619 | 0.313 | 0.416 | 0.579 | 0.615 | 0.151 |
| full_dsp | 0.619 | 0.677 | 0.253 | 0.368 | 0.624 | 0.616 | 0.094 |
| without_bandpass | 0.619 | 0.690 | 0.241 | 0.357 | 0.652 | 0.614 | 0.085 |
| without_stft | 0.577 | 0.588 | 0.120 | 0.200 | 0.459 | 0.436 | 0.066 |

**Key findings**

- `without_bandpower` achieved the strongest overall clean-condition performance.
- Preserving richer spectral information substantially outperformed the compact band-power representation in this experiment.
- Removing STFT produced the weakest discrimination performance.
- Removing bandpass filtering had a comparatively small and mixed effect.
- Increasing additive noise substantially increased false-alarm behavior in several configurations.

The results therefore do **not** support the claim that a more complex DSP pipeline is automatically better.

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

*Note:* The reported experiments use the prepared dataset described in the research report. Synthetic bootstrap data, if used for demonstration, are not part of the reported evaluation.

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
python run.py --data data --sample-rate 16000 --seed 42 --epochs 50 --threshold-percentile 97.0
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

- **Ethics & Privacy:** The project uses publicly available audiovisual/audio resources for research purposes. YouTube-derived material was used as source material for audio samples, while abnormal sound resources included material obtained from Pixabay. The study does not attempt to identify individuals or infer personal characteristics. Any future deployment using real classroom recordings should address consent, privacy, secure data handling, and applicable institutional requirements.
- **AI Tool Usage:** ChatGPT, NotebookLM, and Google AI Studio/Gemini were used as supporting tools for research planning, literature review, academic writing, technical coding support, result interpretation, and figure/table preparation. The group remained responsible for implementation, data preparation, debugging, experimental decisions, verification, and the final report. Experimental results were generated from the actual project execution and were checked by the group before submission.
