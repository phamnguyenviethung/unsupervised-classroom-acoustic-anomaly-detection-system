import os
import sys
import glob
import pandas as pd
import numpy as np
import streamlit as st

from src.config import AppConfig, load_config
from src.audio import load_and_preprocess_audio, create_synthetic_audio
from src.dataset import build_dataset_manifest, split_dataset_unsupervised, find_class_directories
from src.pipelines import PIPELINE_VARIANTS, extract_pipeline_features, extract_features_from_files
from src.models import UnsupervisedAnomalyDetector
from src.training import train_and_evaluate_all_pipelines
from src.evaluation import calculate_quantitative_metrics

st.set_page_config(
    page_title="Classroom Acoustic Anomaly Detection System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎙️ Unsupervised Classroom Acoustic Anomaly Detection System")
st.caption("FPT University HCMC — DSP501 Digital Signal & Image Processing Final Assignment | Research & Demonstration Dashboard")

# Initialize Session State Configuration
if "config" not in st.session_state:
    st.session_state.config = load_config()

cfg = st.session_state.config

# ==========================================
# SIDEBAR CONTROL PANEL
# ==========================================
st.sidebar.header("⚙️ Experiment Configuration")

# 1. Dataset Settings
with st.sidebar.expander("📁 Dataset Settings", expanded=True):
    data_root = st.text_input("Dataset Root Folder", value=cfg.data_root)
    cfg.data_root = data_root
    normal_dir, abnormal_dir = find_class_directories(data_root)
    st.caption(f"Normal: `{normal_dir}`")
    st.caption(f"Abnormal: `{abnormal_dir}`")

# 2. DSP Settings
with st.sidebar.expander("🎛️ DSP Parameters", expanded=False):
    sample_rate = st.selectbox("Sample Rate (Hz)", [8000, 16000, 22050, 32000, 44100, 48000], index=1)
    cfg.sample_rate = sample_rate

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        low_cut = st.number_input("Low Cut (Hz)", min_value=10.0, max_value=2000.0, value=float(cfg.low_cut), step=10.0)
    with col_f2:
        high_cut = st.number_input("High Cut (Hz)", min_value=100.0, max_value=20000.0, value=float(cfg.high_cut), step=100.0)
    cfg.low_cut = low_cut
    cfg.high_cut = high_cut

    filter_order = st.slider("Filter Order", min_value=1, max_value=8, value=int(cfg.filter_order))
    cfg.filter_order = filter_order

    stft_nfft = st.selectbox("STFT n_fft", [256, 512, 1024, 2048], index=2)
    cfg.n_fft = stft_nfft
    cfg.hop_length = stft_nfft // 4
    cfg.win_length = stft_nfft

# 3. Model & Anomaly Settings
with st.sidebar.expander("🤖 Anomaly Model", expanded=False):
    model_type = st.selectbox("Model Type", ["autoencoder", "isolation_forest"], index=0)
    cfg.model_type = model_type
    epochs = st.number_input("Epochs", min_value=5, max_value=200, value=int(cfg.epochs), step=5)
    cfg.epochs = epochs
    percentile = st.slider("Threshold Percentile (Normal Train)", min_value=90.0, max_value=99.9, value=float(cfg.threshold_percentile), step=0.5)
    cfg.threshold_percentile = percentile

# 4. Reproducibility
with st.sidebar.expander("🎲 Seed & Noise", expanded=False):
    seed = st.number_input("Random Seed", value=int(cfg.random_seed), step=1)
    cfg.random_seed = seed


# ==========================================
# MAIN DASHBOARD TABS
# ==========================================
tab_retrain, tab_demo, tab_analytics, tab_ablation, tab_robustness, tab_errors = st.tabs([
    "🔄 Training & Retrain",
    "🎙️ Dual Pipeline Inference",
    "📊 Quantitative Analytics",
    "🧪 Ablation Study",
    "🔊 Noise Robustness",
    "🔍 Error Analysis"
])


# ------------------------------------------
# TAB 1: RETRAIN & EXPERIMENT STATUS
# ------------------------------------------
with tab_retrain:
    st.header("🔄 Retrain All Five DSP Pipeline Variants")
    st.markdown("""
    Click the button below to trigger the complete unsupervised training pipeline across all 5 variants:
    1. **Full DSP** (`Bandpass Filter + STFT + Bandpower`)
    2. **Without Bandpass** (`STFT + Bandpower`)
    3. **Without Bandpower** (`Bandpass Filter + STFT Spectrogram`)
    4. **Without STFT** (`Bandpass Filter + Time-domain Energy`)
    5. **Baseline** (`Minimal Raw Waveform Representation`)
    """)

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        retrain_clicked = st.button("🔄 RETRAIN ALL PIPELINES", type="primary", use_container_width=True)

    if retrain_clicked:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(prog, msg):
            progress_bar.progress(prog)
            status_text.text(msg)

        with st.spinner("Executing experiment pipeline..."):
            detectors, summary_df, robustness_df = train_and_evaluate_all_pipelines(
                cfg,
                progress_callback=update_progress
            )

        st.success("🎉 Retraining and full evaluation completed successfully!")
        st.session_state["trained_detectors"] = detectors
        st.session_state["summary_df"] = summary_df
        st.session_state["robustness_df"] = robustness_df

    # Display dataset summary manifest
    st.subheader("📋 Dataset Summary Manifest")
    manifest_csv_path = os.path.join(cfg.results_dir, "dataset_manifest.csv")
    if os.path.exists(manifest_csv_path):
        m_df = pd.read_csv(manifest_csv_path)
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Audio Files", len(m_df))
        col_m2.metric("Normal Samples", len(m_df[m_df["class"] == "normal"]))
        col_m3.metric("Abnormal Samples", len(m_df[m_df["class"] == "abnormal"]))
        st.dataframe(m_df, use_container_width=True)
    else:
        st.info("No dataset manifest generated yet. Click 'RETRAIN ALL PIPELINES' above to start.")


# ------------------------------------------
# TAB 2: DUAL PIPELINE INFERENCE (LIVE / DEMO)
# ------------------------------------------
with tab_demo:
    st.header("🎙️ Dual Pipeline Side-by-Side Inference Demonstration")
    st.markdown("Compare **Pipeline A (Baseline)** vs **Pipeline B (Full DSP)** in real time on any audio input!")

    input_mode = st.radio("Select Audio Source", ["Upload Audio File (WAV)", "Microphone Recording"], horizontal=True)

    test_audio = None
    input_sr = cfg.sample_rate

    if input_mode == "Upload Audio File (WAV)":
        uploaded_file = st.file_uploader("Upload WAV file (5 seconds)", type=["wav"])
        if uploaded_file is not None:
            # Save temporary file
            temp_path = "temp_uploaded.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            test_audio, _ = load_and_preprocess_audio(
                temp_path,
                target_sample_rate=cfg.sample_rate,
                duration_seconds=cfg.duration_seconds,
                normalize=cfg.normalize_audio
            )
            st.audio(uploaded_file, format="audio/wav")

    else:
        audio_input = st.audio_input("Record 5-second classroom audio sample")
        if audio_input is not None:
            temp_path = "temp_recorded.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_input.read())
            test_audio, _ = load_and_preprocess_audio(
                temp_path,
                target_sample_rate=cfg.sample_rate,
                duration_seconds=cfg.duration_seconds,
                normalize=cfg.normalize_audio
            )
            st.audio(audio_input, format="audio/wav")

    if test_audio is not None:
        st.subheader("🔍 Dual Pipeline Analysis Results")

        artifact_base = cfg.artifacts_dir

        if os.path.exists(os.path.join(artifact_base, "full_dsp")) and os.path.exists(os.path.join(artifact_base, "baseline")):
            det_full = UnsupervisedAnomalyDetector.load(os.path.join(artifact_base, "full_dsp"))
            det_base = UnsupervisedAnomalyDetector.load(os.path.join(artifact_base, "baseline"))
        else:
            st.warning("Trained model artifacts not found. Training quick models for demo...")
            train_and_evaluate_all_pipelines(cfg)
            det_full = UnsupervisedAnomalyDetector.load(os.path.join(artifact_base, "full_dsp"))
            det_base = UnsupervisedAnomalyDetector.load(os.path.join(artifact_base, "baseline"))

        # Extract features
        feat_full, meta_full = extract_pipeline_features(test_audio, "full_dsp", cfg)
        feat_base, meta_base = extract_pipeline_features(test_audio, "baseline", cfg)

        pred_full, score_full = det_full.predict(feat_full.reshape(1, -1))
        pred_base, score_base = det_base.predict(feat_base.reshape(1, -1))

        col_p_a, col_p_b = st.columns(2)

        # Pipeline A — Baseline
        with col_p_a:
            st.markdown("### 🔴 Pipeline A — Minimal Baseline")
            dec_str_base = "🚨 ANOMALY" if pred_base[0] == 1 else "✅ NORMAL"

            st.metric("Decision", dec_str_base)
            st.write(f"**Anomaly Score:** `{score_base[0]:.6f}`")
            st.write(f"**Threshold:** `{det_base.threshold:.6f}`")
            st.line_chart(feat_base, use_container_width=True)

        # Pipeline B — Proposed Full DSP
        with col_p_b:
            st.markdown("### 🟢 Pipeline B — Proposed Full DSP")
            dec_str_full = "🚨 ANOMALY" if pred_full[0] == 1 else "✅ NORMAL"
            st.metric("Decision", dec_str_full)
            st.write(f"**Anomaly Score:** `{score_full[0]:.6f}`")
            st.write(f"**Threshold:** `{det_full.threshold:.6f}`")

            filtered_audio = meta_full.get("filtered_audio", test_audio)
            st.write("**Filtered Time-Domain Waveform:**")
            st.line_chart(filtered_audio[:2000], use_container_width=True)


# ------------------------------------------
# TAB 3: QUANTITATIVE ANALYTICS
# ------------------------------------------
with tab_analytics:
    st.header("📊 Quantitative Analytics & Comparative Performance")

    summary_path = os.path.join(cfg.results_dir, "summary_metrics.csv")
    if os.path.exists(summary_path):
        s_df = pd.read_csv(summary_path)
        st.subheader("Metrics Summary Table")
        st.dataframe(s_df, use_container_width=True)

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("ROC Curves")
            roc_path = os.path.join(cfg.results_dir, "roc_curves.png")
            if os.path.exists(roc_path):
                st.image(roc_path, use_container_width=True)
        with col_img2:
            st.subheader("Precision-Recall Curves")
            pr_path = os.path.join(cfg.results_dir, "precision_recall_curves.png")
            if os.path.exists(pr_path):
                st.image(pr_path, use_container_width=True)

        st.subheader("Confusion Matrices")
        cm_files = sorted(glob.glob(os.path.join(cfg.results_dir, "confusion_matrix_*.png")))
        if cm_files:
            cols = st.columns(len(cm_files))
            for i, cm_f in enumerate(cm_files):
                cols[i].image(cm_f, caption=os.path.basename(cm_f).replace("confusion_matrix_", "").replace(".png", ""), use_container_width=True)
    else:
        st.info("Run Retrain to populate quantitative analytics dashboard.")


# ------------------------------------------
# TAB 4: ABLATION STUDY
# ------------------------------------------
with tab_ablation:
    st.header("🧪 Ablation Study — DSP Component Contribution Analysis")
    st.markdown("""
    This section evaluates the explicit contribution of each digital signal processing stage:
    - **Full DSP** vs **Without Bandpass**: Isolates the impact of narrow Butterworth acoustic filtering.
    - **Full DSP** vs **Without Bandpower**: Isolates the impact of integrating power into acoustic frequency bands.
    - **Full DSP** vs **Without STFT**: Isolates time-domain vs frequency-domain spectral representations.
    - **Full DSP** vs **Baseline**: Demonstrates overall benefit over raw waveform feature representations.
    """)

    abl_img_path = os.path.join(cfg.results_dir, "ablation_comparison.png")
    if os.path.exists(abl_img_path):
        st.image(abl_img_path, use_container_width=True)

    summary_path = os.path.join(cfg.results_dir, "summary_metrics.csv")
    if os.path.exists(summary_path):
        s_df = pd.read_csv(summary_path)
        st.table(s_df[["pipeline", "accuracy", "f1_score", "roc_auc", "false_alarm_rate", "detection_rate"]])


# ------------------------------------------
# TAB 5: NOISE ROBUSTNESS
# ------------------------------------------
with tab_robustness:
    st.header("🔊 Noise Robustness Experiment")
    st.markdown("Evaluates performance stability under additive Gaussian noise strength levels `0.01, 0.03, 0.05, 0.10`.")

    noise_img_path = os.path.join(cfg.results_dir, "noise_robustness.png")
    if os.path.exists(noise_img_path):
        st.image(noise_img_path, use_container_width=True)

    noise_csv_path = os.path.join(cfg.results_dir, "noise_robustness.csv")
    if os.path.exists(noise_csv_path):
        n_df = pd.read_csv(noise_csv_path)
        st.dataframe(n_df, use_container_width=True)


# ------------------------------------------
# TAB 6: ERROR ANALYSIS
# ------------------------------------------
with tab_errors:
    st.header("🔍 Interactive Error Analysis & Diagnostic Workflow")
    st.markdown("Select a pipeline variant to inspect misclassified samples (False Positives & False Negatives).")

    selected_var = st.selectbox("Select Pipeline Variant for Inspection", PIPELINE_VARIANTS, index=0)

    err_dir = os.path.join(cfg.results_dir, "error_analysis")
    fp_path = os.path.join(err_dir, f"false_positives_{selected_var}.csv")
    fn_path = os.path.join(err_dir, f"false_negatives_{selected_var}.csv")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.subheader("🚨 False Positives (Normal Misclassified as Anomaly)")
        if os.path.exists(fp_path):
            fp_df = pd.read_csv(fp_path)
            st.dataframe(fp_df, use_container_width=True)
            st.caption(f"Count: {len(fp_df)}")
        else:
            st.write("No False Positives recorded.")

    with col_e2:
        st.subheader("⚠️ False Negatives (Anomaly Misclassified as Normal)")
        if os.path.exists(fn_path):
            fn_df = pd.read_csv(fn_path)
            st.dataframe(fn_df, use_container_width=True)
            st.caption(f"Count: {len(fn_df)}")
        else:
            st.write("No False Negatives recorded.")
