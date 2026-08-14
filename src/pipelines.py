import numpy as np
import pandas as pd
from src.config import AppConfig
from src.audio import load_and_preprocess_audio
from src.dsp import apply_bandpass_filter, compute_stft
from src.features import (
    extract_bandpower_features,
    extract_time_domain_bandpower,
    extract_baseline_representation
)

PIPELINE_VARIANTS = [
    "full_dsp",
    "without_bandpass",
    "without_bandpower",
    "without_stft",
    "baseline"
]

def extract_pipeline_features(audio: np.ndarray, pipeline_type: str, cfg: AppConfig) -> tuple[np.ndarray, dict]:
    meta = {"pipeline": pipeline_type}

    if pipeline_type == "full_dsp":
        # Proposed: Bandpass Filter -> STFT -> Band-power Features
        filtered, filter_info = apply_bandpass_filter(
            audio, cfg.sample_rate, cfg.low_cut, cfg.high_cut, cfg.filter_order
        )
        stft_res = compute_stft(
            filtered, cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.win_length
        )
        bp_res = extract_bandpower_features(
            stft_res["power"], stft_res["freqs"], stft_res["times"], cfg.bands
        )
        meta["filtered_audio"] = filtered
        meta["stft_res"] = stft_res
        return bp_res["feature_vector"], meta

    elif pipeline_type == "without_bandpass":
        # Ablation 1: STFT -> Band-power Features (No Butterworth filtering)
        stft_res = compute_stft(
            audio, cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.win_length
        )
        bp_res = extract_bandpower_features(
            stft_res["power"], stft_res["freqs"], stft_res["times"], cfg.bands
        )
        meta["stft_res"] = stft_res
        return bp_res["feature_vector"], meta

    elif pipeline_type == "without_bandpower":
        # Ablation 2: Bandpass Filter -> STFT -> Mean Spectrum (No band power pooling)
        filtered, _ = apply_bandpass_filter(
            audio, cfg.sample_rate, cfg.low_cut, cfg.high_cut, cfg.filter_order
        )
        stft_res = compute_stft(
            filtered, cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.win_length
        )
        # Mean spectrum over time frames
        mean_spectrum = np.mean(stft_res["power_db"], axis=1)
        meta["filtered_audio"] = filtered
        return mean_spectrum.astype(np.float32), meta

    elif pipeline_type == "without_stft":
        # Ablation 3: Bandpass Filter -> Time-Domain Band Energy (No STFT)
        filtered, _ = apply_bandpass_filter(
            audio, cfg.sample_rate, cfg.low_cut, cfg.high_cut, cfg.filter_order
        )
        time_feats = extract_time_domain_bandpower(filtered, cfg.sample_rate, cfg.bands)
        meta["filtered_audio"] = filtered
        return time_feats, meta

    elif pipeline_type == "baseline":
        # Baseline: Raw waveform representation without specialized DSP
        base_feats = extract_baseline_representation(audio, target_dim=64)
        return base_feats, meta

    else:
        raise ValueError(f"Unknown pipeline type: '{pipeline_type}'")

def extract_features_from_files(
    df_manifest: pd.DataFrame,
    pipeline_type: str,
    cfg: AppConfig,
    additive_noise_std: float = 0.0
) -> tuple[np.ndarray, np.ndarray, list]:
    X_list = []
    y_list = []
    paths_list = []

    for _, row in df_manifest.iterrows():
        audio, _ = load_and_preprocess_audio(
            row["file_path"],
            target_sample_rate=cfg.sample_rate,
            duration_seconds=cfg.duration_seconds,
            normalize=cfg.normalize_audio
        )

        # Additive Gaussian noise for robustness experiments
        if additive_noise_std > 0.0:
            noise = np.random.normal(0, additive_noise_std, len(audio)).astype(np.float32)
            audio = audio + noise

        feat_vector, _ = extract_pipeline_features(audio, pipeline_type, cfg)
        X_list.append(feat_vector)
        y_list.append(row["label"])
        paths_list.append(row["file_path"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)
    return X, y, paths_list
