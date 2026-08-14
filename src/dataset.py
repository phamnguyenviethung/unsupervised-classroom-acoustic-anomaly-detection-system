import os
import glob
import pandas as pd
import numpy as np
from src.audio import create_synthetic_audio, sf

def find_class_directories(data_root: str) -> tuple[str, str]:
    normal_candidates = ["normal", "normal_audio", "clean", "classroom_normal"]
    abnormal_candidates = ["abnormal", "anomalous", "anomaly", "classroom_abnormal"]

    normal_dir = os.path.join(data_root, "normal")
    abnormal_dir = os.path.join(data_root, "abnormal")

    for cand in normal_candidates:
        p = os.path.join(data_root, cand)
        if os.path.exists(p) and len(glob.glob(os.path.join(p, "*.wav"))) > 0:
            normal_dir = p
            break

    for cand in abnormal_candidates:
        p = os.path.join(data_root, cand)
        if os.path.exists(p) and len(glob.glob(os.path.join(p, "*.wav"))) > 0:
            abnormal_dir = p
            break

    return normal_dir, abnormal_dir

def bootstrap_synthetic_dataset_if_empty(
    data_root: str = "data",
    num_normal: int = 40,
    num_abnormal: int = 15,
    sample_rate: int = 16000,
    duration_seconds: float = 5.0
):
    normal_dir = os.path.join(data_root, "normal")
    abnormal_dir = os.path.join(data_root, "abnormal")

    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)

    norm_existing = glob.glob(os.path.join(normal_dir, "*.wav"))
    abnorm_existing = glob.glob(os.path.join(abnormal_dir, "*.wav"))

    if len(norm_existing) == 0:
        for i in range(1, num_normal + 1):
            audio = create_synthetic_audio("normal", sample_rate, duration_seconds, seed=i)
            sf.write(os.path.join(normal_dir, f"normal_{i:04d}.wav"), audio, sample_rate)

    if len(abnorm_existing) == 0:
        for i in range(1, num_abnormal + 1):
            audio = create_synthetic_audio("abnormal", sample_rate, duration_seconds, seed=i + 1000)
            sf.write(os.path.join(abnormal_dir, f"abnormal_{i:04d}.wav"), audio, sample_rate)

def build_dataset_manifest(data_root: str = "data") -> pd.DataFrame:
    bootstrap_synthetic_dataset_if_empty(data_root)
    normal_dir, abnormal_dir = find_class_directories(data_root)

    records = []

    for f in sorted(glob.glob(os.path.join(normal_dir, "*.wav"))):
        records.append({
            "filename": os.path.basename(f),
            "file_path": f,
            "class": "normal",
            "label": 0
        })

    for f in sorted(glob.glob(os.path.join(abnormal_dir, "*.wav"))):
        records.append({
            "filename": os.path.basename(f),
            "file_path": f,
            "class": "abnormal",
            "label": 1
        })

    df = pd.DataFrame(records)
    return df

def split_dataset_unsupervised(
    df_manifest: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_normal = df_manifest[df_manifest["label"] == 0].copy()
    df_abnormal = df_manifest[df_manifest["label"] == 1].copy()

    df_normal = df_normal.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_train = int(len(df_normal) * train_ratio)
    df_train_normal = df_normal.iloc[:n_train].copy()
    df_eval_normal = df_normal.iloc[n_train:].copy()

    # Unsupervised Evaluation set: Remaining Normal + ALL Abnormal samples
    df_eval_combined = pd.concat([df_eval_normal, df_abnormal], ignore_index=True)
    df_eval_combined = df_eval_combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df_train_normal, df_eval_combined
