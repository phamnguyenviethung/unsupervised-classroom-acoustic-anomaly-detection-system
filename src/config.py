import os
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class AppConfig:
    data_root: str = "data"
    normal_folder: str = "normal"
    abnormal_folder: str = "abnormal"
    sample_rate: int = 16000
    duration_seconds: float = 5.0
    train_ratio: float = 0.8
    normalize_audio: bool = True

    low_cut: float = 80.0
    high_cut: float = 7500.0
    filter_order: int = 4
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    bands: List[Tuple[float, float]] = field(default_factory=lambda: [
        (80.0, 300.0),
        (300.0, 1000.0),
        (1000.0, 2500.0),
        (2500.0, 4500.0),
        (4500.0, 6000.0),
        (6000.0, 7500.0)
    ])

    model_type: str = "autoencoder"
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 40
    threshold_percentile: float = 99.0

    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.05, 0.10])
    random_seed: int = 42

    artifacts_dir: str = "artifacts"
    results_dir: str = "results"

    def to_dict(self):
        return {
            "data_root": self.data_root,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "low_cut": self.low_cut,
            "high_cut": self.high_cut,
            "filter_order": self.filter_order,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "model_type": self.model_type,
            "hidden_dims": self.hidden_dims,
            "epochs": self.epochs,
            "threshold_percentile": self.threshold_percentile,
            "random_seed": self.random_seed
        }

def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    cfg = AppConfig()
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        ds = data.get("dataset", {})
        cfg.data_root = ds.get("data_root", cfg.data_root)
        cfg.normal_folder = ds.get("normal_folder", cfg.normal_folder)
        cfg.abnormal_folder = ds.get("abnormal_folder", cfg.abnormal_folder)
        cfg.sample_rate = int(ds.get("sample_rate", cfg.sample_rate))
        cfg.duration_seconds = float(ds.get("duration_seconds", cfg.duration_seconds))
        cfg.train_ratio = float(ds.get("train_ratio", cfg.train_ratio))
        cfg.normalize_audio = bool(ds.get("normalize_audio", cfg.normalize_audio))

        dsp = data.get("dsp", {})
        cfg.low_cut = float(dsp.get("low_cut", cfg.low_cut))
        cfg.high_cut = float(dsp.get("high_cut", cfg.high_cut))
        cfg.filter_order = int(dsp.get("filter_order", cfg.filter_order))
        cfg.n_fft = int(dsp.get("n_fft", cfg.n_fft))
        cfg.hop_length = int(dsp.get("hop_length", cfg.hop_length))
        cfg.win_length = int(dsp.get("win_length", cfg.win_length))
        if "bands" in dsp:
            cfg.bands = [tuple(b) for b in dsp["bands"]]

        m = data.get("model", {})
        cfg.model_type = str(m.get("type", cfg.model_type))
        cfg.hidden_dims = m.get("hidden_dims", cfg.hidden_dims)
        cfg.learning_rate = float(m.get("learning_rate", cfg.learning_rate))
        cfg.batch_size = int(m.get("batch_size", cfg.batch_size))
        cfg.epochs = int(m.get("epochs", cfg.epochs))
        cfg.threshold_percentile = float(m.get("threshold_percentile", cfg.threshold_percentile))

        rob = data.get("robustness", {})
        if "noise_levels" in rob:
            cfg.noise_levels = [float(x) for x in rob["noise_levels"]]

        rep = data.get("reproducibility", {})
        cfg.random_seed = int(rep.get("seed", cfg.random_seed))

        paths = data.get("paths", {})
        cfg.artifacts_dir = str(paths.get("artifacts_dir", cfg.artifacts_dir))
        cfg.results_dir = str(paths.get("results_dir", cfg.results_dir))

    return cfg
