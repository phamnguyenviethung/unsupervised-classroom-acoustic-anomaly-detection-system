import numpy as np
from typing import List, Tuple

DEFAULT_BANDS = [
    (80.0, 300.0),
    (300.0, 1000.0),
    (1000.0, 2500.0),
    (2500.0, 4500.0),
    (4500.0, 6000.0),
    (6000.0, 7500.0)
]

def extract_bandpower_features(
    power_spectrogram: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    bands: List[Tuple[float, float]] = None
) -> dict:
    if bands is None:
        bands = DEFAULT_BANDS

    band_power_series = []
    feature_stats = []

    total_power = np.sum(power_spectrogram, axis=0) + 1e-10

    for low, high in bands:
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            sub_power = np.zeros(power_spectrogram.shape[1])
        else:
            sub_power = np.sum(power_spectrogram[mask, :], axis=0)

        band_power_series.append(sub_power)

        mean_p = np.mean(sub_power)
        std_p = np.std(sub_power)
        max_p = np.max(sub_power)
        min_p = np.min(sub_power)
        rel_ratio = np.mean(sub_power / total_power)

        feature_stats.extend([mean_p, std_p, max_p, min_p, rel_ratio])

    feature_vector = np.array(feature_stats, dtype=np.float32)

    return {
        "band_power_series": np.array(band_power_series),
        "feature_vector": feature_vector,
        "band_names": [f"{int(b[0])}-{int(b[1])}Hz" for b in bands]
    }

def extract_time_domain_bandpower(
    audio: np.ndarray,
    sample_rate: int = 16000,
    bands: List[Tuple[float, float]] = None
) -> np.ndarray:
    if bands is None:
        bands = DEFAULT_BANDS

    from scipy.signal import butter, filtfilt

    nyquist = 0.5 * sample_rate
    feature_stats = []

    for low, high in bands:
        try:
            l_norm = max(low / nyquist, 0.001)
            h_norm = min(high / nyquist, 0.999)
            b, a = butter(2, [l_norm, h_norm], btype="band")
            band_audio = filtfilt(b, a, audio)
            energy = band_audio ** 2

            feature_stats.extend([
                np.mean(energy),
                np.std(energy),
                np.max(energy),
                np.min(energy),
                np.percentile(energy, 95)
            ])
        except Exception:
            feature_stats.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    return np.array(feature_stats, dtype=np.float32)

def extract_baseline_representation(
    audio: np.ndarray,
    target_dim: int = 64
) -> np.ndarray:
    # Downsample audio into chunked block energies
    block_size = len(audio) // target_dim
    if block_size < 1:
        block_size = 1

    blocks = []
    for i in range(target_dim):
        start = i * block_size
        chunk = audio[start : start + block_size]
        blocks.append(np.sqrt(np.mean(chunk**2) + 1e-10))

    blocks = np.array(blocks, dtype=np.float32)
    envelope_stats = np.array([np.mean(audio**2), np.max(np.abs(audio)), np.std(audio)], dtype=np.float32)
    return np.concatenate([blocks, envelope_stats])
