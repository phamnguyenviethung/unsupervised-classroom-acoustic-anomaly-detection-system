import os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def load_and_preprocess_audio(
    file_path: str,
    target_sample_rate: int = 16000,
    duration_seconds: float = 5.0,
    normalize: bool = True
) -> tuple[np.ndarray, dict]:
    metadata = {
        "original_path": file_path,
        "valid": False,
        "original_sample_rate": 0,
        "original_channels": 1,
        "original_num_samples": 0
    }

    try:
        data, sr = sf.read(file_path, dtype="float32")
        metadata["original_sample_rate"] = sr
        metadata["original_num_samples"] = len(data)

        # Convert stereo to mono
        if data.ndim > 1:
            metadata["original_channels"] = data.shape[1]
            data = np.mean(data, axis=1)

        # Resample if sample rate differs
        if sr != target_sample_rate and sr > 0:
            gcd = np.gcd(sr, target_sample_rate)
            up = target_sample_rate // gcd
            down = sr // gcd
            data = resample_poly(data, up, down).astype(np.float32)

        target_length = int(target_sample_rate * duration_seconds)

        # Truncate or Pad to exact length
        if len(data) > target_length:
            data = data[:target_length]
        elif len(data) < target_length:
            pad_length = target_length - len(data)
            data = np.pad(data, (0, pad_length), mode="constant")

        # Peak Normalization
        if normalize and np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))

        metadata["valid"] = True
        return data, metadata

    except Exception as e:
        metadata["error"] = str(e)
        target_length = int(target_sample_rate * duration_seconds)
        return np.zeros(target_length, dtype=np.float32), metadata

def create_synthetic_audio(
    category: str = "normal",
    sample_rate: int = 16000,
    duration_seconds: float = 5.0,
    seed: int = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)

    if category == "normal":
        # Background ambient classroom sound (low hum + mild speech harmonics)
        hum = 0.05 * np.sin(2 * np.pi * 50.0 * t) + 0.03 * np.sin(2 * np.pi * 120.0 * t)
        speech = 0.08 * np.sin(2 * np.pi * 350.0 * t) * np.sin(2 * np.pi * 2.0 * t)
        white_noise = np.random.normal(0, 0.02, len(t))
        audio = hum + speech + white_noise

    else:
        # Acoustic anomaly: sudden high-frequency transient or screaming/shouting burst
        hum = 0.05 * np.sin(2 * np.pi * 50.0 * t)
        transient_center = int(0.5 * len(t))
        burst_len = int(0.3 * sample_rate)
        
        # High impulse noise (e.g. glass break, desk impact, scream)
        transient = np.zeros(len(t))
        burst = 0.8 * np.sin(2 * np.pi * 2800.0 * t[:burst_len]) + 0.6 * np.random.normal(0, 0.2, burst_len)
        transient[transient_center:transient_center+burst_len] = burst
        
        white_noise = np.random.normal(0, 0.02, len(t))
        audio = hum + transient + white_noise

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    return audio.astype(np.float32)
