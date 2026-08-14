import numpy as np
from scipy.signal import butter, filtfilt, get_window

def apply_bandpass_filter(
    audio: np.ndarray,
    sample_rate: int = 16000,
    low_cut: float = 80.0,
    high_cut: float = 7500.0,
    order: int = 4
) -> tuple[np.ndarray, dict]:
    nyquist = 0.5 * sample_rate
    low = max(low_cut / nyquist, 0.001)
    high = min(high_cut / nyquist, 0.999)

    b, a = butter(order, [low, high], btype="band")
    # Zero-phase Butterworth filtering
    filtered_audio = filtfilt(b, a, audio).astype(np.float32)

    info = {
        "applied": True,
        "low_cut": low_cut,
        "high_cut": high_cut,
        "order": order
    }

    return filtered_audio, info

def compute_stft(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024
) -> dict:
    window = get_window("hann", win_length)
    num_frames = 1 + (len(audio) - win_length) // hop_length

    stft_matrix = []
    for i in range(num_frames):
        start = i * hop_length
        frame = audio[start : start + win_length] * window
        fft_frame = np.fft.rfft(frame, n=n_fft)
        stft_matrix.append(fft_frame)

    stft_matrix = np.array(stft_matrix).T # (freq_bins, time_frames)
    magnitude = np.abs(stft_matrix)
    power = magnitude ** 2
    power_db = 10 * np.log10(power + 1e-10)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    times = np.arange(num_frames) * (hop_length / sample_rate)

    return {
        "stft": stft_matrix,
        "magnitude": magnitude,
        "power": power,
        "power_db": power_db,
        "freqs": freqs,
        "times": times
    }
