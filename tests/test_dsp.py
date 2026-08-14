import numpy as np
from src.dsp import apply_bandpass_filter, compute_stft

def test_bandpass_filter():
    audio = np.random.normal(0, 0.1, 80000).astype(np.float32)
    filtered, info = apply_bandpass_filter(audio, sample_rate=16000, low_cut=80.0, high_cut=7500.0, order=4)

    assert len(filtered) == 80000
    assert info["applied"] == True
    assert not np.isnan(filtered).any()

def test_stft_computation():
    audio = np.random.normal(0, 0.1, 80000).astype(np.float32)
    res = compute_stft(audio, sample_rate=16000, n_fft=1024, hop_length=256, win_length=1024)

    assert "magnitude" in res
    assert "power" in res
    assert "power_db" in res
    assert res["magnitude"].shape[0] == 513
    assert not np.isnan(res["magnitude"]).any()

if __name__ == "__main__":
    test_bandpass_filter()
    test_stft_computation()
    print("DSP unit tests passed!")
