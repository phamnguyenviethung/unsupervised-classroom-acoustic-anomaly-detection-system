import numpy as np
from src.features import extract_bandpower_features, extract_time_domain_bandpower, extract_baseline_representation

def test_bandpower_extraction():
    power_spec = np.random.uniform(1e-5, 1.0, (513, 313)).astype(np.float32)
    freqs = np.linspace(0, 8000, 513)
    times = np.linspace(0, 5.0, 313)

    res = extract_bandpower_features(power_spec, freqs, times)
    feat_vec = res["feature_vector"]

    assert len(feat_vec) == 30
    assert not np.isnan(feat_vec).any()

def test_baseline_representation():
    audio = np.random.normal(0, 0.1, 80000).astype(np.float32)
    feat = extract_baseline_representation(audio, target_dim=64)
    assert len(feat) == 67
    assert not np.isnan(feat).any()

if __name__ == "__main__":
    test_bandpower_extraction()
    test_baseline_representation()
    print("Feature unit tests passed!")
