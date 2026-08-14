import os
import numpy as np
import soundfile as sf
from src.audio import load_and_preprocess_audio, create_synthetic_audio

def test_synthetic_audio_generation():
    audio_norm = create_synthetic_audio("normal", sample_rate=16000, duration_seconds=5.0)
    audio_abnorm = create_synthetic_audio("abnormal", sample_rate=16000, duration_seconds=5.0)

    assert len(audio_norm) == 80000
    assert len(audio_abnorm) == 80000
    assert audio_norm.dtype == np.float32
    assert audio_abnorm.dtype == np.float32

def test_audio_load_and_pad(tmp_path="temp_test_audio.wav"):
    audio_data = np.random.normal(0, 0.1, 8000).astype(np.float32) # 0.5s at 16k
    sf.write(tmp_path, audio_data, 16000)

    loaded, meta = load_and_preprocess_audio(tmp_path, target_sample_rate=16000, duration_seconds=5.0)

    assert len(loaded) == 80000 # Exactly 5 seconds padded
    assert meta["valid"] == True

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

if __name__ == "__main__":
    test_synthetic_audio_generation()
    test_audio_load_and_pad()
    print("Audio unit tests passed!")
