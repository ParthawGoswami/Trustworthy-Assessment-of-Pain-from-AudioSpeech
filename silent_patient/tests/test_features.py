import numpy as np

from silent_patient.config import AudioConfig
from silent_patient.features import extract_features, simple_cue_features


def test_extract_features_shape():
    cfg = AudioConfig(sample_rate=16000, clip_seconds=1.0, n_mels=32, n_mfcc=13)
    audio = np.random.randn(int(cfg.sample_rate * cfg.clip_seconds)).astype(np.float32) * 0.01
    x = extract_features(audio, cfg)
    assert x.ndim == 2
    assert x.shape[0] == cfg.n_mels + cfg.n_mfcc
    assert x.shape[1] > 1


def test_cue_features_keys():
    sr = 16000
    audio = np.random.randn(sr).astype(np.float32) * 0.01
    cues = simple_cue_features(audio, sr)
    assert "breathiness" in cues
    assert "pitch_var" in cues
