from __future__ import annotations

import numpy as np

from .config import AudioConfig


def _pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) == target_len:
        return audio
    if len(audio) > target_len:
        return audio[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[: len(audio)] = audio
    return out


def extract_features(audio: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Return (C, T) float32 features for CNN.

    We use a log-mel spectrogram as the main feature, plus MFCCs concatenated as extra channels.
    """
    import librosa

    n_samples = int(cfg.sample_rate * cfg.clip_seconds)
    audio = _pad_or_trim(audio.astype(np.float32), n_samples)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel + 1e-8).astype(np.float32)  # (n_mels, T)

    mfcc = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=cfg.n_mfcc,
    ).astype(np.float32)  # (n_mfcc, T)

    # Normalize per-channel for stability
    def norm(x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=1, keepdims=True)
        sig = x.std(axis=1, keepdims=True) + 1e-6
        return (x - mu) / sig

    x = np.concatenate([norm(log_mel), norm(mfcc)], axis=0)  # (C, T)
    return x.astype(np.float32)


def simple_cue_features(audio: np.ndarray, sr: int) -> dict:
    """Heuristic cues for the UI: breathiness / tremor-ish instability / pitch variance / speech rate proxy."""
    import librosa

    res = {}
    audio = audio.astype(np.float32)
    if len(audio) < sr * 0.25:
        return {"too_short": 1.0}

    # Pitch (F0) via YIN
    f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
    f0 = f0[np.isfinite(f0)]
    if len(f0) > 5:
        res["pitch_var"] = float(np.std(f0) / (np.mean(f0) + 1e-6))
    else:
        res["pitch_var"] = 0.0

    # Jitter-ish proxy: frame-to-frame relative pitch change
    if len(f0) > 5:
        df = np.abs(np.diff(f0)) / (np.abs(f0[:-1]) + 1e-6)
        res["pitch_instability"] = float(np.median(df))
    else:
        res["pitch_instability"] = 0.0

    # Breathiness proxy: high-frequency energy ratio
    S = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    hi = S[freqs >= 3000].mean()
    lo = S[(freqs >= 300) & (freqs < 3000)].mean()
    res["breathiness"] = float(hi / (lo + 1e-6))

    # Speech rate proxy: number of onset events per second
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=256, units="time")
    dur = len(audio) / sr
    res["onsets_per_sec"] = float(len(onsets) / max(dur, 1e-6))

    # Non-speech mode hint: spectral flatness + low onset
    flatness = librosa.feature.spectral_flatness(y=audio).mean()
    res["spectral_flatness"] = float(flatness)
    res["non_speech_likelihood"] = float((flatness > 0.2) and (res["onsets_per_sec"] < 1.0))

    return res
