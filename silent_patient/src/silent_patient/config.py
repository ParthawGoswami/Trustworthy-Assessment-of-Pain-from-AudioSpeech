from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    clip_seconds: float = 3.5  # dataset utterances ~2-4s; pad/trim for batching
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    n_mfcc: int = 13


@dataclass(frozen=True)
class LabelConfig:
    pain_min: int = 1
    pain_max: int = 10

    @staticmethod
    def pain_to_class(pain: float) -> int:
        # mild=0 (1-3), moderate=1 (4-6), severe=2 (7-10)
        if pain <= 3:
            return 0
        if pain <= 6:
            return 1
        return 2
