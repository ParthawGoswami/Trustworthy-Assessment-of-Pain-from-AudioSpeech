from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio

from .model import Wav2Vec2Head


@dataclass
class W2V2Info:
    bundle_name: str
    sample_rate: int
    encoder_dim: int


_BUNDLES: dict[str, torchaudio.pipelines.Wav2Vec2ASRBundle] = {
    # Good default: compact but strong.
    "WAV2VEC2_BASE": torchaudio.pipelines.WAV2VEC2_BASE,
    # If you want a bigger model later:
    "WAV2VEC2_LARGE": torchaudio.pipelines.WAV2VEC2_LARGE,
}


def get_bundle(bundle_name: str = "WAV2VEC2_BASE"):
    if bundle_name not in _BUNDLES:
        raise ValueError(f"Unknown wav2vec2 bundle '{bundle_name}'. Options: {sorted(_BUNDLES)}")
    return _BUNDLES[bundle_name]


def get_w2v2_info(bundle_name: str = "WAV2VEC2_BASE") -> W2V2Info:
    bundle = get_bundle(bundle_name)
    sample_rate = int(bundle.sample_rate)

    # Infer encoder dim from the model's encoder projection.
    # This is more robust across torchaudio versions.
    model = bundle.get_model()
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, sample_rate)  # 1 second
        feats, _ = model.extract_features(dummy)
        enc_dim = int(feats[-1].shape[-1])

    return W2V2Info(bundle_name=bundle_name, sample_rate=sample_rate, encoder_dim=enc_dim)


def ensure_sr(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    """Resample a mono waveform to target_sr.

    Args:
        wav: (T,) float tensor in [-1,1]
    """
    if sr == target_sr:
        return wav
    wav = wav.unsqueeze(0)
    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav.squeeze(0)


def build_wav2vec2_model(
    task: str,
    bundle_name: str = "WAV2VEC2_BASE",
    n_classes: int = 3,
    head_hidden: int = 256,
    freeze_encoder: bool = True,
) -> tuple[Wav2Vec2Head, W2V2Info]:
    info = get_w2v2_info(bundle_name)
    bundle = get_bundle(bundle_name)
    encoder = bundle.get_model()

    model = Wav2Vec2Head(
        encoder=encoder,
        encoder_dim=info.encoder_dim,
        task=task,
        n_classes=n_classes,
        head_hidden=head_hidden,
        freeze_encoder=freeze_encoder,
    )

    return model, info
