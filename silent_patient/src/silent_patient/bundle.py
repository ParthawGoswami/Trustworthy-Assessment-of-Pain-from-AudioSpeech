from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .config import AudioConfig, LabelConfig


@dataclass
class ModelBundle:
    task: str
    audio_cfg: AudioConfig
    label_cfg: LabelConfig
    state_dict: dict
    in_channels: int
    # for confidence: store validation MAE (regression) or accuracy (classification)
    val_metric: float

    # Model architecture identifier.
    # - "cnn": legacy baseline using log-mel features
    # - "wav2vec2": fine-tuned torchaudio wav2vec2 encoder + small head
    arch: str = "cnn"
    # For wav2vec2 bundles
    w2v2_bundle: str | None = None
    head_hidden: int | None = None


def save_bundle(bundle: ModelBundle, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": bundle.task,
            "arch": bundle.arch,
            "w2v2_bundle": bundle.w2v2_bundle,
            "head_hidden": bundle.head_hidden,
            "audio_cfg": bundle.audio_cfg.__dict__,
            "label_cfg": bundle.label_cfg.__dict__,
            "state_dict": bundle.state_dict,
            "in_channels": bundle.in_channels,
            "val_metric": bundle.val_metric,
        },
        out_file,
    )


def load_bundle(path: Path) -> ModelBundle:
    obj = torch.load(path, map_location="cpu")
    audio_cfg = AudioConfig(**obj["audio_cfg"])
    label_cfg = LabelConfig(**obj["label_cfg"])
    return ModelBundle(
        task=str(obj["task"]),
    arch=str(obj.get("arch", "cnn")),
    w2v2_bundle=obj.get("w2v2_bundle"),
    head_hidden=obj.get("head_hidden"),
        audio_cfg=audio_cfg,
        label_cfg=label_cfg,
        state_dict=obj["state_dict"],
        in_channels=int(obj["in_channels"]),
        val_metric=float(obj["val_metric"]),
    )
