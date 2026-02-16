from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .config import AudioConfig, LabelConfig
from .features import extract_features


META_REQUIRED_COLS = [
    "PID",
    "COND",
    "UTTNUM",
    "UTTID",
    "REVISED PAIN",
    "ACTION LABEL",
]


@dataclass(frozen=True)
class ExampleId:
    pid: str
    cond: str
    uttnum: int
    uttid: int

    def filename(self) -> str:
        return f"{self.pid}.{self.cond}.{self.uttnum}.{self.uttid}.wav"


def load_meta(meta_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    missing = [c for c in META_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"meta csv missing columns: {missing}")

    # normalize numeric columns (some annotation csvs store floats)
    df = df.copy()
    # Some rows are intentionally unlabeled (README mentions 5 excluded utterances).
    # Coerce to numeric then drop rows where labels/quality are missing.
    for col in ["UTTNUM", "UTTID", "REVISED PAIN", "ACTION LABEL"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["UTTNUM", "UTTID", "REVISED PAIN", "ACTION LABEL"]).reset_index(drop=True)

    df["UTTNUM"] = df["UTTNUM"].astype(int)
    df["UTTID"] = df["UTTID"].astype(int)
    df["REVISED PAIN"] = df["REVISED PAIN"].astype(float)
    df["ACTION LABEL"] = df["ACTION LABEL"].astype(int)
    return df


def resolve_wav_path(audio_root: Path, row: pd.Series) -> Path:
    ex = ExampleId(
        pid=str(row["PID"]),
        cond=str(row["COND"]),
        uttnum=int(row["UTTNUM"]),
        uttid=int(row["UTTID"]),
    )
    return audio_root / ex.pid / ex.filename()


def read_audio_any(path_or_bytes: bytes | str | Path, target_sr: int) -> tuple[np.ndarray, int]:
    if isinstance(path_or_bytes, (str, Path)):
        audio, sr = sf.read(str(path_or_bytes), dtype="float32", always_2d=False)
    else:
        bio = io.BytesIO(path_or_bytes)
        audio, sr = sf.read(bio, dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio.astype(np.float32), sr


class TamePainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        audio_root: Path,
        audio_cfg: AudioConfig,
        label_cfg: LabelConfig,
        max_action_label: int = 2,
        task: str = "regression",
        return_audio: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_root = audio_root
        self.audio_cfg = audio_cfg
        self.label_cfg = label_cfg
        self.max_action_label = max_action_label
        if task not in {"regression", "classification"}:
            raise ValueError("task must be regression or classification")
        self.task = task
        self.return_audio = bool(return_audio)

        # filter low-quality audio by default
        self.df = self.df[self.df["ACTION LABEL"] <= max_action_label].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = resolve_wav_path(self.audio_root, row)
        if not wav_path.exists():
            raise FileNotFoundError(f"missing wav: {wav_path}")

        audio, _sr = read_audio_any(wav_path, self.audio_cfg.sample_rate)
        feats = extract_features(audio, self.audio_cfg)

        pain = float(row["REVISED PAIN"])
        if self.task == "regression":
            y = np.float32(pain)
        else:
            y = np.int64(self.label_cfg.pain_to_class(pain))

        out = {
            "x": torch.from_numpy(feats),  # (C, T)
            "y": torch.tensor(y),
            "pid": str(row["PID"]),
            "meta": {
                "cond": str(row["COND"]),
                "uttnum": int(row["UTTNUM"]),
                "uttid": int(row["UTTID"]),
            },
        }

        if self.return_audio:
            out["audio"] = audio
            out["sr"] = int(_sr)

        return out


def pid_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(df["PID"].unique()))
    rng.shuffle(pids)
    n_val = max(1, int(len(pids) * val_frac))
    val_pids = set(pids[:n_val].tolist())
    train_df = df[~df["PID"].isin(val_pids)].copy()
    val_df = df[df["PID"].isin(val_pids)].copy()
    return train_df, val_df
