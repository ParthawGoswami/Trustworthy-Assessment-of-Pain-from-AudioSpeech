from __future__ import annotations

"""Fine-tune a wav2vec2 model (torchaudio pipeline) for pain prediction.

This is an alternative to the log-mel CNN baseline and typically yields better accuracy.

Outputs:
- `<outdir>/checkpoint.pt`
- `<outdir>/model_bundle.pt`
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from silent_patient.bundle import ModelBundle, save_bundle
from silent_patient.config import AudioConfig, LabelConfig
from silent_patient.data import TamePainDataset, load_meta, pid_split
from silent_patient.train_utils import eval_classification, eval_regression
from silent_patient.wav2vec2 import build_wav2vec2_model, ensure_sr


def _collate_wav2vec2(batch: list[dict], target_sr: int) -> dict:
    wavs = []
    ys = []
    for ex in batch:
        # TamePainDataset returns float32 numpy waveform at audio_cfg.sample_rate.
        wav = torch.from_numpy(ex["audio"]).float()
        sr = int(ex["sr"])
        wav = ensure_sr(wav, sr, target_sr)
        wavs.append(wav)
        ys.append(ex["y"])

    max_len = max(int(w.shape[0]) for w in wavs)
    x = torch.zeros(len(wavs), max_len, dtype=torch.float32)
    for i, w in enumerate(wavs):
        x[i, : w.shape[0]] = w

    y0 = ys[0]
    # Dataset stores y as a torch scalar tensor; normalize here.
    if isinstance(y0, torch.Tensor):
        y0 = y0.item()
        ys = [y.item() if isinstance(y, torch.Tensor) else y for y in ys]

    if isinstance(y0, (float, np.floating)):
        y = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
    else:
        y = torch.tensor(ys, dtype=torch.long)
    return {"wav": x, "y": y}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-csv", type=Path, required=True)
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--task", choices=["regression", "classification"], default="regression")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-action-label", type=int, default=2)
    ap.add_argument("--bundle", type=str, default="WAV2VEC2_BASE")
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--unfreeze", action="store_true", help="Fine-tune encoder too (slower, better)")
    args = ap.parse_args()

    # We still use AudioConfig for dataset loading, but wav2vec2 will override SR internally.
    audio_cfg = AudioConfig(sample_rate=16000)
    label_cfg = LabelConfig()

    df = load_meta(args.meta_csv)
    train_df, val_df = pid_split(df, val_frac=0.2, seed=7)

    train_ds = TamePainDataset(
        train_df,
        audio_root=args.audio_root,
        audio_cfg=audio_cfg,
        label_cfg=label_cfg,
        max_action_label=args.max_action_label,
        task=args.task,
        return_audio=True,
    )
    val_ds = TamePainDataset(
        val_df,
        audio_root=args.audio_root,
        audio_cfg=audio_cfg,
        label_cfg=label_cfg,
        max_action_label=args.max_action_label,
        task=args.task,
        return_audio=True,
    )

    model, info = build_wav2vec2_model(
        task=args.task,
        bundle_name=args.bundle,
        head_hidden=args.head_hidden,
        freeze_encoder=(not args.unfreeze),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: _collate_wav2vec2(b, info.sample_rate),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: _collate_wav2vec2(b, info.sample_rate),
    )

    # Loss
    if args.task == "regression":
        loss_fn = torch.nn.SmoothL1Loss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_score = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            wav = batch["wav"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(wav)
            if args.task == "classification":
                loss = loss_fn(out, y)
            else:
                loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        tr_loss = float(np.mean(losses)) if losses else float("nan")

        # metrics
        if args.task == "regression":
            vm = eval_regression(model, val_loader, device=device, input_key="wav")
            score = -float(vm.mae)
            print(
                f"epoch {epoch:02d} train_loss={tr_loss:.4f} val_loss={vm.loss:.4f} val_mae={vm.mae:.3f}"
            )
        else:
            vm = eval_classification(model, val_loader, device=device, input_key="wav")
            score = float(vm.acc)
            print(
                f"epoch {epoch:02d} train_loss={tr_loss:.4f} val_loss={vm.loss:.4f} val_acc={vm.acc:.3f}"
            )

        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    args.outdir.mkdir(parents=True, exist_ok=True)
    val_metric = float(-best_score if args.task == "regression" else best_score)

    torch.save(
        {
            "task": args.task,
            "arch": "wav2vec2",
            "w2v2_bundle": info.bundle_name,
            "head_hidden": int(args.head_hidden),
            "state_dict": best_state,
            "audio_cfg": audio_cfg.__dict__,
            "label_cfg": label_cfg.__dict__,
            "val_metric": val_metric,
        },
        args.outdir / "checkpoint.pt",
    )

    bundle = ModelBundle(
        task=args.task,
        arch="wav2vec2",
        w2v2_bundle=info.bundle_name,
        head_hidden=int(args.head_hidden),
        audio_cfg=audio_cfg,
        label_cfg=label_cfg,
        state_dict=best_state,
        in_channels=0,
        val_metric=val_metric,
    )
    save_bundle(bundle, args.outdir / "model_bundle.pt")


if __name__ == "__main__":
    main()
