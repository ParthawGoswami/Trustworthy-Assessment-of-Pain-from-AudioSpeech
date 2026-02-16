from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from silent_patient.bundle import ModelBundle, save_bundle
from silent_patient.config import AudioConfig, LabelConfig
from silent_patient.data import TamePainDataset, load_meta, pid_split
from silent_patient.model import SmallCnnClassifier, SmallCnnRegressor
from silent_patient.train_utils import eval_classification, eval_regression, fit_one_epoch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-csv", type=Path, required=True)
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--task", choices=["regression", "classification"], default="regression")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-action-label", type=int, default=2)
    args = ap.parse_args()

    audio_cfg = AudioConfig()
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
    )
    val_ds = TamePainDataset(
        val_df,
        audio_root=args.audio_root,
        audio_cfg=audio_cfg,
        label_cfg=label_cfg,
        max_action_label=args.max_action_label,
        task=args.task,
    )

    # infer channels
    sample = train_ds[0]["x"]
    in_channels = int(sample.shape[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.task == "regression":
        model = SmallCnnRegressor(in_channels=in_channels)
    else:
        model = SmallCnnClassifier(in_channels=in_channels, n_classes=3)

    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss = fit_one_epoch(model, train_loader, opt, device=device, task=args.task)
        if args.task == "regression":
            vm = eval_regression(model, val_loader, device=device)
            score = -vm.mae  # maximize
            print(f"epoch {epoch:02d} train_loss={tr_loss:.4f} val_loss={vm.loss:.4f} val_mae={vm.mae:.3f}")
        else:
            vm = eval_classification(model, val_loader, device=device)
            score = vm.acc
            print(f"epoch {epoch:02d} train_loss={tr_loss:.4f} val_loss={vm.loss:.4f} val_acc={vm.acc:.3f}")

        if best_val is None or score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Save raw state
    torch.save(
        {
            "task": args.task,
            "in_channels": in_channels,
            "state_dict": best_state,
            "audio_cfg": audio_cfg.__dict__,
            "label_cfg": label_cfg.__dict__,
            "val_metric": float(-best_val if args.task == "regression" else best_val),
        },
        args.outdir / "checkpoint.pt",
    )

    # Also save bundle for demo
    bundle = ModelBundle(
        task=args.task,
        audio_cfg=audio_cfg,
        label_cfg=label_cfg,
        state_dict=best_state,
        in_channels=in_channels,
        val_metric=float(-best_val if args.task == "regression" else best_val),
    )
    save_bundle(bundle, args.outdir / "model_bundle.pt")


if __name__ == "__main__":
    main()
