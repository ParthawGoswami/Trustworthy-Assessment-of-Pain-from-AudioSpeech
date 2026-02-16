from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-file", type=Path, required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.model_dir / "checkpoint.pt", map_location="cpu")
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": ckpt["task"],
            "audio_cfg": ckpt["audio_cfg"],
            "label_cfg": ckpt["label_cfg"],
            "state_dict": ckpt["state_dict"],
            "in_channels": ckpt["in_channels"],
            "val_metric": ckpt.get("val_metric", 0.0),
        },
        args.out_file,
    )
    print(f"wrote {args.out_file}")


if __name__ == "__main__":
    main()
