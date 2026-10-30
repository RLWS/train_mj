from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def build_inference_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return a checkpoint with training-only optimizer state removed."""
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint is missing model_state_dict")
    return {
        key: value
        for key, value in checkpoint.items()
        if key != "optimizer_state_dict"
    }


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.inference{input_path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove optimizer state from a PyTorch checkpoint for inference-only use."
    )
    parser.add_argument("checkpoint", type=Path, help="Input PyTorch checkpoint.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output checkpoint path (default: <input>.inference.pt).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    args = parser.parse_args()

    input_path = args.checkpoint.resolve()
    output_path = (args.output or default_output_path(input_path)).resolve()
    if not input_path.is_file():
        parser.error(f"checkpoint does not exist: {input_path}")
    if input_path == output_path:
        parser.error("output path must differ from the input checkpoint")
    if output_path.exists() and not args.force:
        parser.error(f"output already exists; use --force to overwrite: {output_path}")

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint root must be a dictionary")
    inference_checkpoint = build_inference_checkpoint(checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inference_checkpoint, output_path)

    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    print(f"input:  {input_path} ({input_size:.1f} MiB)")
    print(f"output: {output_path} ({output_size:.1f} MiB)")
    print("removed: optimizer_state_dict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
