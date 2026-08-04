"""Export the pose weight to a TensorRT engine on this GPU.

Run ON the serving GPU (engines are specific to the GPU model and TensorRT
version — an engine built on an L4 will not load on a 4090):

    uv run python scripts/export_tensorrt.py                # yolo26x-pose.pt
    uv run python scripts/export_tensorrt.py yolo26l-pose.pt --imgsz 1536

The service also does this automatically at first load when
TENSORRT_EXPORT=true; this script exists so the export can be baked into pod
provisioning (before the first job is billed) and re-run after a GPU or
TensorRT upgrade.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weight", nargs="?", default="yolo26x-pose.pt")
    ap.add_argument(
        "--imgsz",
        type=int,
        default=1536,
        help="max inference size the dynamic engine supports (default 1536)",
    )
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; TensorRT export needs the serving GPU")

    from ultralytics import YOLO

    out = YOLO(args.weight).export(
        format="engine",
        half=True,
        dynamic=True,
        batch=1,
        imgsz=args.imgsz,
        device="cuda",
    )
    print(f"engine written: {out}")
    print("serve it with MODEL_NAME=auto (picked up automatically) or "
          f"MODEL_NAME={out}")


if __name__ == "__main__":
    main()
