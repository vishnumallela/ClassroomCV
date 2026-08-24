"""Serve RF-DETR as a TensorRT engine on the GPU pod.

RF-DETR ships TensorRT EXPORT but not TensorRT SERVING: `rfdetr.export._tensorrt`
builds an engine and then points you at a separate library for inference. The
only in-package runtime is `TRTInference`, which lives in a benchmark module and
hands back raw `pred_logits` / `pred_boxes`. So this module is the missing half —
and it is written to borrow rather than reimplement, because a post-processing
mismatch would not raise, it would silently shift every box and therefore every
KPI:

- preprocessing reads `means`/`stds` OFF THE LOADED MODEL rather than hardcoding
  ImageNet constants, so it cannot drift from what `predict()` does;
- decoding calls rfdetr's own `PostProcess`, the same class the PyTorch path
  uses, rather than a local top-k + cxcywh->xyxy.

What is left that this module owns — the resize, the /255, and the batch
plumbing — is exactly what `tools/trt_parity.py` checks against the PyTorch
path before you trust it.

THE ENGINE IS NOT PORTABLE. TensorRT compiles for the specific GPU model and
TensorRT version present at build time, which is why it is built ON THE POD at
first load and cached on the volume, never baked into the image. A pod that
changes GPU type must delete the `.trt` file and let it rebuild.

Enabled with RFDETR_TENSORRT=true and `uv sync --extra tensorrt`. Off by
default, and a failure anywhere here degrades to the fp16 PyTorch path rather
than failing the analysis: TensorRT is a throughput optimisation, and losing it
should cost speed, not results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# TensorRT specialises the graph for a batch size. The detector's last batch of
# a video is usually short, so the engine is built with a dynamic batch
# dimension and the runtime is told the real size per call.
_MAX_BATCH = 32


class _EngineDetections:
    """The three arrays app.detector._to_detections reads, per frame.

    Deliberately duck-typed rather than a supervision.Detections: the detector
    only ever touches class_id / confidence / xyxy / len, so matching that
    surface keeps the TensorRT path a drop-in for the PyTorch one.
    """

    __slots__ = ("class_id", "confidence", "xyxy")

    def __init__(self, class_id: np.ndarray, confidence: np.ndarray, xyxy: np.ndarray) -> None:
        self.class_id = class_id
        self.confidence = confidence
        self.xyxy = xyxy

    def __len__(self) -> int:
        return len(self.class_id)


def engine_path_for(weights: str, resolution: int) -> Path:
    """Where this weight's engine is cached — next to the weight, on the volume.

    The resolution is in the filename because an engine is compiled for one
    input shape; changing RFDETR_RESOLUTION must build a new engine rather than
    silently serve a mismatched one.
    """
    p = Path(weights)
    return p.with_name(f"{p.stem}.r{resolution}.trt")


def export_engine(model, weights: str, resolution: int, batch: int) -> Optional[Path]:
    """Build the TensorRT engine for this checkpoint. Minutes, once, on the pod.

    Returns the engine path, or None when export is unavailable or failed —
    the caller then serves PyTorch.
    """
    out = engine_path_for(weights, resolution)
    if out.is_file():
        logger.info("TensorRT engine already built: %s", out)
        return out
    try:
        logger.info(
            "TensorRT export starting for %s (one-time on this GPU, several minutes)",
            weights,
        )
        produced = model.export(
            output_dir=str(out.parent),
            format="tensorrt",
            fp16=True,
            batch_size=batch,
            dynamic_batch=True,
            shape=(resolution, resolution),
            verbose=False,
        )
        produced = Path(produced)
        if produced != out:
            produced.replace(out)
        logger.info("TensorRT export complete: %s", out)
        return out
    except Exception:
        logger.warning(
            "TensorRT export failed; serving the fp16 PyTorch model instead",
            exc_info=True,
        )
        return None


class TensorRTBackend:
    """A loaded engine plus the preprocessing and decoding around it."""

    def __init__(self, engine: Path, model, device: str, resolution: int) -> None:
        from rfdetr.export.benchmark import TRTInference

        self._trt = TRTInference(
            engine_path=str(engine), device=device, max_batch_size=_MAX_BATCH
        )
        self._device = device
        self._resolution = resolution
        # Borrowed from the loaded model so preprocessing cannot drift from
        # what RFDETR.predict does.
        self._means = model.means
        self._stds = model.stds
        self._post = _build_postprocessor(model)
        logger.info("serving RF-DETR from TensorRT engine %s", engine)

    def predict(self, frames: list[np.ndarray], threshold: float) -> list[_EngineDetections]:
        """RGB frames -> one _EngineDetections per frame, in source pixels."""
        import torch
        import torchvision.transforms.functional as F

        sizes = [(f.shape[0], f.shape[1]) for f in frames]
        batch = torch.stack(
            [
                torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float() / 255.0
                for f in frames
            ]
        ).to(self._device)
        batch = F.resize(batch, [self._resolution, self._resolution])
        batch = F.normalize(batch, self._means, self._stds)
        if batch.dtype != torch.float16:
            batch = batch.half()

        raw = self._trt.run_sync({"input": batch})
        outputs = _as_output_dict(raw)
        target_sizes = torch.tensor(sizes, device=self._device)
        results = self._post(outputs, target_sizes=target_sizes)

        out: list[_EngineDetections] = []
        for r in results:
            scores = r["scores"].detach().float().cpu().numpy()
            keep = scores >= threshold
            out.append(
                _EngineDetections(
                    class_id=r["labels"].detach().cpu().numpy()[keep],
                    confidence=scores[keep],
                    xyxy=r["boxes"].detach().float().cpu().numpy()[keep],
                )
            )
        return out


def _build_postprocessor(model):
    """rfdetr's own PostProcess, so decoding is never reimplemented here."""
    from rfdetr.inference import PostProcess

    ctx = getattr(model, "model", None)
    existing = getattr(ctx, "postprocessors", None)
    if isinstance(existing, dict) and "bbox" in existing:
        return existing["bbox"]
    return PostProcess()


def _as_output_dict(raw) -> dict:
    """Normalise the engine's outputs to the {pred_logits, pred_boxes} contract.

    TRTInference returns whatever the engine's output bindings are named, and
    the names come from the ONNX export. Accept the common shapes rather than
    assuming one, and fail loudly on an unrecognised set — a silently mismatched
    binding would decode noise into confident boxes.
    """
    if isinstance(raw, dict):
        if "pred_logits" in raw and "pred_boxes" in raw:
            return {"pred_logits": raw["pred_logits"], "pred_boxes": raw["pred_boxes"]}
        values = list(raw.values())
    else:
        values = list(raw)
    if len(values) == 2:
        a, b = values
        # Boxes are the 4-wide tensor; logits carry one channel per class.
        if a.shape[-1] == 4:
            return {"pred_boxes": a, "pred_logits": b}
        if b.shape[-1] == 4:
            return {"pred_logits": a, "pred_boxes": b}
    raise RuntimeError(
        f"unrecognised TensorRT engine outputs: {getattr(raw, 'keys', lambda: raw)()}; "
        "refusing to guess which binding is boxes"
    )


def try_load(model, weights: str, device: str, resolution: int, batch: int):
    """Export (once) and load the engine, or return None to stay on PyTorch."""
    if device.split(":", 1)[0] != "cuda":
        logger.info("TensorRT requested but device is %s; staying on PyTorch", device)
        return None
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        logger.warning(
            "RFDETR_TENSORRT=true but the tensorrt extra is not installed "
            "(`uv sync --extra tensorrt`); serving the fp16 PyTorch model"
        )
        return None
    engine = export_engine(model, weights, resolution, batch)
    if engine is None:
        return None
    try:
        return TensorRTBackend(engine, model, device, resolution)
    except Exception:
        logger.warning(
            "TensorRT engine failed to load; serving the fp16 PyTorch model",
            exc_info=True,
        )
        return None
