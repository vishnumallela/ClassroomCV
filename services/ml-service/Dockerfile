# ml-service on a RunPod CUDA pod (L4 24 GB is the sizing target).
#
# The official PyTorch CUDA runtime image ships torch + CUDA + cuDNN matched;
# uv then installs only the remaining deps against the system interpreter.
# TensorRT engines are NOT built here — they are per-GPU-model artifacts, so
# the service exports one on the pod at first load (TENSORRT_EXPORT=true) and
# caches it on the volume.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# ffmpeg feeds cv2's decoder; libgl/libglib are opencv runtime deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /srv/ml-service
COPY pyproject.toml uv.lock ./
# Torch comes from the base image; install the rest of the lockfile against it.
RUN uv export --frozen --no-hashes --no-emit-package torch --no-emit-package torchvision \
      -o /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt

COPY app ./app
COPY scripts ./scripts

# Weights: MODEL_NAME=auto resolves to yolo26x-pose.pt on cuda; ultralytics
# downloads it on first use into this dir. Mount a volume here so the weight
# AND the exported .engine survive pod restarts.
ENV DEVICE=cuda \
    REQUIRE_DEVICE=cuda \
    TENSORRT_EXPORT=true \
    IMGSZ=1536 \
    MODEL_NAME=auto

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
