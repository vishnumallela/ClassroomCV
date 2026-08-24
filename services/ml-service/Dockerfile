# ml-service on a RunPod CUDA pod (L4 24 GB is the sizing target).
#
# The base image must match the LOCKFILE, not merely "some PyTorch": uv.lock
# pins torch 2.12.1 with the CUDA 13 wheel stack (cuda-toolkit 13.0.2,
# nvidia-*-cu13, triton 3.7.1) and pyproject requires Python >=3.12. The
# earlier 2.7.1-cuda12.6 base shipped Python 3.11 + torch 2.7.1, which made
# `uv pip install` unsatisfiable (numpy 2.5.1 requires >=3.12) and, once past
# that, would have re-downloaded ~2.9 GB of torch + CUDA 13 wheels on top of a
# CUDA 12.6 image. Matching the base to the lock is what makes
# --no-emit-package torch honest: torch/torchvision/triton/nvidia-* are then
# already satisfied and only a short tail of packages is installed.
FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime

# ffmpeg feeds cv2's decoder; libgl/libglib are opencv runtime deps.
# libglib2.0-0 was renamed libglib2.0-0t64 in Ubuntu 24.04's time_t transition
# (the base is 24.04) — the old name is not installable and fails the build.
# openssh-server is here for one reason: the checkpoint is not baked into the
# image, so the volume has to be populated from outside, and a service image
# with no shell cannot be. rsync is what populates it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0t64 openssh-server rsync \
    && rm -rf /var/lib/apt/lists/*

# Pinned, not :latest — a uv release that changes `uv export` semantics would
# otherwise break this build with no commit to point at.
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv

WORKDIR /srv/ml-service
COPY pyproject.toml uv.lock ./
# Torch comes from the base image; install the rest of the lockfile against it.
# --break-system-packages: this base image's interpreter is Ubuntu's system
# python, which is PEP-668 externally-managed. In a container the image IS the
# environment, so installing into it is correct; without the flag uv refuses.
RUN uv export --frozen --no-hashes --no-emit-package torch --no-emit-package torchvision \
      -o /tmp/requirements.txt \
    && uv pip install --system --break-system-packages -r /tmp/requirements.txt \
    && rm -rf /root/.cache

COPY app ./app
# run_one.py drives one video with write_db=False, which is how a lesson gets
# measured on the pod without Postgres, MinIO or a tunnel back to the laptop.
COPY run_one.py ./
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Fail the BUILD, not the first request, if the wiring is wrong: import the app
# and the heavy third-party stack that only loads lazily at runtime.
RUN python -c "import cv2, torch, rfdetr; from app.main import app; print('import OK')"

# The fine-tuned checkpoint is NOT baked into the image — it is a 260 MB
# artifact that changes on its own retraining cadence. Mount the pod volume at
# /workspace (NOT at the workdir — that would shadow the app code) and put the
# checkpoint on it, so it survives pod stop/start and can be swapped without a
# rebuild.
# TensorRT is NOT installed here: it is a heavyweight CUDA-only stack for an
# optimisation that must be measured on the target GPU first (tools/trt_parity.py).
# To enable it on a pod: `uv sync --extra tensorrt` and set RFDETR_TENSORRT=true.
ENV DEVICE=cuda \
    REQUIRE_DEVICE=cuda \
    RFDETR_WEIGHTS=/workspace/weights/rfdetr-medium.pth \
    RFDETR_BATCH=16 \
    RFDETR_TENSORRT=false

EXPOSE 8000 22
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
