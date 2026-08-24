#!/usr/bin/env bash
# Provision a RunPod GPU pod to run the RF-DETR pipeline on one long video.
#
#   ./tools/pod_setup.sh <ssh-host> <ssh-port> <video.mp4> [checkpoint.pth]
#
# Use RunPod's "SSH over exposed TCP" line (ssh root@<ip> -p <port>), NOT the
# ssh.runpod.io proxy — the proxy does not support SCP/SFTP and this script
# rsyncs several hundred MB.
#
# Every step below that looks paranoid is a trap this project has already paid
# for. They are listed here so the next person does not rediscover them:
#
#  1. VENV ON LOCAL DISK, NOT /workspace. /workspace is a network mount and
#     uv's cache is on local disk, so uv cannot hardlink and full-copies ~6 GB
#     of PyTorch across the network at ~90 MB/min. UV_PROJECT_ENVIRONMENT fixes
#     it. Code stays on /workspace — it is tiny.
#  2. LONG STEPS IN tmux. A dropped SSH connection killed a uv sync mid-copy and
#     left a venv that looked complete but was missing a torch shared library;
#     the service then started on CPU and would have billed hours looking fine.
#  3. VERIFY CUDA WITH A REAL MATMUL. torch.cuda.is_available() returns True
#     with wrong-architecture kernels.
#  4. DRIVER >= r580. uv.lock pins torch 2.12.1, which resolves to a cu13 wheel.
#     An older driver gives cuda:False. Filter RunPod by CUDA 13.0.
#  5. NEVER COPY .env TO THE POD. Keys stay on the Mac; the pod gets exactly the
#     env vars it needs, passed explicitly.
set -euo pipefail

HOST="${1:?usage: pod_setup.sh <ssh-host> <ssh-port> <video.mp4> [checkpoint.pth]}"
PORT="${2:?missing ssh port}"
VIDEO="${3:?missing video path}"
CKPT="${4:-/Users/vangaladineshreddy/Desktop/classroomcv-models/rfdetr-medium-mt2vr2m9__checkpoint_best_total.pth}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE=/workspace/ml-service
# Reuse ONE ssh connection: RunPod's sshd hits MaxStartups and starts returning
# exit 255 if you open a connection per command.
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$HOME/.ssh/cm-pod-$PORT" -o ControlPersist=20m
          -o StrictHostKeyChecking=accept-new -p "$PORT")
sshp() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }

echo "==> 1/6 system packages (opencv needs libgl1 + libglib on Ubuntu 24.04)"
sshp 'apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0t64 ffmpeg rsync tmux >/dev/null 2>&1 || \
      apt-get install -y -qq libgl1 libglib2.0-0 ffmpeg rsync tmux >/dev/null 2>&1; \
      command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; \
      mkdir -p '"$REMOTE"'/data'

echo "==> 2/6 code (excluding .env, venv, weights, fixtures)"
rsync -az --stats -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.venv' --exclude '.env' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude '*.pth' --exclude '*.trt' --exclude 'eval/gt' --exclude 'data' \
  "$HERE/" "$HOST:$REMOTE/" | tail -3

echo "==> 3/6 checkpoint ($(du -h "$CKPT" | cut -f1)) — resumable"
rsync -az --partial --progress -e "ssh ${SSH_OPTS[*]}" "$CKPT" "$HOST:$REMOTE/rfdetr-medium.pth"

echo "==> 4/6 video ($(du -h "$VIDEO" | cut -f1)) — resumable"
rsync -az --partial --progress -e "ssh ${SSH_OPTS[*]}" "$VIDEO" "$HOST:$REMOTE/data/lesson.mp4"

echo "==> 5/6 dependencies (venv on LOCAL disk, in tmux so a dropped ssh cannot kill it)"
sshp 'export PATH=$HOME/.local/bin:$PATH; cd '"$REMOTE"' && \
      tmux new-session -d -s sync "UV_PROJECT_ENVIRONMENT=/root/ml-venv uv sync 2>&1 | tee /workspace/sync.log" ; \
      while tmux has-session -t sync 2>/dev/null; do sleep 10; printf .; done; echo; tail -3 /workspace/sync.log'

echo "==> 6/6 verifying CUDA with a real matmul"
sshp 'cd '"$REMOTE"' && /root/ml-venv/bin/python -c "
import torch
print(\"torch\", torch.__version__, \"| cuda available:\", torch.cuda.is_available())
assert torch.cuda.is_available(), \"CUDA unavailable — driver is probably older than r580 for this cu13 torch build\"
print(\"gpu:\", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device=\"cuda\")
assert float((x @ x).sum()) != 0.0, \"CUDA matmul returned zeros — wrong-arch kernels\"
print(\"matmul OK\")
"'

cat <<EOF

==> ready. Start the run (detached, survives disconnect):

ssh ${SSH_OPTS[*]} $HOST \\
  'cd $REMOTE && tmux new-session -d -s run "RFDETR_WEIGHTS=$REMOTE/rfdetr-medium.pth \\
     DEVICE=cuda REQUIRE_DEVICE=cuda RFDETR_BATCH=16 \\
     /root/ml-venv/bin/python run_one.py $REMOTE/data/lesson.mp4 2>&1 | tee /workspace/run.log"'

==> then poll:

ssh ${SSH_OPTS[*]} $HOST 'tail -20 /workspace/run.log'

EOF
