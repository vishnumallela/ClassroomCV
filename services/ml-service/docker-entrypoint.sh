#!/usr/bin/env bash
# Start sshd (when RunPod gave us a key) and then hand off to the service.
#
# Why sshd is in a service image at all: the fine-tuned checkpoint is
# deliberately not baked in (255 MB, its own retraining cadence), so SOMETHING
# has to put it on the volume. Without a shell the pod boots, reports healthy,
# and then fails every /analyze with "checkpoint not found" — which is exactly
# what happened the first time this image was deployed. RunPod injects the
# account's registered key as PUBLIC_KEY; honouring it is the convention its own
# templates follow.
#
# sshd is best-effort and never gates the service: no key, no sshd, and uvicorn
# still starts. `exec` on the last line so uvicorn is PID 1 and receives the
# stop signal directly.
set -euo pipefail

if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
  ssh-keygen -A >/dev/null 2>&1 || true
  mkdir -p /run/sshd
  /usr/sbin/sshd -e 2>/dev/null && echo "sshd listening on 22" || echo "sshd failed to start (continuing)"
else
  echo "no PUBLIC_KEY in env; skipping sshd"
fi

exec "$@"
