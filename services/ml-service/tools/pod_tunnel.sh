#!/usr/bin/env bash
# Open the reverse tunnels a RunPod pod needs to reach THIS machine, and keep
# them open.
#
#   ./tools/pod_tunnel.sh [api-base]        # default http://127.0.0.1:8787
#
# WHY THIS EXISTS. The pod runs in a datacenter; MinIO and Postgres run on the
# laptop. Two of the three legs of an analysis run inbound to the laptop and
# cannot work without a tunnel:
#
#   pod:9000 -> MinIO     the pod fetches the video by PRESIGNED URL. The host
#                         in that URL is API_SERVICE__S3_ENDPOINT, so it must
#                         stay reachable under the SAME name inside the pod —
#                         the Host header is what SigV4 signs, and it is also
#                         what MEDIA_URL_ALLOWLIST checks. Rewriting the host
#                         breaks both.
#   pod:5533 -> Postgres  the pipeline COPYs detection_events straight into the
#                         DB (jobs.run_pipeline write_db=True), it does not hand
#                         them back through the API.
#
# The third leg, api-service -> pod, needs no tunnel: it goes out over RunPod's
# proxy hostname.
#
# Without this, an analysis fails with `<urlopen error [Errno 111] Connection
# refused>` — which reads like the GPU is broken but is only a missing route.
#
# The pod endpoint is read from the running api-service rather than pasted in,
# because a new pod gets a new IP and a new SSH port EVERY time, and a stale
# copy-pasted one tunnels to nothing.
set -euo pipefail

API="${1:-http://127.0.0.1:8787}"

read -r IP PORT <<<"$(
  curl -s -m 20 -X POST "$API/rpc/gpu/status" \
    -H 'content-type: application/json' -d '{}' |
    python3 -c '
import json, sys
pod = (json.load(sys.stdin).get("json") or {}).get("pod")
if not pod:
    sys.exit("no pod: create one in Settings first")
ip, pm = pod.get("publicIp"), pod.get("portMappings") or {}
if not ip or "22" not in {str(k) for k in pm}:
    sys.exit(f"pod {pod.get(\"id\")} has no SSH yet (status {pod.get(\"desiredStatus\")}) — wait and retry")
print(ip, pm["22"])
'
)"

echo "==> tunnelling to $IP:$PORT  (MinIO 9000, Postgres 5533)"
echo "    leave this running for the whole analysis; Ctrl-C to stop."

# ExitOnForwardFailure: fail loudly rather than sit connected with no forward,
# which would look healthy and still refuse every fetch. The reconnect loop is
# because RunPod's exposed-TCP sshd drops connections on its own schedule.
while true; do
  ssh -N \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -p "$PORT" \
    -R 9000:127.0.0.1:9000 \
    -R 5533:127.0.0.1:5533 \
    "root@$IP" || true
  echo "[tunnel] dropped $(date +%T) — reconnecting" >&2
  sleep 3
done
