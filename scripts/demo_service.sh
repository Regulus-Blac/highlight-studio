#!/usr/bin/env bash
# Run from the deployed candidate root. No downloads, promotion, or paid GPU startup.
set -euo pipefail
root=$(pwd -P)
mode=${1:?Use mock or real}
if [[ "$mode" == mock ]]; then
    bash "$root/mock-service.sh" start
elif [[ "$mode" == real ]]; then
    if ! nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q '[A-Za-z].*'; then
        printf 'GPU_NOT_READY: start the instance with a GPU, then run this connector again.\n' >&2
        exit 2
    fi
    "$root/.venv-f1/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA_NOT_READY"'
    evidence=$(cat "$root/gpu-runtime/de-service/acceptance-dir.txt")
    bash "$root/sj-f1-service.sh" start "$evidence"
    for ((i=0;i<120;i++)); do
        if bash "$root/sj-f1-service.sh" status >/dev/null 2>&1; then
            printf 'Real F1 service ready.\n'; exit 0
        fi
        sleep 1
    done
    printf 'F1 startup timed out. Check gpu-runtime/de-service/server.log\n' >&2
    exit 1
else
    printf 'Unknown mode: %s\n' "$mode" >&2; exit 2
fi
