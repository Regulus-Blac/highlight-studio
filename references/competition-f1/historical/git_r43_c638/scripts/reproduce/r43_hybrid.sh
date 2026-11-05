#!/bin/bash
set -euo pipefail
cd /root/sj-fs/work/sym41_8a18db2
exec 9>/root/sj-fs/r1t_experiment.lock
flock 9
OUT=outputs/experiments/sym43_crop_context
echo $$ > "$OUT/hybrid.pid"
trap 'echo $? > "$OUT/hybrid.exit"' EXIT
test "$(cat "$OUT/queue.exit")" = 0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
/root/sj-fs/envs/hl/bin/python -u scripts/reproduce/r43_hybrid.py
