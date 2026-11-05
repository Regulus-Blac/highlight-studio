#!/usr/bin/env python3
"""Append aligned VideoMAE appearance/change features to each TimeLens feature layer."""
import json
import sys
from pathlib import Path

import numpy as np

base, motion, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
count = 0
for path in sorted(base.glob('*.npz')):
    a, b = np.load(path), np.load(motion / path.name)
    features = {key: a[key] for key in a.files}
    for key in ['vis', 'last']:
        assert a[key].shape[0] == b[key].shape[0], path.name
        features[key] = np.concatenate([a[key], np.repeat(b[key], a[key].shape[1], axis=1)], axis=2)
    np.savez(out / path.name, **features)
    count += 1
print(json.dumps({'videos': count, 'timelens': str(base), 'motion': str(motion), 'out': str(out)}))
