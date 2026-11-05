#!/usr/bin/env python3
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import spatial_feats
from reliable_decode import fallback, metadata

original = spatial_feats.frames_at_indices


def frames_at_indices(path, indices, max_side):
    if metadata(path)['supported']:
        return original(path, indices, max_side)
    result = []
    for i, rgb in fallback(path, indices):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        scale = min(1., max_side / max(h, w))
        if scale < 1:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        assert ok
        result.append((i, jpg.tobytes()))
    assert [i for i, _ in result] == indices
    print('alternate decoder; sampled', len(indices), 'frames', flush=True)
    return result


if __name__ == '__main__':
    spatial_feats.frames_at_indices = frames_at_indices
    sys.exit(spatial_feats.main())
