"""Capability-based repair for unsupported OpenCV FFmpeg color transfers.

Supported streams retain the original upstream reader verbatim. Unsupported
streams use the already installed CPU Decord decoder, with strict original
frame identity, dimensions and average-rate checks. No video-ID policies.
"""
from functools import lru_cache, wraps
import ctypes
import math
import numbers
import operator
from pathlib import Path


@lru_cache(maxsize=1)
def transfer_capability():
    import cv2
    site = Path(cv2.__file__).resolve().parents[1]
    libraries = list(site.glob('opencv_python_headless.libs/libswscale*.so*'))
    libraries += list(site.glob('opencv_python.libs/libswscale*.so*'))
    if len(libraries) != 1:
        raise RuntimeError('The verified Linux OpenCV bundled libswscale is required')
    library = ctypes.CDLL(str(libraries[0]))
    function = library.sws_test_transfer
    function.argtypes = [ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    # Retain the owning handle as well as its function pointer.
    return library, function


def decoder_metadata(path):
    import av
    import cv2
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('Video stream required')
        transfer = int(container.streams.video[0].codec_context.color_trc)
    _, supported = transfer_capability()
    cap = cv2.VideoCapture(str(path))
    try:
        metadata = {'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'fps': float(cap.get(cv2.CAP_PROP_FPS)),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'color_trc': transfer, 'opencv_transfer_supported': supported(transfer, 0) > 0}
    finally:
        cap.release()
    if (metadata['frames'] <= 0 or metadata['width'] <= 0 or metadata['height'] <= 0
            or not math.isfinite(metadata['fps']) or metadata['fps'] <= 0):
        raise ValueError('Complete original frame clock and dimensions required')
    metadata['backend'] = 'opencv_original' if metadata['opencv_transfer_supported'] else 'decord_cpu'
    return metadata


def _indices(values, frames):
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError('Integer source-frame identity required')
        value = operator.index(value)
        if not 0 <= value < frames:
            raise ValueError('Requested frame lies outside the original source')
        result.append(value)
    return sorted(set(result))


def fallback_rgb(path, indices, metadata):
    """Return native-resolution RGB; never resample time or resize the source."""
    import decord
    import numpy as np
    wanted = _indices(indices, metadata['frames'])
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    if len(reader) != metadata['frames'] or not math.isclose(reader.get_avg_fps(), metadata['fps'], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('Alternate decoder changes the declared source frame clock')
    result = {}
    for offset in range(0, len(wanted), 16):
        batch = wanted[offset:offset + 16]
        rgb = reader.get_batch(batch).asnumpy()
        if rgb.dtype != np.uint8 or rgb.shape != (len(batch), metadata['height'], metadata['width'], 3):
            raise ValueError('Alternate decoder changes source dimensions or pixel type')
        result.update((index, frame.copy()) for index, frame in zip(batch, rgb))
    if set(result) != set(wanted):
        raise ValueError('Incomplete alternate-decoder frame population')
    return result


def temporal_reader(original):
    @wraps(original)
    def read(path, frame_ids, max_side, portrait_crop=False):
        metadata = decoder_metadata(path)
        if metadata['opencv_transfer_supported']:
            return original(path, frame_ids, max_side, portrait_crop)
        import cv2
        result = {}
        for index, rgb in fallback_rgb(path, frame_ids, metadata).items():
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]
            if portrait_crop and w > h:
                crop = int(h * 9 / 16)
                frame = frame[:, (w - crop) // 2:(w - crop) // 2 + crop]
                h, w = frame.shape[:2]
                scale = max_side / float(max(h, w))
                frame = cv2.resize(frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_CUBIC)
            else:
                scale = max_side / float(max(h, w))
                if scale < 1:
                    frame = cv2.resize(frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
            result[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return result
    return read


def spatial_reader(original):
    @wraps(original)
    def read(path, max_side):
        metadata = decoder_metadata(path)
        if metadata['opencv_transfer_supported']:
            return original(path, max_side)
        import cv2
        import numpy as np
        count = int(np.floor((metadata['frames'] - 1) / metadata['fps'])) + 1
        requested = {int(round(second * metadata['fps'])): second for second in range(count)}
        result = []
        for index, rgb in fallback_rgb(path, requested, metadata).items():
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]
            scale = max_side / float(max(h, w))
            if scale < 1:
                frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError('JPEG encoding failed; source may not be omitted')
            result.append((requested[index], buffer.tobytes()))
        if len(result) != count:
            raise ValueError('Incomplete per-second source-frame population')
        return result
    return read


def detector_reader(original):
    @wraps(original)
    def read(path, wanted):
        metadata = decoder_metadata(path)
        if metadata['opencv_transfer_supported']:
            yield from original(path, wanted)
            return
        import cv2
        for index, rgb in fallback_rgb(path, wanted, metadata).items():
            yield index, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return read
