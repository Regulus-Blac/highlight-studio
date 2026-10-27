"""Explicit synthetic adapter for D/E media integration, never a trained model."""

from services.model_adapter.adapter import MockAdapter, validate_request
from services.media.probe import probe


class SyntheticMediaAdapter(MockAdapter):
    engine = identity = "synthetic-media-v1"

    def predict(self, request):
        validate_request(request)
        metadata = probe(request["video_path"])
        rw, rh = request["target_ratio"]
        width = min(metadata["width"], metadata["height"] * rw / rh)
        height = width * rh / rw
        box = [(metadata["width"] - width) / 2, (metadata["height"] - height) / 2, width]
        count = metadata["frame_count"]
        quarter = max(1, count // 4)
        frames = sorted(set(range(quarter)) | set(range(count - quarter, count)))
        return {**super().predict(request), "engine": self.engine, "predictions": [{"frame": frame, "bboxes": box} for frame in frames]}
