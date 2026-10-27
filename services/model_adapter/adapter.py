"""产品层的固定模型接口；不加载上游训练依赖。"""

import math
from typing import Protocol


class AdapterError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable

    def as_dict(self):
        return {"code": self.code, "message": str(self), "retryable": self.retryable}


def validate_request(request: dict) -> None:
    fields = {"api_version", "video_id", "video_path", "target_ratio"}
    if not isinstance(request, dict) or set(request) != fields:
        raise ValueError("Expected api_version, video_id, video_path and target_ratio")
    if request["api_version"] != "1":
        raise ValueError("Unsupported api_version")
    for field in ("video_id", "video_path"):
        if not isinstance(request[field], str) or not request[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    ratio = request["target_ratio"]
    if not isinstance(ratio, list) or len(ratio) != 2 or any(type(x) is not int or x <= 0 for x in ratio):
        raise ValueError("target_ratio must contain two positive integers")


class ModelAdapter(Protocol):
    engine: str
    mock: bool
    identity: str

    def predict(self, request: dict) -> dict: ...


class MockAdapter:
    engine = "mock-v1"
    mock = True
    identity = "mock-v1"
    available = True

    def predict(self, request: dict) -> dict:
        validate_request(request)
        return {
            "api_version": "1",
            "video_id": request["video_id"],
            "target_ratio": request["target_ratio"],
            "engine": self.engine,
            "mock": True,
            "predictions": [],
        }


def validate_result(result: dict, request: dict) -> None:
    """Validate the v1 boundary, including identity and source-frame semantics."""
    fields = {"api_version", "video_id", "target_ratio", "engine", "mock", "predictions"}
    if not isinstance(result, dict) or set(result) != fields:
        raise ValueError("Prediction result has unexpected fields")
    validate_request({"api_version": result["api_version"], "video_id": result["video_id"], "video_path": request["video_path"], "target_ratio": result["target_ratio"]})
    for field in ("api_version", "video_id", "target_ratio"):
        if result[field] != request[field]:
            raise ValueError(f"Prediction {field} does not match the request")
    if not isinstance(result["engine"], str) or not result["engine"].strip() or type(result["mock"]) is not bool:
        raise ValueError("Prediction engine and mock flag are required")
    if not isinstance(result["predictions"], list):
        raise ValueError("predictions must be an array")
    previous = -1
    for prediction in result["predictions"]:
        if not isinstance(prediction, dict) or set(prediction) != {"frame", "bboxes"}:
            raise ValueError("Each prediction requires frame and bboxes")
        frame, box = prediction["frame"], prediction["bboxes"]
        if type(frame) is not int or frame <= previous:
            raise ValueError("Original frame numbers must be non-negative, unique and increasing")
        if not isinstance(box, list) or len(box) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in box):
            raise ValueError("bboxes must contain three finite numbers")
        if box[0] < 0 or box[1] < 0 or box[2] <= 0:
            raise ValueError("bboxes requires non-negative x/y and positive width")
        previous = frame
