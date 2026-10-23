"""产品层的固定模型接口；不加载上游训练依赖。"""

from typing import Protocol


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
    def predict(self, request: dict) -> dict: ...


class MockAdapter:
    def predict(self, request: dict) -> dict:
        validate_request(request)
        return {
            "api_version": "1",
            "video_id": request["video_id"],
            "target_ratio": request["target_ratio"],
            "engine": "mock-v1",
            "mock": True,
            "predictions": [],
        }
