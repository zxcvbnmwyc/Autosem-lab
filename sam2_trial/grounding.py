"""Qwen visual grounding adapter for the local SAM2 prototype.

The adapter receives an image plus a referring expression and returns one or
more coarse, normalized boxes. It deliberately does not perform segmentation:
SAM2 remains the source of masks and contours.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image


DEFAULT_BASE_URL = (
    "https://ws-jezurpmuo05q16c9.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_MODEL = "qwen3-vl-flash"
MAX_GROUNDING_EDGE = 1600
MAX_IMAGE_DATA_BYTES = 10 * 1024 * 1024
MAX_CANDIDATES = 3
NORMALIZED_COORDINATE_MAX = 1000.0
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class GroundingError(RuntimeError):
    """A safe, user-facing failure from the visual grounding layer."""


@dataclass(frozen=True)
class GroundingCandidate:
    """A candidate box and optional interior anchor point in 0–1000 space."""

    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float
    label: str | None
    point_x: float | None = None
    point_y: float | None = None

    def absolute_box(self, width: int, height: int) -> list[float]:
        if width < 2 or height < 2:
            raise ValueError("图片至少需要 2 个像素宽和高。")
        x_scale = (width - 1) / NORMALIZED_COORDINATE_MAX
        y_scale = (height - 1) / NORMALIZED_COORDINATE_MAX
        return [
            self.x0 * x_scale,
            self.y0 * y_scale,
            self.x1 * x_scale,
            self.y1 * y_scale,
        ]

    def absolute_point(self, width: int, height: int) -> list[float] | None:
        if self.point_x is None or self.point_y is None:
            return None
        if width < 2 or height < 2:
            raise ValueError("图片至少需要 2 个像素宽和高。")
        return [
            self.point_x * (width - 1) / NORMALIZED_COORDINATE_MAX,
            self.point_y * (height - 1) / NORMALIZED_COORDINATE_MAX,
        ]

    def as_metadata(self, width: int, height: int) -> dict[str, Any]:
        point_1000 = [self.point_x, self.point_y] if self.point_x is not None and self.point_y is not None else None
        return {
            "box_xyxy": self.absolute_box(width, height),
            "box_1000": [self.x0, self.y0, self.x1, self.y1],
            "point_xy": self.absolute_point(width, height),
            "point_1000": point_1000,
            "confidence": self.confidence,
            "label": self.label,
        }


@dataclass(frozen=True)
class GroundingProposal:
    """Validated response data without raw upstream model text."""

    status: str
    candidates: tuple[GroundingCandidate, ...]
    note: str | None

    def as_public(self, width: int, height: int) -> dict[str, Any]:
        candidates = [
            candidate.as_metadata(width, height) for candidate in self.candidates
        ]
        return {
            "status": self.status,
            "note": self.note,
            "candidates": candidates,
            "candidate": candidates[0] if candidates else None,
        }


def load_local_dotenv(path: Path) -> None:
    """Load simple KEY=value entries without overriding real environment values."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not ENV_KEY_RE.fullmatch(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise GroundingError(f"Qwen 返回的 {name} 不是数字。")
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError as error:
            raise GroundingError(f"Qwen 返回的 {name} 不是数字。") from error
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise GroundingError(f"Qwen 返回的 {name} 不是数字。")
    if not math.isfinite(number):
        raise GroundingError(f"Qwen 返回的 {name} 不是有限数字。")
    return number


def _parse_candidate(value: Any, index: int) -> GroundingCandidate:
    if not isinstance(value, dict):
        raise GroundingError(f"Qwen 返回的第 {index} 个候选框格式错误。")
    x0 = _number(value.get("x0"), f"boxes[{index}].x0")
    y0 = _number(value.get("y0"), f"boxes[{index}].y0")
    x1 = _number(value.get("x1"), f"boxes[{index}].x1")
    y1 = _number(value.get("y1"), f"boxes[{index}].y1")
    confidence = _number(value.get("confidence"), f"boxes[{index}].confidence")
    if not all(
        0 <= coordinate <= NORMALIZED_COORDINATE_MAX
        for coordinate in (x0, y0, x1, y1)
    ):
        raise GroundingError("Qwen 返回的候选框超出 0 到 1000 的坐标范围。")
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise GroundingError("Qwen 返回的候选框过小。")
    if not 0 <= confidence <= 1:
        raise GroundingError("Qwen 返回的候选框置信度不在 0 到 1 之间。")
    label = value.get("label")
    if label is not None and not isinstance(label, str):
        raise GroundingError("Qwen 返回的候选框标签不是文字。")
    label = label.strip()[:80] if isinstance(label, str) else None
    point = value.get("point")
    point_x: float | None = None
    point_y: float | None = None
    if point is not None:
        if not isinstance(point, dict):
            raise GroundingError(f"Qwen 返回的第 {index} 个候选内部点格式错误。")
        point_x = _number(point.get("x"), f"boxes[{index}].point.x")
        point_y = _number(point.get("y"), f"boxes[{index}].point.y")
        if not 0 <= point_x <= NORMALIZED_COORDINATE_MAX or not 0 <= point_y <= NORMALIZED_COORDINATE_MAX:
            raise GroundingError("Qwen 返回的候选内部点超出 0 到 1000 的坐标范围。")
        if not x0 < point_x < x1 or not y0 < point_y < y1:
            raise GroundingError("Qwen 返回的候选内部点必须位于候选框内部。")
    return GroundingCandidate(x0, y0, x1, y1, confidence, label or None, point_x, point_y)


def parse_grounding_payload(value: Any) -> GroundingProposal:
    """Validate a model JSON payload before it reaches SAM2."""
    if not isinstance(value, dict):
        raise GroundingError("Qwen 没有返回 JSON 对象。")
    status = value.get("status")
    if status not in {"found", "ambiguous", "not_found"}:
        raise GroundingError("Qwen 返回了未知的定位状态。")
    boxes = value.get("boxes")
    if not isinstance(boxes, list):
        raise GroundingError("Qwen 返回中缺少 boxes 数组。")
    if len(boxes) > MAX_CANDIDATES:
        boxes = boxes[:MAX_CANDIDATES]
    candidates = tuple(
        _parse_candidate(candidate, index) for index, candidate in enumerate(boxes)
    )
    if status == "found" and not candidates:
        raise GroundingError("Qwen 声称找到目标，但没有给出候选框。")
    if status == "not_found" and candidates:
        raise GroundingError("Qwen 同时返回了未找到状态和候选框。")

    note = value.get("note")
    if note is not None and not isinstance(note, str):
        raise GroundingError("Qwen 返回的说明不是文字。")
    note = note.strip()[:240] if isinstance(note, str) else None
    return GroundingProposal(status, candidates, note or None)


def _strip_code_fence(content: str) -> str:
    value = content.strip()
    fence = chr(96) * 3
    if not value.startswith(fence):
        return value
    lines = value.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == fence:
        return "\n".join(lines[1:-1]).strip()
    return value


def _data_url_for_image(image_rgb: np.ndarray) -> str:
    """Compress RGB input and return the OpenAI-compatible image data URL."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("grounding image must have shape (height, width, 3).")
    height, width = image_rgb.shape[:2]
    if min(height, width) < 2:
        raise ValueError("图片至少需要 2 个像素宽和高。")

    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    scale = min(1.0, MAX_GROUNDING_EDGE / max(image.size))
    if scale < 1.0:
        resized = (
            max(2, round(image.width * scale)),
            max(2, round(image.height * scale)),
        )
        image = image.resize(resized, Image.Resampling.LANCZOS)

    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=88, optimize=True)
    image_bytes = encoded.getvalue()
    if len(image_bytes) > MAX_IMAGE_DATA_BYTES:
        raise GroundingError("图片压缩后仍然过大，暂时不能发送给阿里云百炼。")
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")


def _model_request(
    image_data_url: str, description: str, model: str
) -> dict[str, Any]:
    system = (
        "You are the visual grounding stage before SAM2 segmentation. "
        "Return JSON only, with no markdown. Use a normalized image coordinate "
        "system: x grows left to right, y grows top to bottom, and every "
        "coordinate is in the inclusive 0 to 1000 range. "
        "Return exactly this JSON shape: "
        '{"status":"found|ambiguous|not_found","boxes":'
        '[{"x0":number,"y0":number,"x1":number,"y1":number,'
        '"confidence":number,"label":string|null,'
        '"point":{"x":number,"y":number}|null}],"note":string|null}. '
        "Choose at most three tight boxes. If the target is absent, return "
        "status not_found and an empty boxes array. "
        "For each found object, return point only when you can place it clearly "
        "inside the visible target and away from its boundary; otherwise use null. "
        "Do not claim pixel-perfect contours; SAM2 will refine the prompt."
    )
    return {
        "model": model,
        "enable_thinking": False,
        "temperature": 0,
        "max_completion_tokens": 320,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": "Target description: " + description},
                ],
            },
        ],
    }


def _model_response_text(value: Any) -> str:
    if not isinstance(value, dict):
        raise GroundingError("百炼响应格式不正确。")
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GroundingError("百炼没有返回可用候选结果。")
    first = choices[0]
    if not isinstance(first, dict):
        raise GroundingError("百炼返回的候选结果格式不正确。")
    message = first.get("message")
    if not isinstance(message, dict):
        raise GroundingError("百炼返回中缺少 message。")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text_parts = [
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if text_parts:
            return "".join(text_parts)
    raise GroundingError("百炼没有返回可解析的候选框 JSON。")


class QwenGrounder:
    """One non-streaming Qwen Chat Completions call, with no API key logging."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._timeout_seconds = timeout_seconds

    @property
    def model(self) -> str:
        value = self._model if self._model is not None else os.getenv(
            "DASHSCOPE_MODEL", DEFAULT_MODEL
        )
        return value.strip() or DEFAULT_MODEL

    @property
    def configured(self) -> bool:
        return bool(self._api_key_value())

    def _api_key_value(self) -> str:
        value = self._api_key
        if value is None:
            value = os.getenv("DASHSCOPE_API_KEY", "")
        return value.strip()

    def _base_url_value(self) -> str:
        value = self._base_url
        if value is None:
            value = os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
        value = value.strip().rstrip("/")
        parsed = urlparse(value)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        is_model_studio_host = (
            hostname in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"}
            or hostname.endswith(".maas.aliyuncs.com")
        )
        if (
            parsed.scheme != "https"
            or not is_model_studio_host
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path != "/compatible-mode/v1"
        ):
            raise GroundingError(
                "DASHSCOPE_BASE_URL 必须是阿里云百炼的 HTTPS "
                "OpenAI 兼容地址，且以 /compatible-mode/v1 结尾。"
            )
        return value

    def _timeout_value(self) -> float:
        value: str | float | None = self._timeout_seconds
        if value is None:
            value = os.getenv("DASHSCOPE_TIMEOUT_SECONDS", "45")
        try:
            timeout = float(value)
        except (TypeError, ValueError) as error:
            raise GroundingError("DASHSCOPE_TIMEOUT_SECONDS 必须是数字。") from error
        if not 5 <= timeout <= 120:
            raise GroundingError(
                "DASHSCOPE_TIMEOUT_SECONDS 必须在 5 到 120 秒之间。"
            )
        return timeout

    def _model_value(self) -> str:
        model = self.model
        if not MODEL_NAME_RE.fullmatch(model):
            raise GroundingError(
                "DASHSCOPE_MODEL 只能包含字母、数字、点、下划线和连字符。"
            )
        return model

    def ground(self, image_rgb: np.ndarray, description: str) -> GroundingProposal:
        api_key = self._api_key_value()
        if not api_key:
            raise GroundingError(
                "尚未配置 DASHSCOPE_API_KEY。复制 .env.example 为 .env 后填入本机密钥。"
            )

        endpoint = self._base_url_value() + "/chat/completions"
        payload = _model_request(
            _data_url_for_image(image_rgb), description, self._model_value()
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_value()) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            if error.code in {401, 403}:
                raise GroundingError(
                    "百炼鉴权失败，请确认 API Key 属于这个北京业务空间且模型已开通。"
                ) from error
            if error.code == 429:
                raise GroundingError("百炼当前限流，请稍后重试。") from error
            raise GroundingError(
                f"百炼接口暂时不可用（HTTP {error.code}）。"
            ) from error
        except urllib.error.URLError as error:
            raise GroundingError("无法连接阿里云百炼，请检查网络和服务地址。") from error
        except TimeoutError as error:
            raise GroundingError("Qwen 请求超时，请稍后重试。") from error

        try:
            response_payload = json.loads(response_body)
            model_payload = json.loads(
                _strip_code_fence(_model_response_text(response_payload))
            )
        except (TypeError, json.JSONDecodeError, GroundingError) as error:
            raise GroundingError("百炼返回的数据无法解析为候选框 JSON。") from error
        return parse_grounding_payload(model_payload)
