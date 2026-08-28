"""AutoSEM: local-first image-to-contour website.

Qwen can turn a description into a coarse box; SAM2 receives only spatial
prompts and makes the final mask.  SAM2 runs through one durable background
worker so CPU inference never blocks the website or an HTTP request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from flask import Flask, abort, jsonify, render_template, request, send_from_directory, session
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from contours import mask_to_contours
from grounding import (
    GroundingCandidate,
    GroundingError,
    GroundingProposal,
    QwenGrounder,
    load_local_dotenv,
)


APP_DIR = Path(__file__).resolve().parent
load_local_dotenv(APP_DIR / ".env")


def _static_asset_version() -> str:
    digest = hashlib.sha256()
    for asset_name in ("style.css", "app.js"):
        try:
            digest.update((APP_DIR / "static" / asset_name).read_bytes())
        except OSError:
            continue
    return digest.hexdigest()[:12]


STATIC_ASSET_VERSION = _static_asset_version()


def _environment_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
IMAGE_PREVIEW_DIR = DATA_DIR / "previews"
IMAGE_META_DIR = DATA_DIR / "images"
GROUNDING_DIR = DATA_DIR / "groundings"
JOBS_DIR = DATA_DIR / "jobs"
RESULTS_DIR = DATA_DIR / "results"

DEFAULT_SAM2_SOURCE = r"C:\Users\11609\Documents\Autosem\vendor\sam2-main"
DEFAULT_TINY_CHECKPOINT = r"C:\Users\11609\Documents\Autosem\models\sam2.1_hiera_tiny.pt"
SAM2_SOURCE = Path(os.environ.get("SAM2_SOURCE", DEFAULT_SAM2_SOURCE))
SAM2_VARIANTS = {
    "tiny": ("sam2.1_hiera_tiny", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
SAM2_VARIANT = os.environ.get("SAM2_MODEL_VARIANT", "tiny").strip().lower()
if SAM2_VARIANT not in SAM2_VARIANTS:
    SAM2_VARIANT = "tiny"
DEFAULT_MODEL_NAME, DEFAULT_MODEL_CONFIG = SAM2_VARIANTS[SAM2_VARIANT]
SAM2_MODEL_NAME = os.environ.get("SAM2_MODEL_NAME", DEFAULT_MODEL_NAME).strip()
MODEL_CONFIG = os.environ.get("SAM2_MODEL_CONFIG", DEFAULT_MODEL_CONFIG).strip()
CHECKPOINT = Path(os.environ.get("SAM2_CHECKPOINT", DEFAULT_TINY_CHECKPOINT))
SAM2_DEVICE_SETTING = os.environ.get("SAM2_DEVICE", "cpu").strip().lower()
SAM2_CPU_THREADS = _environment_int("SAM2_CPU_THREADS", 4, 1)
SAM2_MAX_IMAGE_EDGE = _environment_int("SAM2_MAX_IMAGE_EDGE", 1280, 0)
MAX_JOB_QUEUE = _environment_int("SAM2_MAX_QUEUE", 8, 1)
DATA_TTL_HOURS = _environment_int("DATA_TTL_HOURS", 72, 0)
MAX_UPLOAD_BYTES = _environment_int("MAX_UPLOAD_MB", 25, 1) * 1024 * 1024
MAX_IMAGE_PIXELS = _environment_int("MAX_IMAGE_PIXELS", 12_000_000, 1)
DISPLAY_MAX_IMAGE_EDGE = _environment_int("DISPLAY_MAX_IMAGE_EDGE", 1600, 256)
RESULT_PREVIEW_MAX_EDGE = _environment_int("RESULT_PREVIEW_MAX_EDGE", 1200, 256)
IMAGE_JPEG_QUALITY = min(_environment_int("IMAGE_JPEG_QUALITY", 92, 50), 95)
PREVIEW_JPEG_QUALITY = min(_environment_int("PREVIEW_JPEG_QUALITY", 84, 40), 92)
CLEANUP_INTERVAL_SECONDS = _environment_int("CLEANUP_INTERVAL_SECONDS", 15 * 60, 60)
MAX_PROMPT_POINTS = 24
MAX_DESCRIPTION_CHARS = 500
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
RESULT_FILE_NAMES = {"mask.png", "overlay.png", "preview.jpg", "contours.png", "result.json"}

if str(SAM2_SOURCE) not in sys.path:
    sys.path.insert(0, str(SAM2_SOURCE))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _expiry_timestamp() -> str | None:
    if DATA_TTL_HOURS == 0:
        return None
    return (datetime.now(UTC) + timedelta(hours=DATA_TTL_HOURS)).isoformat()


def _safe_iso_before_now(value: str | None) -> bool:
    if not value:
        return False
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp < datetime.now(UTC)


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one metadata file inside our fixed data directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _json_read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    filename: str
    original_name: str
    width: int
    height: int
    owner_id: str
    created_at: str
    expires_at: str | None
    preview_filename: str | None = None
    preview_width: int | None = None
    preview_height: int | None = None

    @property
    def path(self) -> Path:
        return UPLOAD_DIR / self.filename

    @property
    def preview_path(self) -> Path | None:
        return IMAGE_PREVIEW_DIR / self.preview_filename if self.preview_filename else None

    def as_storage(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "original_name": self.original_name,
            "width": self.width,
            "height": self.height,
            "owner_id": self.owner_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "preview_filename": self.preview_filename,
            "preview_width": self.preview_width,
            "preview_height": self.preview_height,
        }

    @classmethod
    def from_storage(cls, value: dict[str, Any]) -> "ImageRecord | None":
        try:
            record = cls(
                image_id=str(value["image_id"]),
                filename=str(value["filename"]),
                original_name=str(value["original_name"]),
                width=int(value["width"]),
                height=int(value["height"]),
                owner_id=str(value["owner_id"]),
                created_at=str(value["created_at"]),
                expires_at=value.get("expires_at") if isinstance(value.get("expires_at"), str) else None,
                preview_filename=value.get("preview_filename") if isinstance(value.get("preview_filename"), str) else None,
                preview_width=int(value["preview_width"]) if value.get("preview_width") is not None else None,
                preview_height=int(value["preview_height"]) if value.get("preview_height") is not None else None,
            )
        except (KeyError, TypeError, ValueError):
            return None
        if (
            not IMAGE_ID_RE.fullmatch(record.image_id)
            or record.filename not in {f"{record.image_id}.png", f"{record.image_id}.jpg"}
            or record.width < 2
            or record.height < 2
            or not IMAGE_ID_RE.fullmatch(record.owner_id)
            or (record.preview_filename is not None and record.preview_filename != f"{record.image_id}.jpg")
            or (record.preview_filename is None and (record.preview_width is not None or record.preview_height is not None))
            or (record.preview_filename is not None and (record.preview_width is None or record.preview_height is None))
            or (record.preview_width is not None and record.preview_width < 2)
            or (record.preview_height is not None and record.preview_height < 2)
        ):
            return None
        return record


@dataclass(frozen=True)
class GroundingRecord:
    """A server-issued link between one image, description and Qwen response."""

    grounding_id: str
    image_id: str
    description: str
    model: str
    proposal: GroundingProposal
    created_at: str
    owner_id: str = ""

    def as_metadata(self, width: int, height: int) -> dict[str, Any]:
        return {
            "id": self.grounding_id,
            "provider": "Alibaba Cloud Model Studio",
            "model": self.model,
            "description": self.description,
            "created_at": self.created_at,
            **self.proposal.as_public(width, height),
        }

    def as_storage(self) -> dict[str, Any]:
        return {
            "grounding_id": self.grounding_id,
            "image_id": self.image_id,
            "description": self.description,
            "model": self.model,
            "created_at": self.created_at,
            "owner_id": self.owner_id,
            "proposal": {
                "status": self.proposal.status,
                "note": self.proposal.note,
                "candidates": [
                    {
                        "x0": candidate.x0,
                        "y0": candidate.y0,
                        "x1": candidate.x1,
                        "y1": candidate.y1,
                        "confidence": candidate.confidence,
                        "label": candidate.label,
                    }
                    for candidate in self.proposal.candidates
                ],
            },
        }

    @classmethod
    def from_storage(cls, value: dict[str, Any]) -> "GroundingRecord | None":
        try:
            raw_proposal = value["proposal"]
            raw_candidates = raw_proposal["candidates"]
            if not isinstance(raw_candidates, list):
                return None
            candidates = tuple(
                GroundingCandidate(
                    float(candidate["x0"]),
                    float(candidate["y0"]),
                    float(candidate["x1"]),
                    float(candidate["y1"]),
                    float(candidate["confidence"]),
                    candidate.get("label") if isinstance(candidate.get("label"), str) else None,
                )
                for candidate in raw_candidates
                if isinstance(candidate, dict)
            )
            record = cls(
                grounding_id=str(value["grounding_id"]),
                image_id=str(value["image_id"]),
                description=str(value["description"]),
                model=str(value["model"]),
                proposal=GroundingProposal(
                    str(raw_proposal["status"]),
                    candidates,
                    raw_proposal.get("note") if isinstance(raw_proposal.get("note"), str) else None,
                ),
                created_at=str(value["created_at"]),
                owner_id=str(value.get("owner_id", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not IMAGE_ID_RE.fullmatch(record.grounding_id) or not IMAGE_ID_RE.fullmatch(record.image_id):
            return None
        return record


class Sam2ConfigurationError(RuntimeError):
    """A configuration error that is safe to return through a job."""


def _resolve_device() -> str:
    if SAM2_DEVICE_SETTING == "cpu":
        return "cpu"
    if SAM2_DEVICE_SETTING == "cuda":
        if not torch.cuda.is_available():
            raise Sam2ConfigurationError("已选择 CUDA，但当前没有可用的 NVIDIA CUDA 设备。")
        return "cuda"
    if SAM2_DEVICE_SETTING == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise Sam2ConfigurationError("SAM2_DEVICE 只能是 cpu、cuda 或 auto。")


class Sam2Engine:
    """Lazily load one SAM2 predictor and serialize all model access."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._predictor: Any | None = None
        self._current_image_key: str | None = None
        self.device = _resolve_device()

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    def configuration_ready(self) -> bool:
        return SAM2_SOURCE.is_dir() and CHECKPOINT.is_file() and bool(MODEL_CONFIG)

    def public_status(self) -> dict[str, Any]:
        return {
            "model": SAM2_MODEL_NAME,
            "variant": SAM2_VARIANT,
            "device": self.device,
            "loaded": self.loaded,
            "configured": self.configuration_ready(),
            "max_image_edge": SAM2_MAX_IMAGE_EDGE if self.device == "cpu" else 0,
        }

    def _ensure_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        if not SAM2_SOURCE.is_dir():
            raise Sam2ConfigurationError("找不到 SAM2 源码目录，请检查 SAM2_SOURCE。")
        if not CHECKPOINT.is_file():
            raise Sam2ConfigurationError("找不到 SAM2 权重。下载完成后，请检查 SAM2_CHECKPOINT。")
        if not MODEL_CONFIG:
            raise Sam2ConfigurationError("没有设置 SAM2 模型配置。")

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if self.device == "cpu":
            torch.set_num_threads(SAM2_CPU_THREADS)
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        model = build_sam2(MODEL_CONFIG, ckpt_path=str(CHECKPOINT), device=self.device, mode="eval")
        model.eval()
        self._predictor = SAM2ImagePredictor(model)
        return self._predictor

    @staticmethod
    def _scale_prompts(
        image_rgb: np.ndarray,
        point_coords: np.ndarray | None,
        box: np.ndarray | None,
        max_edge: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
        """Downscale CPU inputs while returning the final mask at source scale."""
        height, width = image_rgb.shape[:2]
        largest_edge = max(width, height)
        if max_edge <= 0 or largest_edge <= max_edge:
            return image_rgb, point_coords, box, 1.0
        scale = max_edge / largest_edge
        resized = cv2.resize(
            image_rgb,
            (max(2, round(width * scale)), max(2, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return (
            resized,
            point_coords * scale if point_coords is not None else None,
            box * scale if box is not None else None,
            scale,
        )

    def segment(
        self,
        image_id: str,
        image_rgb: np.ndarray,
        point_coords: np.ndarray | None,
        point_labels: np.ndarray | None,
        box: np.ndarray | None,
    ) -> tuple[np.ndarray, float, int]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("图片不是可供 SAM2 使用的 RGB 图像。")
        original_height, original_width = image_rgb.shape[:2]
        with self._lock:
            predictor = self._ensure_predictor()
            max_edge = SAM2_MAX_IMAGE_EDGE if self.device == "cpu" else 0
            model_image, model_points, model_box, scale = self._scale_prompts(
                np.ascontiguousarray(image_rgb), point_coords, box, max_edge
            )
            cache_key = f"{image_id}:{model_image.shape[1]}x{model_image.shape[0]}:{scale:.6f}"
            with torch.inference_mode():
                if self._current_image_key != cache_key:
                    predictor.set_image(np.ascontiguousarray(model_image))
                    self._current_image_key = cache_key
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        masks, scores, _ = predictor.predict(
                            point_coords=model_points,
                            point_labels=point_labels,
                            box=model_box,
                            multimask_output=True,
                            return_logits=False,
                        )
                else:
                    masks, scores, _ = predictor.predict(
                        point_coords=model_points,
                        point_labels=point_labels,
                        box=model_box,
                        multimask_output=True,
                        return_logits=False,
                    )

            selected_index = int(np.argmax(scores))
            mask = np.asarray(masks[selected_index], dtype=np.uint8)
            if scale != 1.0:
                mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            return mask.astype(bool), float(scores[selected_index]), selected_index


@dataclass
class JobRecord:
    job_id: str
    image_id: str
    owner_id: str
    input_payload: dict[str, Any]
    created_at: str
    expires_at: str | None
    status: str = "queued"
    phase: str = "queued"
    message: str = "任务已排队。"
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def as_storage(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "image_id": self.image_id,
            "owner_id": self.owner_id,
            "input_payload": self.input_payload,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_storage(cls, value: dict[str, Any]) -> "JobRecord | None":
        try:
            record = cls(
                job_id=str(value["job_id"]),
                image_id=str(value["image_id"]),
                owner_id=str(value["owner_id"]),
                input_payload=value["input_payload"],
                created_at=str(value["created_at"]),
                expires_at=value.get("expires_at") if isinstance(value.get("expires_at"), str) else None,
                status=str(value.get("status", "failed")),
                phase=str(value.get("phase", "failed")),
                message=str(value.get("message", "")),
                started_at=value.get("started_at") if isinstance(value.get("started_at"), str) else None,
                completed_at=value.get("completed_at") if isinstance(value.get("completed_at"), str) else None,
                result=value.get("result") if isinstance(value.get("result"), dict) else None,
                error=value.get("error") if isinstance(value.get("error"), str) else None,
            )
        except (KeyError, TypeError, ValueError):
            return None
        if (
            not IMAGE_ID_RE.fullmatch(record.job_id)
            or not IMAGE_ID_RE.fullmatch(record.image_id)
            or not IMAGE_ID_RE.fullmatch(record.owner_id)
            or not isinstance(record.input_payload, dict)
        ):
            return None
        return record

    def as_public(self, queue_depth: int) -> dict[str, Any]:
        elapsed = None
        if self.started_at:
            try:
                started = datetime.fromisoformat(self.started_at)
                if started.tzinfo is None:
                    started = started.replace(tzinfo=UTC)
                end = datetime.now(UTC)
                if self.completed_at:
                    end = datetime.fromisoformat(self.completed_at)
                    if end.tzinfo is None:
                        end = end.replace(tzinfo=UTC)
                elapsed = max(0, round((end - started).total_seconds()))
            except ValueError:
                elapsed = None
        return {
            "job_id": self.job_id,
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": elapsed,
            "queue_depth": queue_depth,
            "poll_after_ms": 1500 if self.status in {"queued", "running"} else None,
            "result": self.result if self.status == "succeeded" else None,
            "error": self.error if self.status == "failed" else None,
        }


class SlidingWindowLimiter:
    """A small local guardrail; production can replace it with a shared limiter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[tuple[str, str], list[float]] = {}

    def allow(self, owner_id: str, action: str, limit: int, seconds: int) -> int | None:
        now = time.monotonic()
        key = (owner_id, action)
        with self._lock:
            events = [event for event in self._events.get(key, []) if event > now - seconds]
            if len(events) >= limit:
                retry = max(1, math.ceil(seconds - (now - events[0])))
                self._events[key] = events
                return retry
            events.append(now)
            self._events[key] = events
        return None


records: dict[str, ImageRecord] = {}
records_lock = threading.Lock()
grounding_records: dict[str, GroundingRecord] = {}
grounding_records_lock = threading.Lock()
engine = Sam2Engine()
grounder = QwenGrounder()
limiter = SlidingWindowLimiter()

for directory in (UPLOAD_DIR, IMAGE_PREVIEW_DIR, IMAGE_META_DIR, GROUNDING_DIR, JOBS_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SECRET_KEY=os.environ.get("APP_SECRET_KEY") or "autosem-local-development-only-change-me",
    PERMANENT_SESSION_LIFETIME=timedelta(days=DATA_TTL_HOURS or 30),
)


@app.context_processor
def _static_template_context() -> dict[str, str]:
    return {"static_version": STATIC_ASSET_VERSION}


def _current_owner() -> str:
    owner_id = session.get("autosem_owner_id")
    if not isinstance(owner_id, str) or not IMAGE_ID_RE.fullmatch(owner_id):
        owner_id = uuid.uuid4().hex
        session["autosem_owner_id"] = owner_id
        session.permanent = True
    return owner_id


def _json_error(message: str, status: int, retry_after: int | None = None) -> Any:
    if retry_after is None:
        return jsonify({"error": message}), status
    return jsonify({"error": message}), status, {"Retry-After": str(retry_after)}


def _limit_or_error(owner_id: str, action: str, limit: int, seconds: int) -> Any | None:
    retry_after = limiter.allow(owner_id, action, limit, seconds)
    return _json_error("请求过于频繁，请稍后再试。", 429, retry_after) if retry_after else None


def _image_meta_path(image_id: str) -> Path:
    return IMAGE_META_DIR / f"{image_id}.json"


def _grounding_path(grounding_id: str) -> Path:
    return GROUNDING_DIR / f"{grounding_id}.json"


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _load_image_record(image_id: str) -> ImageRecord | None:
    with records_lock:
        cached = records.get(image_id)
    if cached is not None:
        return cached
    storage = _json_read(_image_meta_path(image_id))
    record = ImageRecord.from_storage(storage) if storage else None
    if record is not None:
        with records_lock:
            records[image_id] = record
    return record


def _require_record(image_id: Any) -> ImageRecord:
    if not isinstance(image_id, str) or not IMAGE_ID_RE.fullmatch(image_id):
        abort(404)
    record = _load_image_record(image_id)
    if (
        record is None
        or record.owner_id != _current_owner()
        or _safe_iso_before_now(record.expires_at)
        or not record.path.is_file()
    ):
        abort(404)
    return record


def _load_grounding(grounding_id: str) -> GroundingRecord | None:
    with grounding_records_lock:
        cached = grounding_records.get(grounding_id)
    if cached is not None:
        return cached
    storage = _json_read(_grounding_path(grounding_id))
    record = GroundingRecord.from_storage(storage) if storage else None
    if record is not None:
        with grounding_records_lock:
            grounding_records[grounding_id] = record
    return record


def _require_grounding(grounding_id: Any, image_id: str, description: str | None) -> GroundingRecord | None:
    if grounding_id is None:
        return None
    if not isinstance(grounding_id, str) or not IMAGE_ID_RE.fullmatch(grounding_id):
        raise ValueError("自动定位记录无效，请重新点击 Qwen 自动定位。")
    grounding = _load_grounding(grounding_id)
    if grounding is None:
        raise ValueError("自动定位记录已失效，请重新点击 Qwen 自动定位。")
    if grounding.owner_id and grounding.owner_id != _current_owner():
        raise ValueError("自动定位记录不属于当前浏览器。")
    if grounding.image_id != image_id:
        raise ValueError("自动定位记录不属于当前图片。")
    if grounding.description != description:
        raise ValueError("目标描述已修改，请重新点击 Qwen 自动定位。")
    return grounding


def _parse_grounding_candidate_index(value: Any, grounding: GroundingRecord | None) -> int | None:
    if grounding is None:
        if value is not None:
            raise ValueError("当前请求没有可关联的 Qwen 候选框。")
        return None
    if value is None:
        return 0 if grounding.proposal.candidates else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Qwen 候选框序号必须是整数。")
    if not 0 <= value < len(grounding.proposal.candidates):
        raise ValueError("Qwen 候选框序号超出范围。")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字。")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数字。")
    return number


def _coordinate(value: Any, name: str, maximum: int) -> float:
    number = _number(value, name)
    if number < 0 or number > maximum - 1:
        raise ValueError(f"{name} 超出图片范围。")
    return number


def _parse_points(value: Any, width: int, height: int) -> tuple[np.ndarray | None, np.ndarray | None, list[dict[str, float | int]]]:
    if value is None:
        return None, None, []
    if not isinstance(value, list):
        raise ValueError("points 必须是数组。")
    if len(value) > MAX_PROMPT_POINTS:
        raise ValueError(f"最多支持 {MAX_PROMPT_POINTS} 个点提示。")
    normalized: list[dict[str, float | int]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 个点提示格式不正确。")
        x = _coordinate(item.get("x"), f"第 {index} 个点的 x", width)
        y = _coordinate(item.get("y"), f"第 {index} 个点的 y", height)
        label = item.get("label")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(f"第 {index} 个点的 label 必须为 0（排除）或 1（包含）。")
        normalized.append({"x": x, "y": y, "label": int(label)})
    if not normalized:
        return None, None, normalized
    point_coords = np.asarray([[item["x"], item["y"]] for item in normalized], dtype=np.float32)
    point_labels = np.asarray([item["label"] for item in normalized], dtype=np.int32)
    return point_coords, point_labels, normalized


def _parse_box(value: Any, width: int, height: int) -> tuple[np.ndarray | None, list[float] | None]:
    if value is None:
        return None, None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("box 必须是 [x0, y0, x1, y1]。")
    x0 = _coordinate(value[0], "box x0", width)
    y0 = _coordinate(value[1], "box y0", height)
    x1 = _coordinate(value[2], "box x1", width)
    y1 = _coordinate(value[3], "box y1", height)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError("框选范围太小，请拖出至少 1 个像素宽和高的框。")
    normalized = [x0, y0, x1, y1]
    return np.asarray(normalized, dtype=np.float32), normalized


def _parse_description(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("目标描述必须是文字。")
    description = value.strip()
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ValueError(f"目标描述不能超过 {MAX_DESCRIPTION_CHARS} 个字符。")
    return description or None


def _load_rgb(record: ImageRecord) -> np.ndarray:
    with Image.open(record.path) as opened:
        return np.array(opened.convert("RGB"), copy=True)


def _preview_image(image: Image.Image, maximum_edge: int) -> Image.Image:
    """Create a display-sized RGB copy without changing the source image."""
    preview = image.copy()
    preview.thumbnail((maximum_edge, maximum_edge), Image.Resampling.LANCZOS)
    return preview


def _drawing_contours(mask: np.ndarray) -> list[np.ndarray]:
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    found, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return found


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _render_overlay(image_rgb: np.ndarray, mask: np.ndarray, contours: list[np.ndarray] | None = None) -> np.ndarray:
    rendered = image_rgb.copy()
    fill = np.empty_like(rendered)
    fill[:, :] = (0, 210, 145)
    rendered[mask] = cv2.addWeighted(rendered[mask], 0.42, fill[mask], 0.58, 0)
    cv2.drawContours(rendered, contours if contours is not None else _drawing_contours(mask), -1, (255, 220, 0), 2, cv2.LINE_AA)
    return rendered


def _render_transparent_contours(mask: np.ndarray, contours: list[np.ndarray] | None = None) -> np.ndarray:
    height, width = mask.shape
    rendered = np.zeros((height, width, 4), dtype=np.uint8)
    cv2.drawContours(rendered, contours if contours is not None else _drawing_contours(mask), -1, (255, 220, 0, 255), 2, cv2.LINE_AA)
    return rendered


def _serialize_prompt(record: ImageRecord, payload: dict[str, Any]) -> dict[str, Any]:
    point_coords, point_labels, points = _parse_points(payload.get("points"), record.width, record.height)
    box, normalized_box = _parse_box(payload.get("box"), record.width, record.height)
    description = _parse_description(payload.get("description"))
    grounding = _require_grounding(payload.get("grounding_id"), record.image_id, description)
    candidate_index = _parse_grounding_candidate_index(payload.get("grounding_candidate_index"), grounding)
    if box is None and (point_labels is None or not bool(np.any(point_labels == 1))):
        raise ValueError("请至少添加一个包含点或一个框选；排除点只能用于细化。")
    del point_coords, point_labels
    grounding_metadata = None
    if grounding is not None:
        grounding_metadata = grounding.as_metadata(record.width, record.height)
        grounding_metadata["selected_candidate_index"] = candidate_index
    return {
        "points": points,
        "box": normalized_box,
        "description": description,
        "grounding": grounding_metadata,
    }


def _write_result(record: ImageRecord, job: JobRecord, image_rgb: np.ndarray, mask: np.ndarray, score: float, selected_index: int) -> dict[str, Any]:
    result_id = uuid.uuid4().hex
    final_dir = RESULTS_DIR / result_id
    temporary_dir = RESULTS_DIR / f".{result_id}.tmp"
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(temporary_dir / "mask.png", compress_level=3)
        drawing_contours = _drawing_contours(mask)
        overlay = Image.fromarray(_render_overlay(image_rgb, mask, drawing_contours), mode="RGB")
        _preview_image(overlay, RESULT_PREVIEW_MAX_EDGE).save(
            temporary_dir / "preview.jpg",
            format="JPEG",
            quality=PREVIEW_JPEG_QUALITY,
            optimize=True,
            progressive=True,
        )
        Image.fromarray(_render_transparent_contours(mask, drawing_contours), mode="RGBA").save(
            temporary_dir / "contours.png",
            compress_level=3,
        )
        result = {
            "schema_version": 4,
            "created_at": _utc_now(),
            "job_id": job.job_id,
            "image": {"id": record.image_id, "filename": record.original_name, "width": record.width, "height": record.height},
            "target_description": job.input_payload["description"],
            "grounding": job.input_payload.get("grounding"),
            "sam2": {
                "model": SAM2_MODEL_NAME,
                "variant": SAM2_VARIANT,
                "device": engine.device,
                "prompt": {"points": job.input_payload["points"], "box": job.input_payload["box"]},
                "selected_mask_index": selected_index,
                "estimated_iou": score,
            },
            "mask_area_px": int(mask.sum()),
            "mask_bbox_xyxy": _mask_bbox(mask),
            "contours": mask_to_contours(mask),
            "caveats": [
                "SAM2 uses only points and the box. Qwen, when enabled, proposes a box from the target description.",
                "estimated_iou is SAM2's internal ranking estimate, not a calibrated probability.",
                "Contours describe the selected visible image region, not a guaranteed semantic or physical boundary.",
            ],
        }
        (temporary_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _json_write(temporary_dir / "access.json", {
            "owner_id": record.owner_id,
            "job_id": job.job_id,
            "created_at": result["created_at"],
            "expires_at": record.expires_at,
        })
        temporary_dir.replace(final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return {
        "result_id": result_id,
        "estimated_iou": score,
        "mask_area_px": int(mask.sum()),
        "mask_url": f"/media/results/{result_id}/mask.png",
        "preview_url": f"/media/results/{result_id}/preview.jpg",
        # Kept for old clients that still look for overlay_url.
        "overlay_url": f"/media/results/{result_id}/preview.jpg",
        "contours_url": f"/media/results/{result_id}/contours.png",
        "json_url": f"/media/results/{result_id}/result.json",
    }


def _run_segment_job(job: JobRecord, update: Callable[..., None]) -> dict[str, Any]:
    record = _load_image_record(job.image_id)
    if record is None or record.owner_id != job.owner_id or not record.path.is_file():
        raise RuntimeError("原始图片已不可用，请重新上传后再试。")
    update(phase="loading_model", message=f"正在准备 {SAM2_MODEL_NAME} 模型…")
    image_rgb = _load_rgb(record)
    update(phase="encoding_image", message="SAM2 正在编码图片…")
    points = job.input_payload["points"]
    point_coords = np.asarray([[point["x"], point["y"]] for point in points], dtype=np.float32) if points else None
    point_labels = np.asarray([point["label"] for point in points], dtype=np.int32) if points else None
    raw_box = job.input_payload["box"]
    box = np.asarray(raw_box, dtype=np.float32) if raw_box is not None else None
    update(phase="predicting", message="SAM2 正在生成像素级轮廓…")
    mask, score, selected_index = engine.segment(record.image_id, image_rgb, point_coords, point_labels, box)
    update(phase="rendering", message="正在整理轮廓与下载文件…")
    return _write_result(record, job, image_rgb, mask, score, selected_index)


class JobManager:
    """One local inference worker, durable JSON job records and polling state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._queue: queue.Queue[str] = queue.Queue(maxsize=MAX_JOB_QUEUE)
        self._recover_previous_jobs()
        self._worker = threading.Thread(target=self._work, name="autosem-sam2-worker", daemon=True)
        self._worker.start()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def queue_capacity(self) -> int:
        return MAX_JOB_QUEUE

    def _save(self, job: JobRecord) -> None:
        _json_write(_job_path(job.job_id), job.as_storage())

    def _recover_previous_jobs(self) -> None:
        for path in JOBS_DIR.glob("*.json"):
            stored = _json_read(path)
            job = JobRecord.from_storage(stored) if stored else None
            if job is None or _safe_iso_before_now(job.expires_at):
                continue
            if job.status in {"queued", "running"}:
                job.status = "failed"
                job.phase = "interrupted"
                job.message = "本机服务在任务完成前重启了，请重新提交一次。"
                job.error = job.message
                job.completed_at = _utc_now()
                self._save(job)
            self._jobs[job.job_id] = job

    def submit(self, job: JobRecord) -> None:
        with self._lock:
            if self._queue.full():
                raise queue.Full
            self._jobs[job.job_id] = job
            self._save(job)
            self._queue.put_nowait(job.job_id)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        stored = _json_read(_job_path(job_id))
        job = JobRecord.from_storage(stored) if stored else None
        if job is not None:
            with self._lock:
                self._jobs[job_id] = job
        return job

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for field_name, value in changes.items():
                setattr(job, field_name, value)
            self._save(job)

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                job = self.get(job_id)
                if job is None:
                    continue
                self._update(job_id, status="running", phase="loading_model", message="正在准备本地 SAM2…", started_at=_utc_now(), error=None)
                fresh_job = self.get(job_id)
                if fresh_job is None:
                    continue
                result = _run_segment_job(fresh_job, lambda **changes: self._update(job_id, **changes))
                self._update(job_id, status="succeeded", phase="succeeded", message="轮廓已生成，可以下载结果。", result=result, completed_at=_utc_now())
            except Sam2ConfigurationError as error:
                message = str(error)
                self._update(job_id, status="failed", phase="failed", message=message, error=message, completed_at=_utc_now())
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    message = "内存不足。请换一张更小的图片，或降低 SAM2_MAX_IMAGE_EDGE。"
                else:
                    app.logger.exception("SAM2 runtime error in job %s", job_id)
                    message = "SAM2 未能完成本次推理，请检查本机运行日志后重试。"
                self._update(job_id, status="failed", phase="failed", message=message, error=message, completed_at=_utc_now())
            except Exception:
                app.logger.exception("Unexpected segmentation job failure: %s", job_id)
                message = "生成轮廓时出现未预期的问题，请重新提交一次。"
                self._update(job_id, status="failed", phase="failed", message=message, error=message, completed_at=_utc_now())
            finally:
                self._queue.task_done()


job_manager = JobManager()
_cleanup_lock = threading.Lock()
_last_cleanup_at = 0.0


def _cleanup_expired_data() -> None:
    """Best-effort cleanup of only UUID-named files inside the app data directory."""
    if DATA_TTL_HOURS == 0:
        return
    expired_images: set[str] = set()
    for metadata_path in IMAGE_META_DIR.glob("*.json"):
        image_id = metadata_path.stem
        record = ImageRecord.from_storage(_json_read(metadata_path) or {})
        if not IMAGE_ID_RE.fullmatch(image_id) or record is None or not _safe_iso_before_now(record.expires_at):
            continue
        expired_images.add(image_id)
        record.path.unlink(missing_ok=True)
        if record.preview_path is not None:
            record.preview_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        with records_lock:
            records.pop(image_id, None)
    for metadata_path in GROUNDING_DIR.glob("*.json"):
        record = GroundingRecord.from_storage(_json_read(metadata_path) or {})
        if record is not None and record.image_id in expired_images:
            metadata_path.unlink(missing_ok=True)
            with grounding_records_lock:
                grounding_records.pop(record.grounding_id, None)
    for metadata_path in JOBS_DIR.glob("*.json"):
        job = JobRecord.from_storage(_json_read(metadata_path) or {})
        if job is None or (job.image_id not in expired_images and not _safe_iso_before_now(job.expires_at)):
            continue
        metadata_path.unlink(missing_ok=True)
        with job_manager._lock:
            job_manager._jobs.pop(job.job_id, None)
        if job.result and isinstance(job.result.get("result_id"), str) and IMAGE_ID_RE.fullmatch(job.result["result_id"]):
            shutil.rmtree(RESULTS_DIR / job.result["result_id"], ignore_errors=True)


def _maybe_cleanup_expired_data() -> None:
    """Avoid a full data-directory scan on every upload."""
    global _last_cleanup_at
    if DATA_TTL_HOURS == 0:
        return
    now = time.monotonic()
    with _cleanup_lock:
        if now - _last_cleanup_at < CLEANUP_INTERVAL_SECONDS:
            return
        _last_cleanup_at = now
    _cleanup_expired_data()


@app.after_request
def _security_headers(response: Any) -> Any:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    if request.path.startswith("/api/"):
        response.cache_control.no_store = True
    return response


@app.errorhandler(RequestEntityTooLarge)
def _upload_too_large(_error: RequestEntityTooLarge) -> Any:
    return _json_error(f"图片超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 上限。", 413)


@app.get("/")
def home() -> str:
    return render_template("home.html")


@app.get("/workspace")
def workspace() -> str:
    return render_template("index.html")


@app.get("/guide")
def guide() -> str:
    return render_template("guide.html")


@app.get("/privacy")
def privacy() -> str:
    return render_template("privacy.html")


@app.get("/healthz")
def healthz() -> Any:
    return jsonify({"ok": True, "service": "autosem"})


@app.get("/readyz")
def readyz() -> Any:
    ready = engine.configuration_ready() and job_manager.queue_depth < job_manager.queue_capacity
    return jsonify({"ready": ready, "runtime": engine.public_status()}), 200 if ready else 503


@app.get("/api/runtime/status")
def runtime_status() -> Any:
    status = engine.public_status()
    status.update({"ready": engine.configuration_ready(), "queue_depth": job_manager.queue_depth, "queue_capacity": job_manager.queue_capacity, "data_ttl_hours": DATA_TTL_HOURS})
    return jsonify(status)


@app.post("/api/upload")
def upload() -> Any:
    owner_id = _current_owner()
    limited = _limit_or_error(owner_id, "upload", 12, 60 * 60)
    if limited is not None:
        return limited
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return _json_error("请选择一张图片。", 400)
    safe_name = secure_filename(uploaded.filename)
    extension = Path(safe_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return _json_error("支持 JPG、PNG 和 WebP 图片。", 400)

    _maybe_cleanup_expired_data()
    image_id = uuid.uuid4().hex
    storage_extension = ".png" if extension == ".png" else ".jpg"
    destination = UPLOAD_DIR / f"{image_id}{storage_extension}"
    preview_filename = f"{image_id}.jpg"
    preview_destination = IMAGE_PREVIEW_DIR / preview_filename
    try:
        with Image.open(uploaded.stream) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                return _json_error(f"图片像素过多，当前上限是 {MAX_IMAGE_PIXELS:,} 像素。", 400)
            normalized = ImageOps.exif_transpose(opened)
            rgb = normalized.convert("RGB")
            width, height = rgb.size
            if min(width, height) < 2:
                return _json_error("图片至少需要 2 个像素宽和高。", 400)
            if storage_extension == ".png":
                rgb.save(destination, format="PNG", compress_level=3)
            else:
                rgb.save(destination, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True, progressive=True)
            preview = _preview_image(rgb, DISPLAY_MAX_IMAGE_EDGE)
            preview_width, preview_height = preview.size
            preview.save(
                preview_destination,
                format="JPEG",
                quality=PREVIEW_JPEG_QUALITY,
                optimize=True,
                progressive=True,
            )
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        destination.unlink(missing_ok=True)
        preview_destination.unlink(missing_ok=True)
        return _json_error("这不是可读取的图片文件。", 400)

    record = ImageRecord(
        image_id=image_id,
        filename=destination.name,
        original_name=safe_name or "image",
        width=width,
        height=height,
        owner_id=owner_id,
        created_at=_utc_now(),
        expires_at=_expiry_timestamp(),
        preview_filename=preview_filename,
        preview_width=preview_width,
        preview_height=preview_height,
    )
    _json_write(_image_meta_path(image_id), record.as_storage())
    with records_lock:
        records[image_id] = record
    return jsonify({
        "image_id": image_id,
        "image_url": f"/media/images/{image_id}",
        "preview_url": f"/media/previews/{image_id}",
        "width": width,
        "height": height,
        "preview_width": preview_width,
        "preview_height": preview_height,
        "expires_at": record.expires_at,
    }), 201


def _send_owned_media(directory: Path, filename: str, *, as_attachment: bool = False) -> Any:
    response = send_from_directory(directory, filename, as_attachment=as_attachment, conditional=True)
    if as_attachment:
        response.cache_control.no_store = True
        return response
    response.cache_control.no_cache = False
    response.cache_control.private = True
    response.cache_control.public = False
    response.cache_control.max_age = min(max(DATA_TTL_HOURS, 1) * 3600, 24 * 3600)
    response.headers["Vary"] = "Cookie"
    return response


@app.get("/media/images/<image_id>")
def uploaded_image(image_id: str) -> Any:
    record = _require_record(image_id)
    return _send_owned_media(UPLOAD_DIR, record.filename)


@app.get("/media/previews/<image_id>")
def uploaded_preview(image_id: str) -> Any:
    record = _require_record(image_id)
    if record.preview_filename and record.preview_path and record.preview_path.is_file():
        return _send_owned_media(IMAGE_PREVIEW_DIR, record.preview_filename)
    # Existing uploads from before the optimization only have their source file.
    return _send_owned_media(UPLOAD_DIR, record.filename)


@app.get("/api/grounding/status")
def grounding_status() -> Any:
    return jsonify({"configured": grounder.configured, "provider": "Alibaba Cloud Model Studio", "model": grounder.model if grounder.configured else None})


@app.post("/api/ground")
def ground() -> Any:
    owner_id = _current_owner()
    limited = _limit_or_error(owner_id, "ground", 24, 24 * 60 * 60)
    if limited is not None:
        return limited
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("请提交 JSON 请求。", 400)
    try:
        record = _require_record(payload.get("image_id"))
        description = _parse_description(payload.get("description"))
        if description is None:
            raise ValueError("请先填写你想要定位的目标。")
    except ValueError as error:
        return _json_error(str(error), 400)
    if not grounder.configured:
        return _json_error("未配置 DASHSCOPE_API_KEY。手动点选和框选仍可使用。", 503)
    try:
        proposal = grounder.ground(_load_rgb(record), description)
    except GroundingError as error:
        app.logger.warning("Qwen grounding failed: %s", error)
        return _json_error(str(error), 502)
    except Exception:
        app.logger.exception("Unexpected Qwen grounding error")
        return _json_error("百炼自动定位失败。完整错误已写入本地终端。", 502)

    grounding_record = GroundingRecord(
        grounding_id=uuid.uuid4().hex,
        image_id=record.image_id,
        description=description,
        model=grounder.model,
        proposal=proposal,
        created_at=_utc_now(),
        owner_id=owner_id,
    )
    _json_write(_grounding_path(grounding_record.grounding_id), grounding_record.as_storage())
    with grounding_records_lock:
        grounding_records[grounding_record.grounding_id] = grounding_record
    public = grounding_record.as_metadata(record.width, record.height)
    return jsonify({"grounding_id": grounding_record.grounding_id, "status": public["status"], "note": public["note"], "model": public["model"], "candidate": public["candidate"], "candidates": public["candidates"]}), 200


def _create_segment_job() -> Any:
    owner_id = _current_owner()
    limited = _limit_or_error(owner_id, "segment", 36, 60 * 60)
    if limited is not None:
        return limited
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("请提交 JSON 请求。", 400)
    try:
        record = _require_record(payload.get("image_id"))
        stored_prompt = _serialize_prompt(record, payload)
    except ValueError as error:
        return _json_error(str(error), 400)
    job = JobRecord(
        job_id=uuid.uuid4().hex,
        image_id=record.image_id,
        owner_id=owner_id,
        input_payload=stored_prompt,
        created_at=_utc_now(),
        expires_at=record.expires_at,
        message="任务已排队，SAM2 将在后台运行。",
    )
    try:
        job_manager.submit(job)
    except queue.Full:
        return _json_error("本机任务队列已满，请等待一个任务完成后再试。", 503, 5)
    return jsonify({"job_id": job.job_id, "status": job.status, "poll_url": f"/api/jobs/{job.job_id}"}), 202


@app.post("/api/segment-jobs")
def segment_jobs() -> Any:
    return _create_segment_job()


@app.post("/api/segment")
def segment_legacy() -> Any:
    """Compatibility endpoint: segmentation is now intentionally asynchronous."""
    return _create_segment_job()


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str) -> Any:
    if not IMAGE_ID_RE.fullmatch(job_id):
        abort(404)
    job = job_manager.get(job_id)
    if job is None or job.owner_id != _current_owner() or _safe_iso_before_now(job.expires_at):
        abort(404)
    return jsonify(job.as_public(job_manager.queue_depth))


@app.get("/media/results/<result_id>/<artifact>")
def result_artifact(result_id: str, artifact: str) -> Any:
    if not IMAGE_ID_RE.fullmatch(result_id) or artifact not in RESULT_FILE_NAMES:
        abort(404)
    result_dir = RESULTS_DIR / result_id
    access = _json_read(result_dir / "access.json")
    if (
        not result_dir.is_dir()
        or access is None
        or access.get("owner_id") != _current_owner()
        or _safe_iso_before_now(access.get("expires_at") if isinstance(access.get("expires_at"), str) else None)
    ):
        abort(404)
    return _send_owned_media(result_dir, artifact, as_attachment=artifact == "result.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local AutoSEM website.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise SystemExit("--port 必须在 1024 到 65535 之间。")
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
