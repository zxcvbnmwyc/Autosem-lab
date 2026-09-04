import io
import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as application
from grounding import (
    GroundingCandidate,
    GroundingProviderError,
    GroundingProposal,
    GroundingSchemaError,
    GroundingTransientError,
    OneClickEditPlan,
)


class _Engine:
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses: list[tuple[np.ndarray, float, int] | Exception] = []

    def public_status(self):
        return {
            "model": "sam2.1_hiera_tiny",
            "variant": "tiny",
            "device": self.device,
            "loaded": True,
            "configured": True,
            "max_image_edge": 1024,
        }

    def configuration_ready(self):
        return True

    def segment(self, _image_id, image_rgb, point_coords, point_labels, box):
        self.calls.append({"points": point_coords, "labels": point_labels, "box": box})
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        mask[2:-2, 2:-2] = True
        return mask, 0.91, 0


class _Grounder:
    configured = True
    model = "qwen3-vl-flash"

    def __init__(self, confidence: float = 0.91) -> None:
        self.calls: list[str] = []
        self.confidence = confidence
        self.proposal: GroundingProposal | None = None
        self.failures: list[Exception] = []

    def ground(self, _image_rgb, description: str) -> GroundingProposal:
        self.calls.append(description)
        if self.failures:
            raise self.failures.pop(0)
        if self.proposal is not None:
            return self.proposal
        return GroundingProposal(
            "found",
            (GroundingCandidate(100, 100, 900, 900, self.confidence, "cup"),),
            "found cup",
        )


class _Planner:
    configured = True
    model = "qwen3-vl-flash"

    def __init__(self, plan: OneClickEditPlan) -> None:
        self.plan_value = plan
        self.calls: list[str] = []

    def plan(self, _image_rgb, instruction: str) -> OneClickEditPlan:
        self.calls.append(instruction)
        return self.plan_value


def _plan(status: str = "ready", target: str | None = "the cup") -> OneClickEditPlan:
    summary = {
        "ready": "保留杯子，背景透明并轻微提亮。",
        "needs_input": "请说明要处理哪个杯子。",
        "unsupported": "当前不支持删除物体后补全背景。",
    }[status]
    reason_code = {
        "ready": "none",
        "needs_input": "missing_information",
        "unsupported": "unsupported_operation",
    }[status]
    return OneClickEditPlan(
        status=status,
        target=target,
        selection={"edge_offset": 0, "feather_px": 2, "cleanup": True},
        background={
            "mode": "transparent",
            "color": "#ffffff",
            "blur_px": 0,
            "brightness": 0,
            "saturation": 0,
            "grayscale": False,
        },
        subject={
            "brightness": 8,
            "saturation": 0,
            "contrast": 0,
            "hue_degrees": 0,
            "temperature": 0,
            "blur_px": 0,
            "sharpen": 0,
            "opacity": 100,
        },
        effects={
            "outline_width_px": 0,
            "outline_color": "#ffffff",
            "outline_opacity": 0,
            "shadow_offset_x": 0,
            "shadow_offset_y": 0,
            "shadow_blur_px": 0,
            "shadow_color": "#000000",
            "shadow_opacity": 0,
        },
        crop={"enabled": False, "padding_px": 24, "aspect_ratio": "free"},
        summary=summary,
        reason_code=reason_code,
    )


def _selection_only_plan(target: str) -> OneClickEditPlan:
    return OneClickEditPlan(
        status="ready",
        target=target,
        # Qwen may tune the mask even when the request has no visible edit.
        # Selection tuning must still stop at selection_ready.
        selection={"edge_offset": 0, "feather_px": 4, "cleanup": True},
        background={
            "mode": "original",
            "color": "#ffffff",
            "blur_px": 0,
            "brightness": 0,
            "saturation": 0,
            "grayscale": False,
        },
        subject={
            "brightness": 0,
            "saturation": 0,
            "contrast": 0,
            "hue_degrees": 0,
            "temperature": 0,
            "blur_px": 0,
            "sharpen": 0,
            "opacity": 100,
        },
        effects={
            "outline_width_px": 0,
            "outline_color": "#ffffff",
            "outline_opacity": 0,
            "shadow_offset_x": 0,
            "shadow_offset_y": 0,
            "shadow_blur_px": 0,
            "shadow_color": "#000000",
            "shadow_opacity": 0,
        },
        crop={"enabled": False, "padding_px": 24, "aspect_ratio": "free"},
        summary="已根据图像反推主体，本次只生成选区。",
        reason_code="selection_only",
    )


class OneClickEditApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.paths = {
            "UPLOAD_DIR": root / "uploads",
            "IMAGE_PREVIEW_DIR": root / "previews",
            "IMAGE_META_DIR": root / "images",
            "GROUNDING_DIR": root / "groundings",
            "AGENT_RUN_DIR": root / "agent-runs",
            "ONE_CLICK_RUN_DIR": root / "one-click-runs",
            "JOBS_DIR": root / "jobs",
            "RESULTS_DIR": root / "results",
            "METRICS_DIR": root / "metrics",
        }
        for directory in self.paths.values():
            directory.mkdir(parents=True)
        self.engine = _Engine()
        self.grounder = _Grounder()
        self.planner = _Planner(_plan())
        self.patches = [patch.object(application, name, value) for name, value in self.paths.items()]
        self.patches.extend(
            [
                patch.object(application, "METRICS_EVENTS_PATH", root / "metrics" / "events.jsonl"),
                patch.object(application, "DATA_TTL_HOURS", 0),
                patch.object(application, "engine", self.engine),
                patch.object(application, "grounder", self.grounder),
                patch.object(application, "edit_planner", self.planner),
                patch.object(application, "QWEN_REPRESENTATIVE_POINT_ENABLED", False),
            ]
        )
        for active_patch in self.patches:
            active_patch.start()
        with application.records_lock:
            application.records.clear()
        with application.grounding_records_lock:
            application.grounding_records.clear()
        with application.one_click_runs_lock:
            application.one_click_runs.clear()
        self.client = application.app.test_client()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temporary.cleanup()

    def _upload(self) -> dict:
        image = Image.new("RGB", (32, 24), (40, 80, 120))
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        encoded.seek(0)
        response = self.client.post(
            "/api/upload",
            data={"image": (encoded, "sample.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201)
        return response.get_json()

    def _wait_until_ready(self, run_id: str) -> dict:
        latest = None
        for _ in range(100):
            response = self.client.get(f"/api/one-click-runs/{run_id}")
            self.assertEqual(response.status_code, 200)
            latest = response.get_json()
            if latest["phase"] in {"ready_to_apply", "selection_ready", "needs_input", "failed"}:
                break
            time.sleep(0.02)
        self.assertIsNotNone(latest)
        return latest

    def _assert_quality_shape(self, quality: dict) -> None:
        self.assertEqual(
            set(quality),
            {
                "verdict",
                "area_ratio",
                "estimated_iou",
                "component_count",
                "largest_component_ratio",
                "border_sides",
                "prompt_box_containment",
                "positive_points_contained",
                "checks",
                "retryable_codes",
                "recommended_action",
                "retry_skipped_reason",
                "auto_retry",
            },
        )
        self.assertEqual(set(quality["auto_retry"]), {"attempted", "outcome", "trigger_codes"})

    def _wait_for_quality_retry(self, run_id: str) -> dict:
        latest = None
        for _ in range(100):
            response = self.client.get(f"/api/one-click-runs/{run_id}")
            self.assertEqual(response.status_code, 200)
            latest = response.get_json()
            if latest["quality_retry_count"] == 1:
                return latest
            time.sleep(0.02)
        self.fail(f"quality retry did not start: {latest}")

    def test_one_click_plan_segments_then_exports_full_size_png(self) -> None:
        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明并提亮"},
        )
        self.assertEqual(started.status_code, 202)
        run = started.get_json()
        self.assertEqual(run["phase"], "segmenting")
        self.assertEqual(self.planner.calls, ["保留这个杯子，背景透明并提亮"])
        self.assertEqual(self.grounder.calls, ["the cup"])

        ready = self._wait_until_ready(run["run_id"])
        self.assertEqual(ready["phase"], "ready_to_apply")
        self.assertEqual(ready["selected_candidate"]["label"], "cup")
        self.assertEqual(len(self.engine.calls), 1)

        completed = self.client.post(f"/api/one-click-runs/{run['run_id']}/apply", json={})
        self.assertEqual(completed.status_code, 201)
        public = completed.get_json()
        self.assertEqual(public["phase"], "completed")
        self.assertEqual(public["edit"]["settings"]["background_mode"], "transparent")
        artifact = self.client.get(public["edit"]["download_url"])
        self.assertEqual(artifact.status_code, 200)
        with Image.open(io.BytesIO(artifact.data)) as output:
            self.assertEqual(output.size, (32, 24))
            self.assertEqual(output.mode, "RGBA")
        artifact.close()

    def test_poor_mask_retries_once_with_qwen_interior_point(self) -> None:
        self.grounder.proposal = GroundingProposal(
            "found",
            (GroundingCandidate(100, 100, 900, 900, 0.95, "cup", 500, 500),),
            "found cup",
        )
        first_mask = np.ones((24, 32), dtype=bool)
        improved_mask = np.zeros((24, 32), dtype=bool)
        improved_mask[5:19, 8:24] = True
        self.engine.responses = [
            (first_mask, 0.42, 0),
            (improved_mask, 0.94, 0),
        ]

        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
        )
        self.assertEqual(started.status_code, 202)
        completed = self._wait_until_ready(started.get_json()["run_id"])

        self.assertEqual(completed["phase"], "ready_to_apply")
        self.assertEqual(completed["quality_retry_count"], 1)
        self.assertEqual(completed["selection_quality"]["verdict"], "pass")
        self._assert_quality_shape(completed["selection_quality"])
        self.assertIn("mask_too_large", completed["selection_quality"]["auto_retry"]["trigger_codes"])
        self.assertEqual(len(self.engine.calls), 2)
        self.assertIsNone(self.engine.calls[0]["points"])
        np.testing.assert_allclose(self.engine.calls[1]["points"], [[15.5, 11.5]])
        np.testing.assert_array_equal(self.engine.calls[1]["labels"], [1])

    def test_quality_retry_never_runs_more_than_once(self) -> None:
        self.grounder.proposal = GroundingProposal(
            "found",
            (GroundingCandidate(100, 100, 900, 900, 0.95, "cup", 500, 500),),
            "found cup",
        )
        failed_mask = np.ones((24, 32), dtype=bool)
        self.engine.responses = [
            (failed_mask, 0.10, 0),
            (failed_mask, 0.99, 0),
        ]

        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
        )
        completed = self._wait_until_ready(started.get_json()["run_id"])

        self.assertEqual(completed["phase"], "needs_input")
        self.assertEqual(completed["quality_retry_count"], 1)
        self.assertEqual(completed["selection_quality"]["verdict"], "needs_input")
        self.assertEqual(completed["selection_quality"]["recommended_action"], "manual_refine")
        self.assertEqual(completed["selection_quality"]["retry_skipped_reason"], "retry_result_worse")
        self.assertEqual(completed["selection_quality"]["auto_retry"]["outcome"], "kept_initial_result")
        # Equal mechanical quality keeps the first result even when retry IoU is higher.
        self.assertEqual(completed["job"]["result"]["estimated_iou"], 0.10)
        self._assert_quality_shape(completed["selection_quality"])
        self.assertIn("自动重试结果", completed["message"])
        self.assertEqual(len(self.engine.calls), 2)

    def test_geometry_improvement_beats_higher_uncalibrated_iou(self) -> None:
        self.grounder.proposal = GroundingProposal(
            "found",
            (GroundingCandidate(50, 50, 950, 950, 0.95, "cup", 500, 500),),
            "found cup",
        )
        full_mask = np.ones((24, 32), dtype=bool)
        fragmented = np.zeros((24, 32), dtype=bool)
        for y in (4, 11, 18):
            for x in (5, 15, 25):
                fragmented[y : y + 2, x : x + 2] = True
        self.engine.responses = [
            (full_mask, 0.99, 0),
            (fragmented, 0.10, 0),
        ]

        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
        )
        completed = self._wait_until_ready(started.get_json()["run_id"])

        self.assertEqual(completed["phase"], "needs_input")
        self.assertEqual(completed["selection_quality"]["auto_retry"]["outcome"], "kept_retry_result")
        self.assertEqual(completed["selection_quality"]["retry_skipped_reason"], "retry_limit_reached")
        self.assertIn("fragmented_mask", completed["selection_quality"]["retryable_codes"])
        self.assertEqual(completed["job"]["result"]["estimated_iou"], 0.10)

    def test_missing_initial_mask_requires_rerun_instead_of_manual_refinement(self) -> None:
        usable_mask = np.zeros((24, 32), dtype=bool)
        usable_mask[5:19, 8:24] = True
        self.engine.responses = [(usable_mask, 0.94, 0)]

        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
        )
        job_id = started.get_json()["job"]["job_id"]
        job = None
        for _ in range(100):
            job = application.job_manager.get(job_id)
            if job is not None and job.status == "succeeded":
                break
            time.sleep(0.02)
        self.assertIsNotNone(job)
        self.assertEqual(job.status, "succeeded")
        (self.paths["RESULTS_DIR"] / job.result["result_id"] / "mask.png").unlink()

        completed = self.client.get(f"/api/one-click-runs/{started.get_json()['run_id']}").get_json()

        self.assertEqual(completed["phase"], "failed")
        self.assertIsNone(completed["job"])
        self.assertIsNone(completed["result_id"])
        self.assertEqual(completed["selection_quality"]["verdict"], "failed")
        self.assertEqual(completed["selection_quality"]["recommended_action"], "rerun_segmentation")
        self.assertEqual(completed["selection_quality"]["retry_skipped_reason"], "initial_result_damaged")
        self.assertIn("重新执行", completed["message"])
        self.assertNotIn("手动微调", completed["message"])

    def test_fragmented_mask_is_detected_before_editing(self) -> None:
        self.grounder.proposal = GroundingProposal(
            "found",
            (GroundingCandidate(50, 50, 950, 950, 0.95, "cup", 500, 500),),
            "found cup",
        )
        fragmented = np.zeros((24, 32), dtype=bool)
        for y in (4, 11, 18):
            for x in (5, 15, 25):
                fragmented[y : y + 2, x : x + 2] = True
        improved_mask = np.zeros((24, 32), dtype=bool)
        improved_mask[5:19, 8:24] = True
        self.engine.responses = [
            (fragmented, 0.95, 0),
            (improved_mask, 0.94, 0),
        ]

        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
        )
        completed = self._wait_until_ready(started.get_json()["run_id"])

        self.assertEqual(completed["phase"], "ready_to_apply")
        self.assertEqual(len(self.engine.calls), 2)
        self.assertIn("fragmented_mask", completed["selection_quality"]["auto_retry"]["trigger_codes"])

    def test_poor_mask_without_validated_point_falls_back_safely(self) -> None:
        failed_mask = np.ones((24, 32), dtype=bool)
        self.engine.responses = [(failed_mask, 0.40, 0)]

        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
        )
        completed = self._wait_until_ready(started.get_json()["run_id"])

        self.assertEqual(completed["phase"], "needs_input")
        self.assertEqual(completed["quality_retry_count"], 0)
        self.assertEqual(completed["selection_quality"]["recommended_action"], "manual_refine")
        self.assertEqual(completed["selection_quality"]["retry_skipped_reason"], "qwen_point_unavailable")
        self.assertEqual(completed["selection_quality"]["auto_retry"]["outcome"], "skipped")
        self._assert_quality_shape(completed["selection_quality"])
        self.assertNotIn("已自动重试一次", completed["message"])
        self.assertEqual(len(self.engine.calls), 1)

    def test_low_sam2_score_alone_is_only_a_review_hint(self) -> None:
        usable_mask = np.zeros((24, 32), dtype=bool)
        usable_mask[5:19, 8:24] = True
        self.engine.responses = [(usable_mask, 0.42, 0)]

        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
        )
        completed = self._wait_until_ready(started.get_json()["run_id"])

        self.assertEqual(completed["phase"], "ready_to_apply")
        self.assertEqual(completed["quality_retry_count"], 0)
        self.assertEqual(completed["selection_quality"]["verdict"], "pass")
        self.assertEqual(completed["selection_quality"]["recommended_action"], "continue")
        self.assertIn("low_sam2_score", [item["code"] for item in completed["selection_quality"]["checks"]])
        self.assertEqual(len(self.engine.calls), 1)

    def test_missing_result_quality_uses_the_same_public_shape(self) -> None:
        record = application.ImageRecord(
            image_id="1" * 32,
            filename=f"{'1' * 32}.png",
            original_name="sample.png",
            width=32,
            height=24,
            owner_id="2" * 32,
            created_at="2026-09-04T00:00:00+00:00",
            expires_at=None,
        )
        job = application.JobRecord(
            job_id="3" * 32,
            image_id=record.image_id,
            owner_id=record.owner_id,
            input_payload={},
            created_at="2026-09-04T00:00:00+00:00",
            expires_at=None,
            status="succeeded",
            result=None,
        )

        quality = application._one_click_selection_quality(record, job)

        self._assert_quality_shape(quality)
        self.assertEqual(quality["verdict"], "needs_input")
        self.assertEqual(quality["recommended_action"], "manual_refine")
        self.assertEqual(quality["checks"][0]["code"], "missing_result")

    def test_existing_qwen_point_is_not_added_or_retried_twice(self) -> None:
        self.grounder.proposal = GroundingProposal(
            "found",
            (GroundingCandidate(100, 100, 900, 900, 0.95, "cup", 500, 500),),
            "found cup",
        )
        failed_mask = np.ones((24, 32), dtype=bool)
        self.engine.responses = [(failed_mask, 0.40, 0)]

        uploaded = self._upload()
        with patch.object(application, "QWEN_REPRESENTATIVE_POINT_ENABLED", True):
            started = self.client.post(
                "/api/one-click-runs",
                json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
            )
            completed = self._wait_until_ready(started.get_json()["run_id"])

        self.assertEqual(completed["phase"], "needs_input")
        self.assertEqual(completed["quality_retry_count"], 0)
        self.assertEqual(completed["selection_quality"]["recommended_action"], "manual_refine")
        self.assertEqual(completed["selection_quality"]["retry_skipped_reason"], "point_already_used")
        self.assertEqual(len(self.engine.calls), 1)
        np.testing.assert_allclose(self.engine.calls[0]["points"], [[15.5, 11.5]])

    def test_full_queue_skips_quality_retry_and_keeps_first_result(self) -> None:
        self.grounder.proposal = GroundingProposal(
            "found",
            (GroundingCandidate(100, 100, 900, 900, 0.95, "cup", 500, 500),),
            "found cup",
        )
        failed_mask = np.ones((24, 32), dtype=bool)
        self.engine.responses = [(failed_mask, 0.40, 0)]
        enqueue_segment_job = application._enqueue_segment_job

        def enqueue_unless_quality_retry(record, owner_id, prompt):
            if "quality_retry" in prompt:
                raise queue.Full
            return enqueue_segment_job(record, owner_id, prompt)

        uploaded = self._upload()
        with patch.object(application, "_enqueue_segment_job", side_effect=enqueue_unless_quality_retry):
            started = self.client.post(
                "/api/one-click-runs",
                json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
            )
            completed = self._wait_until_ready(started.get_json()["run_id"])

        self.assertEqual(completed["phase"], "needs_input")
        self.assertEqual(completed["quality_retry_count"], 0)
        self.assertEqual(completed["selection_quality"]["recommended_action"], "manual_refine")
        self.assertEqual(completed["selection_quality"]["retry_skipped_reason"], "queue_full")
        self.assertEqual(completed["job"]["status"], "succeeded")
        self.assertEqual(len(self.engine.calls), 1)

    def test_failed_quality_retry_restores_first_result_for_manual_refinement(self) -> None:
        self.grounder.proposal = GroundingProposal(
            "found",
            (GroundingCandidate(100, 100, 900, 900, 0.95, "cup", 500, 500),),
            "found cup",
        )
        failed_mask = np.ones((24, 32), dtype=bool)
        self.engine.responses = [(failed_mask, 0.40, 0), RuntimeError("retry failed")]

        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
        )
        completed = self._wait_until_ready(started.get_json()["run_id"])

        self.assertEqual(completed["phase"], "needs_input")
        self.assertEqual(completed["quality_retry_count"], 1)
        self.assertEqual(completed["selection_quality"]["retry_skipped_reason"], "retry_job_failed")
        self.assertEqual(completed["selection_quality"]["auto_retry"]["outcome"], "restored_initial_result")
        self.assertEqual(completed["job"]["status"], "succeeded")
        self.assertEqual(completed["job"]["result"]["estimated_iou"], 0.40)
        self.assertIn("已保留首次选区", completed["message"])
        self.assertEqual(len(self.engine.calls), 2)

    def test_damaged_retry_result_restores_first_result(self) -> None:
        self.grounder.proposal = GroundingProposal(
            "found",
            (GroundingCandidate(100, 100, 900, 900, 0.95, "cup", 500, 500),),
            "found cup",
        )
        failed_mask = np.ones((24, 32), dtype=bool)
        improved_mask = np.zeros((24, 32), dtype=bool)
        improved_mask[5:19, 8:24] = True
        self.engine.responses = [(failed_mask, 0.40, 0), (improved_mask, 0.94, 0)]

        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
        )
        retrying = self._wait_for_quality_retry(started.get_json()["run_id"])
        retry_job_id = retrying["job"]["job_id"]
        retry_job = None
        for _ in range(100):
            retry_job = application.job_manager.get(retry_job_id)
            if retry_job is not None and retry_job.status == "succeeded":
                break
            time.sleep(0.02)
        self.assertIsNotNone(retry_job)
        self.assertEqual(retry_job.status, "succeeded")
        retry_result_id = retry_job.result["result_id"]
        (self.paths["RESULTS_DIR"] / retry_result_id / "mask.png").unlink()

        completed = self.client.get(f"/api/one-click-runs/{started.get_json()['run_id']}").get_json()
        self.assertEqual(completed["phase"], "needs_input")
        self.assertEqual(completed["selection_quality"]["retry_skipped_reason"], "retry_result_damaged")
        self.assertEqual(completed["selection_quality"]["auto_retry"]["outcome"], "restored_initial_result")
        self.assertEqual(completed["job"]["status"], "succeeded")
        self.assertEqual(completed["job"]["result"]["estimated_iou"], 0.40)
        self.assertIn("已保留首次选区", completed["message"])

    def test_one_click_retries_schema_failure_once_then_continues(self) -> None:
        self.grounder.failures = [
            GroundingSchemaError("百炼返回的数据无法解析为候选框 JSON。")
        ]
        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={
                "image_id": uploaded["image_id"],
                "instruction": "保留这个杯子，背景透明并提亮",
            },
        )
        self.assertEqual(started.status_code, 202)
        run = started.get_json()
        self.assertEqual(run["phase"], "segmenting")
        self.assertEqual(self.grounder.calls, ["the cup", "the cup"])
        ready = self._wait_until_ready(run["run_id"])
        self.assertEqual(ready["phase"], "ready_to_apply")

    def test_one_click_reports_retry_exhaustion_cause(self) -> None:
        self.grounder.failures = [
            GroundingTransientError("Qwen 请求超时，请稍后重试。"),
            GroundingTransientError("Qwen 请求超时，请稍后重试。"),
        ]
        uploaded = self._upload()
        response = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子"},
        )
        self.assertEqual(response.status_code, 201)
        run = response.get_json()
        self.assertEqual(run["phase"], "failed")
        self.assertEqual(
            run["message"],
            "Qwen 定位服务暂时不可用，自动重试后仍未成功；可重试或手动编辑。",
        )
        self.assertEqual(self.grounder.calls, ["the cup", "the cup"])
        self.assertIsNone(run["job"])

    def test_one_click_not_found_is_not_retried(self) -> None:
        self.grounder.proposal = GroundingProposal("not_found", (), "target absent")
        uploaded = self._upload()
        response = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子"},
        )
        self.assertEqual(response.status_code, 201)
        run = response.get_json()
        self.assertEqual(run["phase"], "needs_input")
        self.assertIn("未能定位", run["message"])
        self.assertEqual(self.grounder.calls, ["the cup"])
        self.assertIsNone(run["job"])

    def test_manual_grounding_api_retries_transient_failure_once(self) -> None:
        self.grounder.failures = [
            GroundingTransientError("Qwen 请求超时，请稍后重试。")
        ]
        uploaded = self._upload()
        response = self.client.post(
            "/api/ground",
            json={"image_id": uploaded["image_id"], "description": "the cup"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "found")
        self.assertEqual(self.grounder.calls, ["the cup", "the cup"])

    def test_manual_grounding_api_does_not_retry_provider_configuration_error(self) -> None:
        self.grounder.failures = [GroundingProviderError("invalid configuration")]
        uploaded = self._upload()
        response = self.client.post(
            "/api/ground",
            json={"image_id": uploaded["image_id"], "description": "the cup"},
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.get_json()["error"],
            "Qwen 定位服务配置或权限异常；可改用手动编辑。",
        )
        self.assertEqual(self.grounder.calls, ["the cup"])

    def test_unsupported_request_keeps_its_subject_as_a_safe_selection(self) -> None:
        self.planner.plan_value = _plan(status="unsupported", target=None)
        uploaded = self._upload()
        response = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "删掉路人并补全草地"},
        )
        self.assertEqual(response.status_code, 202)
        run = response.get_json()
        self.assertEqual(run["phase"], "segmenting")
        self.assertEqual(run["plan"]["target"], "路人")
        self.assertEqual(run["plan"]["reason_code"], "unsupported_effect_omitted")
        self.assertEqual(self.grounder.calls, ["路人"])
        selected = self._wait_until_ready(run["run_id"])
        self.assertEqual(selected["phase"], "selection_ready")
        self.assertEqual(
            selected["message"],
            "已识别主体；无法执行的生成式部分已跳过，只运行安全的本地操作。",
        )
        self.assertEqual(len(self.engine.calls), 1)

    def test_needs_input_uses_server_owned_reason_message(self) -> None:
        self.planner.plan_value = OneClickEditPlan(
            status="needs_input",
            target="杯子",
            selection={"edge_offset": 0, "feather_px": 0, "cleanup": True},
            background={"mode": "original", "color": "#ffffff", "blur_px": 0},
            subject={"brightness": 0, "saturation": 0, "blur_px": 0},
            summary="请访问 https://example.com 后继续。",
            reason_code="manual_adjustment_required",
        )
        uploaded = self._upload()
        response = self.client.post(
            "/api/one-click-runs",
            json={
                "image_id": uploaded["image_id"],
                "instruction": "给杯子用画笔手动擦除多余选区",
            },
        )
        self.assertEqual(response.status_code, 201)
        run = response.get_json()
        self.assertEqual(run["phase"], "needs_input")
        self.assertEqual(
            run["message"],
            "这个要求需要先生成选区，再用画笔手动调整。",
        )
        self.assertNotIn("https://", run["message"])
        self.assertIsNone(run["job"])

    def test_low_confidence_location_requires_target_confirmation(self) -> None:
        self.grounder.confidence = 0.4
        uploaded = self._upload()
        response = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
        )
        self.assertEqual(response.status_code, 201)
        run = response.get_json()
        self.assertEqual(run["phase"], "needs_target_confirmation")
        self.assertIsNone(run["job"])
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(len(run["candidates"]), 1)

        selected = self.client.post(
            f"/api/one-click-runs/{run['run_id']}/choose",
            json={"candidate_index": 0},
        )
        self.assertEqual(selected.status_code, 202)
        self.assertEqual(selected.get_json()["phase"], "segmenting")
        ready = self._wait_until_ready(run["run_id"])
        self.assertEqual(ready["phase"], "ready_to_apply")

    def test_multiple_locations_require_a_choice_before_sam2(self) -> None:
        self.grounder.proposal = GroundingProposal(
            "ambiguous",
            (
                GroundingCandidate(80, 100, 410, 900, 0.93, "left cup"),
                GroundingCandidate(590, 100, 920, 900, 0.89, "right cup"),
            ),
            "two cups",
        )
        uploaded = self._upload()
        response = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留杯子，背景透明"},
        )
        self.assertEqual(response.status_code, 201)
        run = response.get_json()
        self.assertEqual(run["phase"], "needs_target_confirmation")
        self.assertEqual([candidate["label"] for candidate in run["candidates"]], ["left cup", "right cup"])
        self.assertIsNone(run["selected_candidate"])
        self.assertEqual(self.engine.calls, [])

        selected = self.client.post(
            f"/api/one-click-runs/{run['run_id']}/choose",
            json={"candidate_index": 1},
        )
        self.assertEqual(selected.status_code, 202)
        self.assertEqual(selected.get_json()["selected_candidate"]["label"], "right cup")
        ready = self._wait_until_ready(run["run_id"])
        self.assertEqual(ready["phase"], "ready_to_apply")

    def test_vague_demonstrative_uses_inferred_visual_target(self) -> None:
        self.planner.plan_value = _selection_only_plan("图中唯一的黄色奶酪")
        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "把这个弄出来"},
        )
        self.assertEqual(started.status_code, 202)
        self.assertEqual(started.get_json()["phase"], "segmenting")
        self.assertEqual(self.planner.calls, ["把这个弄出来"])
        self.assertEqual(self.grounder.calls, ["图中唯一的黄色奶酪"])

        selected = self._wait_until_ready(started.get_json()["run_id"])
        self.assertEqual(selected["phase"], "selection_ready")
        self.assertEqual(selected["plan"]["target"], "图中唯一的黄色奶酪")
        self.assertEqual(selected["plan"]["reason_code"], "selection_only")
        self.assertEqual(selected["plan"]["selection"]["feather_px"], 0)
        self.assertEqual(len(self.engine.calls), 1)

    def test_vague_demonstrative_has_a_server_fallback_when_qwen_omits_target(self) -> None:
        self.planner.plan_value = _plan(status="needs_input", target=None)
        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "把这个弄出来"},
        )
        self.assertEqual(started.status_code, 202)
        run = started.get_json()
        self.assertEqual(run["phase"], "segmenting")
        self.assertEqual(
            run["plan"]["target"], "用户指代的图中主要可见物体"
        )
        self.assertEqual(
            self.grounder.calls, ["用户指代的图中主要可见物体"]
        )
        selected = self._wait_until_ready(run["run_id"])
        self.assertEqual(selected["phase"], "selection_ready")

    def test_vague_demonstrative_with_multiple_candidates_requires_confirmation(self) -> None:
        self.planner.plan_value = _selection_only_plan("图中被指代的红色圆形")
        self.grounder.proposal = GroundingProposal(
            "ambiguous",
            (
                GroundingCandidate(90, 120, 410, 880, 0.92, "left red circle"),
                GroundingCandidate(590, 120, 910, 880, 0.90, "right red circle"),
            ),
            "two matching circles",
        )
        uploaded = self._upload()
        response = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "就把这个抠出来"},
        )
        self.assertEqual(response.status_code, 201)
        run = response.get_json()
        self.assertEqual(run["phase"], "needs_target_confirmation")
        self.assertEqual(run["plan"]["target"], "图中被指代的红色圆形")
        self.assertEqual(run["plan"]["background"]["mode"], "transparent")
        self.assertEqual(len(run["candidates"]), 2)
        self.assertIsNone(run["selected_candidate"])
        self.assertEqual(self.engine.calls, [])

    def test_vague_effect_without_any_subject_stays_at_needs_input(self) -> None:
        # Even if Qwen guesses an object and effects from the image, an
        # effect-only sentence does not authorize choosing that object.
        self.planner.plan_value = _plan(target="Qwen 猜测的杯子")
        uploaded = self._upload()
        response = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "弄得好看一点"},
        )
        self.assertEqual(response.status_code, 201)
        run = response.get_json()
        self.assertEqual(run["phase"], "needs_input")
        self.assertEqual(run["plan"]["reason_code"], "missing_subject")
        self.assertEqual(run["message"], "请说明要处理的具体主体。")
        self.assertEqual(self.grounder.calls, [])
        self.assertEqual(self.engine.calls, [])

    def test_one_click_run_is_private_to_its_browser_session(self) -> None:
        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，背景透明"},
        )
        self.assertEqual(started.status_code, 202)
        other_client = application.app.test_client()
        self.assertEqual(other_client.get(f"/api/one-click-runs/{started.get_json()['run_id']}").status_code, 404)
        self._wait_until_ready(started.get_json()["run_id"])
        self.assertEqual(other_client.post(f"/api/one-click-runs/{started.get_json()['run_id']}/apply", json={}).status_code, 404)

    def test_one_click_apply_preserves_new_effect_and_crop_settings(self) -> None:
        self.planner.plan_value = OneClickEditPlan(
            status="ready",
            target="the cup",
            selection={"edge_offset": -5, "feather_px": 2, "cleanup": True},
            background={
                "mode": "color",
                "color": "#ffffff",
                "blur_px": 0,
                "brightness": 8,
                "saturation": -10,
                "grayscale": False,
            },
            subject={
                "brightness": 10,
                "saturation": 4,
                "contrast": 6,
                "hue_degrees": 12,
                "temperature": 10,
                "blur_px": 1,
                "sharpen": 7,
                "opacity": 72,
            },
            effects={
                "outline_width_px": 2,
                "outline_color": "#000000",
                "outline_opacity": 100,
                "shadow_offset_x": 2,
                "shadow_offset_y": 3,
                "shadow_blur_px": 4,
                "shadow_color": "#222222",
                "shadow_opacity": 50,
            },
            crop={"enabled": True, "padding_px": 5, "aspect_ratio": "1:1"},
            summary="保留杯子，裁成正方形并加描边阴影。",
        )
        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子，裁成正方形并加描边阴影"},
        )
        self.assertEqual(started.status_code, 202)
        ready = self._wait_until_ready(started.get_json()["run_id"])
        self.assertEqual(ready["phase"], "ready_to_apply")

        completed = self.client.post(
            f"/api/one-click-runs/{started.get_json()['run_id']}/apply", json={}
        )
        self.assertEqual(completed.status_code, 201)
        public = completed.get_json()
        self.assertEqual(public["edit"]["settings"]["subject_opacity"], 72)
        self.assertEqual(public["edit"]["settings"]["outline_width_px"], 2)
        self.assertEqual(public["edit"]["settings"]["shadow_blur_px"], 4)
        self.assertTrue(public["edit"]["settings"]["crop_enabled"])
        self.assertEqual(public["edit"]["settings"]["crop_aspect_ratio"], "1:1")
        artifact = self.client.get(public["edit"]["download_url"])
        self.assertEqual(artifact.status_code, 200)
        with Image.open(io.BytesIO(artifact.data)) as output:
            self.assertEqual(output.mode, "RGB")
            self.assertEqual(output.size, (24, 24))
        artifact.close()

    def test_selection_only_plan_reaches_sam_and_stops_at_selection_ready(self) -> None:
        self.planner.plan_value = OneClickEditPlan(
            status="ready",
            target="the cup",
            selection={"edge_offset": 0, "feather_px": 0, "cleanup": True},
            background={
                "mode": "original",
                "color": "#ffffff",
                "blur_px": 0,
                "brightness": 0,
                "saturation": 0,
                "grayscale": False,
            },
            subject={
                "brightness": 0,
                "saturation": 0,
                "contrast": 0,
                "hue_degrees": 0,
                "temperature": 0,
                "blur_px": 0,
                "sharpen": 0,
                "opacity": 100,
            },
            effects={
                "outline_width_px": 0,
                "outline_color": "#ffffff",
                "outline_opacity": 0,
                "shadow_offset_x": 0,
                "shadow_offset_y": 0,
                "shadow_blur_px": 0,
                "shadow_color": "#000000",
                "shadow_opacity": 0,
            },
            crop={"enabled": False, "padding_px": 24, "aspect_ratio": "free"},
            summary="只先确定主体选区。",
            reason_code="selection_only",
        )
        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子"},
        )
        self.assertEqual(started.status_code, 202)
        run = started.get_json()
        self.assertEqual(run["phase"], "segmenting")
        self.assertEqual(self.grounder.calls, ["the cup"])

        selected = self._wait_until_ready(run["run_id"])
        self.assertEqual(selected["phase"], "selection_ready")
        self.assertEqual(selected["selected_candidate"]["label"], "cup")
        self.assertEqual(len(self.engine.calls), 1)

        blocked = self.client.post(f"/api/one-click-runs/{run['run_id']}/apply", json={})
        self.assertEqual(blocked.status_code, 409)

    def test_legacy_unsupported_plan_with_target_is_normalized_and_reaches_sam(self) -> None:
        self.planner.plan_value = OneClickEditPlan(
            status="unsupported",
            target="the cup",
            selection={"edge_offset": 0, "feather_px": 0, "cleanup": True},
            background={
                "mode": "original",
                "color": "#ffffff",
                "blur_px": 0,
                "brightness": 0,
                "saturation": 0,
                "grayscale": False,
            },
            subject={
                "brightness": 0,
                "saturation": 0,
                "contrast": 0,
                "hue_degrees": 0,
                "temperature": 0,
                "blur_px": 0,
                "sharpen": 0,
                "opacity": 100,
            },
            effects={
                "outline_width_px": 0,
                "outline_color": "#ffffff",
                "outline_opacity": 0,
                "shadow_offset_x": 0,
                "shadow_offset_y": 0,
                "shadow_blur_px": 0,
                "shadow_color": "#000000",
                "shadow_opacity": 0,
            },
            crop={"enabled": False, "padding_px": 24, "aspect_ratio": "free"},
            summary="删掉路人并补全不支持，但主体可先选出来。",
            reason_code="unsupported_operation",
        )
        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "删掉旁边的人，只保留这个杯子"},
        )
        self.assertEqual(started.status_code, 202)
        self.assertEqual(started.get_json()["phase"], "segmenting")
        selected = self._wait_until_ready(started.get_json()["run_id"])
        self.assertEqual(selected["phase"], "selection_ready")
        self.assertEqual(
            selected["message"],
            "已识别主体；无法执行的生成式部分已跳过，只运行安全的本地操作。",
        )


if __name__ == "__main__":
    unittest.main()
