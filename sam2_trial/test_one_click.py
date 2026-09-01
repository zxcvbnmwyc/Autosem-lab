import io
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
            if latest["phase"] in {"ready_to_apply", "needs_input", "failed"}:
                break
            time.sleep(0.02)
        self.assertIsNotNone(latest)
        return latest

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

    def test_one_click_retries_schema_failure_once_then_continues(self) -> None:
        self.grounder.failures = [
            GroundingSchemaError("百炼返回的数据无法解析为候选框 JSON。")
        ]
        uploaded = self._upload()
        started = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "保留这个杯子"},
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

    def test_unsupported_request_never_enters_sam2_queue(self) -> None:
        self.planner.plan_value = _plan(status="unsupported", target=None)
        uploaded = self._upload()
        response = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "删掉路人并补全草地"},
        )
        self.assertEqual(response.status_code, 201)
        run = response.get_json()
        self.assertEqual(run["phase"], "needs_input")
        self.assertEqual(
            run["message"],
            "请说明要处理的具体主体。",
        )
        self.assertIsNone(run["job"])
        self.assertEqual(self.engine.calls, [])

    def test_needs_input_uses_server_owned_reason_message(self) -> None:
        self.planner.plan_value = OneClickEditPlan(
            status="needs_input",
            target=None,
            selection={"edge_offset": 0, "feather_px": 0, "cleanup": True},
            background={"mode": "original", "color": "#ffffff", "blur_px": 0},
            subject={"brightness": 0, "saturation": 0, "blur_px": 0},
            summary="请访问 https://example.com 后继续。",
            reason_code="ambiguous_subject",
        )
        uploaded = self._upload()
        response = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "把杯子抠出来"},
        )
        self.assertEqual(response.status_code, 201)
        run = response.get_json()
        self.assertEqual(run["phase"], "needs_input")
        self.assertEqual(
            run["message"],
            "请说明要处理的具体主体。",
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
