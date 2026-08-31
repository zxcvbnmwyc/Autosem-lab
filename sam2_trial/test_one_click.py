import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as application
from grounding import GroundingCandidate, GroundingProposal, OneClickEditPlan


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

    def ground(self, _image_rgb, description: str) -> GroundingProposal:
        self.calls.append(description)
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
    return OneClickEditPlan(
        status=status,
        target=target,
        selection={"edge_offset": 0, "feather_px": 2, "cleanup": True},
        background={"mode": "transparent", "color": "#ffffff", "blur_px": 18},
        subject={"brightness": 8, "saturation": 0, "blur_px": 0},
        summary="保留杯子，背景透明并轻微提亮。",
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

    def test_unsupported_request_never_enters_sam2_queue(self) -> None:
        self.planner.plan_value = _plan(status="unsupported", target=None)
        uploaded = self._upload()
        response = self.client.post(
            "/api/one-click-runs",
            json={"image_id": uploaded["image_id"], "instruction": "删掉路人并补全草地"},
        )
        self.assertEqual(response.status_code, 201)
        run = response.get_json()
        self.assertEqual(run["phase"], "unsupported")
        self.assertIsNone(run["job"])
        self.assertEqual(self.engine.calls, [])

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


if __name__ == "__main__":
    unittest.main()
