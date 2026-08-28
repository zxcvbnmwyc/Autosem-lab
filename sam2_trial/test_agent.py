import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as application
from grounding import GroundingCandidate, GroundingProposal


class _RecordingEngine:
    device = "cpu"

    def __init__(self, score: float = 0.91) -> None:
        self.score = score
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
        self.calls.append(
            {
                "points": point_coords.tolist() if point_coords is not None else None,
                "labels": point_labels.tolist() if point_labels is not None else None,
                "box": box.tolist() if box is not None else None,
            }
        )
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        mask[2:-2, 2:-2] = True
        return mask, self.score, 0


class _FakeGrounder:
    model = "qwen3-vl-flash"

    def __init__(self, proposal: GroundingProposal | None, configured: bool = True) -> None:
        self.proposal = proposal
        self.configured = configured
        self.calls: list[str] = []

    def ground(self, _image_rgb, description: str) -> GroundingProposal:
        self.calls.append(description)
        if self.proposal is None:
            raise AssertionError("ground() should not be called when Qwen is disabled")
        return self.proposal


def _proposal(status: str = "found", confidence: float = 0.91, count: int = 1) -> GroundingProposal:
    candidates = tuple(
        GroundingCandidate(
            100.0 + index * 250.0,
            150.0,
            700.0 + index * 50.0,
            800.0,
            confidence - index * 0.05,
            f"object-{index + 1}",
        )
        for index in range(count)
    )
    return GroundingProposal(status, candidates, "test proposal")


class AgentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.paths = {
            "UPLOAD_DIR": root / "uploads",
            "IMAGE_PREVIEW_DIR": root / "previews",
            "IMAGE_META_DIR": root / "images",
            "GROUNDING_DIR": root / "groundings",
            "AGENT_RUN_DIR": root / "agent-runs",
            "JOBS_DIR": root / "jobs",
            "RESULTS_DIR": root / "results",
        }
        for directory in self.paths.values():
            directory.mkdir(parents=True)
        self.engine = _RecordingEngine()
        self.grounder = _FakeGrounder(_proposal())
        self.patches = [patch.object(application, name, value) for name, value in self.paths.items()]
        self.patches.extend(
            [
                patch.object(application, "engine", self.engine),
                patch.object(application, "grounder", self.grounder),
                patch.object(application, "_last_cleanup_at", 0.0),
            ]
        )
        for active_patch in self.patches:
            active_patch.start()
        with application.records_lock:
            application.records.clear()
        with application.grounding_records_lock:
            application.grounding_records.clear()
        with application.agent_runs_lock:
            application.agent_runs.clear()
        self.client = application.app.test_client()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temporary.cleanup()

    def _upload(self) -> dict:
        image = Image.new("RGB", (24, 18), "white")
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

    def _start(self) -> tuple[dict, dict]:
        uploaded = self._upload()
        response = self.client.post(
            "/api/agent-runs",
            json={"image_id": uploaded["image_id"], "description": "target object"},
        )
        self.assertEqual(response.status_code, 201)
        return uploaded, response.get_json()

    def _wait_for_job(self, job: dict) -> dict:
        finished = None
        for _ in range(100):
            response = self.client.get(job["poll_url"])
            self.assertEqual(response.status_code, 200)
            finished = response.get_json()
            if finished["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.02)
        self.assertIsNotNone(finished)
        self.assertEqual(finished["status"], "succeeded")
        return finished

    def test_single_found_candidate_waits_for_user_confirmation(self) -> None:
        _uploaded, run = self._start()
        self.assertEqual(run["phase"], "ready_to_segment")
        self.assertEqual(run["next_action"], "segment")
        self.assertEqual(run["selected_candidate_index"], 0)
        self.assertEqual(len(run["candidates"]), 1)
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.grounder.calls, ["target object"])

    def test_ambiguous_candidates_require_choice_then_use_server_box(self) -> None:
        self.grounder.proposal = _proposal(status="ambiguous", count=2)
        _uploaded, run = self._start()
        self.assertEqual(run["phase"], "needs_choice")
        self.assertIsNone(run["selected_candidate_index"])
        self.assertEqual(self.engine.calls, [])

        chosen = self.client.post(
            f"/api/agent-runs/{run['agent_id']}/choose",
            json={"candidate_index": 1},
        )
        self.assertEqual(chosen.status_code, 200)
        selected = chosen.get_json()
        self.assertEqual(selected["phase"], "ready_to_segment")
        self.assertEqual(selected["selected_candidate_index"], 1)

        started = self.client.post(
            f"/api/agent-runs/{run['agent_id']}/segment",
            json={"box": [0, 0, 2, 2]},
        )
        self.assertEqual(started.status_code, 202)
        self._wait_for_job(started.get_json()["job"])
        self.assertEqual(len(self.engine.calls), 1)
        self.assertGreater(self.engine.calls[0]["box"][0], 5.0)
        self.assertNotEqual(self.engine.calls[0]["box"], [0.0, 0.0, 2.0, 2.0])

    def test_not_found_and_disabled_qwen_fall_back_to_manual_prompt(self) -> None:
        self.grounder.proposal = GroundingProposal("not_found", (), "not present")
        uploaded, run = self._start()
        self.assertEqual(run["phase"], "needs_manual_prompt")
        self.assertEqual(run["next_action"], "add_manual_prompt")
        self.assertEqual(self.engine.calls, [])

        started = self.client.post(
            f"/api/agent-runs/{run['agent_id']}/segment",
            json={"points": [{"x": 8, "y": 8, "label": 1}], "box": None},
        )
        self.assertEqual(started.status_code, 202)
        self._wait_for_job(started.get_json()["job"])
        self.assertEqual(self.engine.calls[0]["points"], [[8.0, 8.0]])

        self.grounder.configured = False
        response = self.client.post(
            "/api/agent-runs",
            json={"image_id": uploaded["image_id"], "description": "another target"},
        )
        self.assertEqual(response.status_code, 201)
        disabled = response.get_json()
        self.assertEqual(disabled["phase"], "needs_manual_prompt")
        self.assertIsNone(disabled["grounding_id"])

    def test_evaluation_marks_normal_result_completed(self) -> None:
        _uploaded, run = self._start()
        started = self.client.post(
            f"/api/agent-runs/{run['agent_id']}/segment",
            json={},
        )
        self.assertEqual(started.status_code, 202)
        self._wait_for_job(started.get_json()["job"])

        evaluated = self.client.post(f"/api/agent-runs/{run['agent_id']}/evaluate", json={})
        self.assertEqual(evaluated.status_code, 200)
        result = evaluated.get_json()
        self.assertEqual(result["phase"], "completed")
        self.assertEqual(result["evaluation"]["verdict"], "pass")
        self.assertIn("area_ratio", result["evaluation"])


if __name__ == "__main__":
    unittest.main()
