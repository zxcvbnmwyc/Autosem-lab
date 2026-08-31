import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as application


class EditApiTests(unittest.TestCase):
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
            "METRICS_DIR": root / "metrics",
        }
        for directory in self.paths.values():
            directory.mkdir(parents=True)
        self.patches = [patch.object(application, name, value) for name, value in self.paths.items()]
        self.patches.extend(
            [
                patch.object(application, "METRICS_EVENTS_PATH", root / "metrics" / "events.jsonl"),
                patch.object(application, "DATA_TTL_HOURS", 0),
            ]
        )
        for active_patch in self.patches:
            active_patch.start()
        with application.records_lock:
            application.records.clear()
        self.client = application.app.test_client()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temporary.cleanup()

    def _result(self) -> tuple[dict, dict]:
        image = Image.new("RGB", (40, 30), (30, 65, 100))
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        encoded.seek(0)
        uploaded_response = self.client.post(
            "/api/upload",
            data={"image": (encoded, "sample.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(uploaded_response.status_code, 201)
        uploaded = uploaded_response.get_json()
        record = application._load_image_record(uploaded["image_id"])
        self.assertIsNotNone(record)
        with self.client.session_transaction() as session:
            owner_id = session["autosem_owner_id"]
        mask = np.zeros((record.height, record.width), dtype=bool)
        mask[6:24, 10:30] = True
        job = application.JobRecord(
            job_id="a" * 32,
            image_id=record.image_id,
            owner_id=owner_id,
            input_payload={"description": "sample", "points": [], "box": None},
            created_at="2026-08-31T00:00:00+00:00",
            expires_at=None,
        )
        result = application._write_result(record, job, application._load_rgb(record), mask, 0.91, 0)
        return uploaded, result

    def test_edit_endpoint_outputs_full_size_transparent_png(self) -> None:
        uploaded, result = self._result()
        response = self.client.post(
            "/api/edits",
            json={
                "image_id": uploaded["image_id"],
                "result_id": result["result_id"],
                "selection": {
                    "strokes": [{"mode": "add", "radius": 3, "points": [{"x": 5, "y": 5}]}],
                    "edge_offset": 1,
                    "feather_px": 2,
                    "cleanup": True,
                },
                "background": {"mode": "transparent", "color": "#ffffff", "blur_px": 18},
                "subject": {"brightness": 10, "saturation": 0, "blur_px": 0},
            },
        )
        self.assertEqual(response.status_code, 201)
        edited = response.get_json()
        self.assertIn("download_url", edited)
        self.assertEqual(edited["settings"]["background_mode"], "transparent")
        artifact = self.client.get(edited["download_url"])
        self.assertEqual(artifact.status_code, 200)
        self.assertEqual(artifact.mimetype, "image/png")
        with Image.open(io.BytesIO(artifact.data)) as output:
            self.assertEqual(output.size, (40, 30))
            self.assertEqual(output.mode, "RGBA")
        artifact.close()

    def test_edit_rejects_result_from_another_browser_session(self) -> None:
        uploaded, result = self._result()
        other_client = application.app.test_client()
        response = other_client.post(
            "/api/edits",
            json={"image_id": uploaded["image_id"], "result_id": result["result_id"]},
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
