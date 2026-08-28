import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as application


class _FakeEngine:
    device = "cpu"

    def public_status(self):
        return {
            "model": "sam2.1_hiera_tiny",
            "variant": "tiny",
            "device": self.device,
            "loaded": True,
            "configured": True,
            "max_image_edge": 1280,
        }

    def configuration_ready(self):
        return True

    def segment(self, _image_id, image_rgb, _point_coords, _point_labels, _box):
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        mask[2:-2, 2:-2] = True
        return mask, 0.91, 0


class JobApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.paths = {
            "UPLOAD_DIR": root / "uploads",
            "IMAGE_PREVIEW_DIR": root / "previews",
            "IMAGE_META_DIR": root / "images",
            "GROUNDING_DIR": root / "groundings",
            "JOBS_DIR": root / "jobs",
            "RESULTS_DIR": root / "results",
        }
        for directory in self.paths.values():
            directory.mkdir(parents=True)
        self.patches = [patch.object(application, name, value) for name, value in self.paths.items()]
        self.patches.append(patch.object(application, "engine", _FakeEngine()))
        self.patches.append(patch.object(application, "_last_cleanup_at", 0.0))
        for active_patch in self.patches:
            active_patch.start()
        with application.records_lock:
            application.records.clear()
        with application.grounding_records_lock:
            application.grounding_records.clear()
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

    def test_segment_job_returns_immediately_then_persists_result(self) -> None:
        uploaded = self._upload()
        response = self.client.post(
            "/api/segment-jobs",
            json={
                "image_id": uploaded["image_id"],
                "description": "white area",
                "points": [{"x": 8, "y": 8, "label": 1}],
                "box": None,
            },
        )
        self.assertEqual(response.status_code, 202)
        created = response.get_json()
        self.assertEqual(created["status"], "queued")

        finished = None
        for _ in range(80):
            status_response = self.client.get(created["poll_url"])
            self.assertEqual(status_response.status_code, 200)
            finished = status_response.get_json()
            if finished["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.025)
        self.assertIsNotNone(finished)
        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual(finished["phase"], "succeeded")
        self.assertIn("result", finished)
        self.assertIn("preview_url", finished["result"])
        artifact = self.client.get(finished["result"]["preview_url"])
        self.assertEqual(artifact.status_code, 200)
        self.assertEqual(artifact.mimetype, "image/jpeg")
        artifact.close()
        self.assertTrue((self.paths["JOBS_DIR"] / f"{created['job_id']}.json").is_file())

    def test_upload_returns_a_private_display_preview(self) -> None:
        uploaded = self._upload()
        self.assertIn("preview_url", uploaded)
        self.assertLessEqual(uploaded["preview_width"], application.DISPLAY_MAX_IMAGE_EDGE)
        self.assertLessEqual(uploaded["preview_height"], application.DISPLAY_MAX_IMAGE_EDGE)
        response = self.client.get(uploaded["preview_url"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertIn("private", response.headers.get("Cache-Control", ""))
        self.assertIn("Cookie", response.headers.get("Vary", ""))
        with Image.open(io.BytesIO(response.data)) as preview:
            self.assertEqual(preview.size, (24, 18))
        response.close()

    def test_runtime_reports_cpu_tiny_and_queue_information(self) -> None:
        response = self.client.get("/api/runtime/status")
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["variant"], "tiny")
        self.assertIn("queue_depth", result)


if __name__ == "__main__":
    unittest.main()
