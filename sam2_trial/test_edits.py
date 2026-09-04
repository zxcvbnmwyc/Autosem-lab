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

    def _upload_image(
        self,
        client,
        color: tuple[int, int, int],
        *,
        size: tuple[int, int] = (12, 12),
        name: str = "background.png",
    ) -> dict:
        image = Image.new("RGB", size, color)
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        encoded.seek(0)
        response = client.post(
            "/api/upload",
            data={"image": (encoded, name)},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201)
        return response.get_json()

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
                "background": {"mode": "transparent", "color": "#ffffff", "blur_px": 0},
                "subject": {"brightness": 10, "saturation": 0, "blur_px": 0},
            },
        )
        self.assertEqual(response.status_code, 201)
        edited = response.get_json()
        self.assertIn("download_url", edited)
        self.assertEqual(edited["settings"]["background_mode"], "transparent")
        self.assertEqual(edited["settings"]["background_blur_px"], 0)
        artifact = self.client.get(edited["download_url"])
        self.assertEqual(artifact.status_code, 200)
        self.assertEqual(artifact.mimetype, "image/png")
        with Image.open(io.BytesIO(artifact.data)) as output:
            self.assertEqual(output.size, (40, 30))
            self.assertEqual(output.mode, "RGBA")
        artifact.close()

    def test_edit_endpoint_accepts_new_settings_and_crops_around_subject(self) -> None:
        uploaded, result = self._result()
        response = self.client.post(
            "/api/edits",
            json={
                "image_id": uploaded["image_id"],
                "result_id": result["result_id"],
                "selection": {"edge_offset": 1, "feather_px": 2, "cleanup": True},
                "background": {
                    "mode": "color",
                    "color": "#ffffff",
                    "blur_px": 0,
                    "brightness": 10,
                    "saturation": -20,
                    "grayscale": False,
                },
                "subject": {
                    "brightness": 10,
                    "saturation": 5,
                    "contrast": 6,
                    "hue_degrees": 15,
                    "temperature": 12,
                    "blur_px": 1,
                    "sharpen": 9,
                    "opacity": 70,
                },
                "effects": {
                    "outline_width_px": 2,
                    "outline_color": "#000000",
                    "outline_opacity": 100,
                    "shadow_offset_x": 0,
                    "shadow_offset_y": 0,
                    "shadow_blur_px": 0,
                    "shadow_color": "#222222",
                    "shadow_opacity": 0,
                },
                "crop": {"enabled": True, "padding_px": 0, "aspect_ratio": "1:1"},
            },
        )
        self.assertEqual(response.status_code, 201)
        edited = response.get_json()
        self.assertEqual(edited["settings"]["background_brightness"], 10)
        self.assertEqual(edited["settings"]["subject_contrast"], 6)
        self.assertEqual(edited["settings"]["subject_opacity"], 70)
        self.assertEqual(edited["settings"]["outline_width_px"], 2)
        self.assertEqual(edited["settings"]["shadow_offset_y"], 0)
        self.assertEqual(edited["settings"]["shadow_blur_px"], 0)
        self.assertTrue(edited["settings"]["crop_enabled"])
        self.assertEqual(edited["settings"]["crop_aspect_ratio"], "1:1")
        artifact = self.client.get(edited["download_url"])
        self.assertEqual(artifact.status_code, 200)
        with Image.open(io.BytesIO(artifact.data)) as output:
            self.assertEqual(output.mode, "RGB")
            self.assertEqual(output.size[0], output.size[1])
            self.assertLess(output.size[0], 40)
        artifact.close()

    def test_edit_endpoint_composites_an_owned_custom_background(self) -> None:
        uploaded, result = self._result()
        background = self._upload_image(self.client, (38, 170, 92))
        response = self.client.post(
            "/api/edits",
            json={
                "image_id": uploaded["image_id"],
                "result_id": result["result_id"],
                "background": {
                    "mode": "image",
                    "image_id": background["image_id"],
                    "brightness": 0,
                    "saturation": 0,
                    "grayscale": False,
                },
            },
        )
        self.assertEqual(response.status_code, 201)
        edited = response.get_json()
        self.assertEqual(edited["settings"]["background_mode"], "image")
        self.assertEqual(
            edited["settings"]["background_image_id"], background["image_id"]
        )
        artifact = self.client.get(edited["download_url"])
        self.assertEqual(artifact.status_code, 200)
        with Image.open(io.BytesIO(artifact.data)) as output:
            rgb = output.convert("RGB")
            self.assertEqual(rgb.size, (40, 30))
            self.assertEqual(rgb.getpixel((1, 1)), (38, 170, 92))
            self.assertEqual(rgb.getpixel((18, 15)), (30, 65, 100))
        artifact.close()

    def test_edit_rejects_custom_background_from_another_session(self) -> None:
        uploaded, result = self._result()
        other_client = application.app.test_client()
        background = self._upload_image(other_client, (38, 170, 92))
        response = self.client.post(
            "/api/edits",
            json={
                "image_id": uploaded["image_id"],
                "result_id": result["result_id"],
                "background": {
                    "mode": "image",
                    "image_id": background["image_id"],
                },
            },
        )
        self.assertEqual(response.status_code, 404)

    def test_edit_requires_an_uploaded_image_for_custom_background(self) -> None:
        uploaded, result = self._result()
        response = self.client.post(
            "/api/edits",
            json={
                "image_id": uploaded["image_id"],
                "result_id": result["result_id"],
                "background": {"mode": "image"},
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("请先上传一张背景图片", response.get_json()["error"])

    def test_edit_rejects_background_image_id_outside_image_mode(self) -> None:
        uploaded, result = self._result()
        background = self._upload_image(self.client, (38, 170, 92))
        response = self.client.post(
            "/api/edits",
            json={
                "image_id": uploaded["image_id"],
                "result_id": result["result_id"],
                "background": {
                    "mode": "color",
                    "image_id": background["image_id"],
                    "color": "#ffffff",
                },
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("只有自定义图片背景", response.get_json()["error"])

    def test_edit_rejects_a_custom_background_whose_file_disappeared(self) -> None:
        uploaded, result = self._result()
        background = self._upload_image(self.client, (38, 170, 92))
        record = application._load_image_record(background["image_id"])
        self.assertIsNotNone(record)
        record.path.unlink()
        response = self.client.post(
            "/api/edits",
            json={
                "image_id": uploaded["image_id"],
                "result_id": result["result_id"],
                "background": {
                    "mode": "image",
                    "image_id": background["image_id"],
                },
            },
        )
        self.assertEqual(response.status_code, 404)

    def test_edit_rejects_subject_opacity_with_original_background(self) -> None:
        uploaded, result = self._result()
        response = self.client.post(
            "/api/edits",
            json={
                "image_id": uploaded["image_id"],
                "result_id": result["result_id"],
                "background": {"mode": "original"},
                "subject": {"opacity": 70},
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("降低主体透明度时，请先选择透明、纯色、虚化或图片背景。", response.get_json()["error"])

    def test_tight_crop_keeps_the_visible_feather_tail(self) -> None:
        uploaded, result = self._result()
        response = self.client.post(
            "/api/edits",
            json={
                "image_id": uploaded["image_id"],
                "result_id": result["result_id"],
                "selection": {"edge_offset": 0, "feather_px": 6, "cleanup": True},
                "background": {"mode": "transparent"},
                "crop": {"enabled": True, "padding_px": 0, "aspect_ratio": "free"},
            },
        )
        self.assertEqual(response.status_code, 201)
        artifact = self.client.get(response.get_json()["download_url"])
        self.assertEqual(artifact.status_code, 200)
        with Image.open(io.BytesIO(artifact.data)) as output:
            alpha = np.asarray(output.getchannel("A"))
            border = np.concatenate((alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]))
            self.assertLess(int(border.max()), 32)
            self.assertGreater(int(alpha.max()), 240)
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
