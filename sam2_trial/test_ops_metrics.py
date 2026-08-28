"""Contract tests for the private AutoSEM operations metrics endpoint.

These tests intentionally seed JSONL rather than depending on a real Qwen call.
They define the durable event format and the dashboard aggregation contract while
keeping all data inside a temporary directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image

import app as application


class OpsMetricsApiTests(unittest.TestCase):
    """The dashboard is private, durable and reports only aggregate usage data."""

    OPS_TOKEN = "test-ops-dashboard-token"

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
            directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.paths["METRICS_DIR"] / "events.jsonl"

        # The production module exposes these paths and token deliberately so
        # deployments and tests can isolate durable operational data.
        self.patches = [
            *(patch.object(application, name, value) for name, value in self.paths.items()),
            patch.object(application, "METRICS_EVENTS_PATH", self.events_path),
            patch.object(application, "OPS_DASHBOARD_TOKEN", self.OPS_TOKEN),
            patch.object(application, "_last_cleanup_at", 0.0),
            patch.dict(os.environ, {"OPS_DASHBOARD_TOKEN": self.OPS_TOKEN}, clear=False),
            patch.dict(application.app.config, {"TESTING": True}),
        ]
        for active_patch in self.patches:
            active_patch.start()
        self._clear_metric_memory()
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

    @staticmethod
    def _session_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _clear_metric_memory(self) -> None:
        """Keep tests independent if the implementation later adds a cache."""
        for name in ("metrics_events", "_metrics_events", "metrics_cache", "_metrics_cache"):
            value = getattr(application, name, None)
            clear = getattr(value, "clear", None)
            if callable(clear):
                clear()

    def _event(
        self,
        kind: str,
        session_name: str,
        *,
        timestamp: datetime,
        duration_ms: float = 0,
        queue_wait_ms: float = 0,
        status: str = "success",
    ) -> dict[str, Any]:
        return {
            "timestamp": timestamp.isoformat(),
            "kind": kind,
            "session_hash": self._session_hash(session_name),
            "duration_ms": duration_ms,
            "queue_wait_ms": queue_wait_ms,
            "status": status,
        }

    def _seed_events(self, *events: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(event, ensure_ascii=False) for event in events]
        self.events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._clear_metric_memory()

    def _ops_get(self, window_hours: str | int = 24, token: str | None = OPS_TOKEN):
        headers = {"X-Ops-Token": token} if token is not None else {}
        return self.client.get(f"/api/ops/metrics?window_hours={window_hours}", headers=headers)

    def _upload(self) -> dict[str, Any]:
        image = Image.new("RGB", (24, 18), "white")
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        encoded.seek(0)
        response = self.client.post(
            "/api/upload",
            data={"image": (encoded, "metrics-sample.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201)
        return response.get_json()

    def test_metrics_endpoint_rejects_missing_or_invalid_token(self) -> None:
        self.assertEqual(self._ops_get(token=None).status_code, 401)
        self.assertEqual(self._ops_get(token="not-the-token").status_code, 401)

        allowed = self._ops_get()
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.get_json()["window_hours"], 24)

    def test_metrics_endpoint_is_unavailable_without_configuration(self) -> None:
        with (
            patch.object(application, "OPS_DASHBOARD_TOKEN", ""),
            patch.dict(os.environ, {"OPS_DASHBOARD_TOKEN": ""}, clear=False),
        ):
            response = self._ops_get()
        self.assertEqual(response.status_code, 503)

    def test_metrics_endpoint_validates_window_hours(self) -> None:
        for window_hours in ("23", "721", "not-a-number"):
            with self.subTest(window_hours=window_hours):
                self.assertEqual(self._ops_get(window_hours).status_code, 400)

    def test_metrics_aggregates_jsonl_events_inside_requested_window(self) -> None:
        now = datetime.now(UTC)
        self._seed_events(
            self._event("page_view", "browser-a", timestamp=now - timedelta(minutes=10), duration_ms=8),
            self._event("page_view", "browser-a", timestamp=now - timedelta(minutes=9), duration_ms=7),
            self._event("upload", "browser-a", timestamp=now - timedelta(minutes=8), duration_ms=130),
            self._event("grounding", "browser-a", timestamp=now - timedelta(minutes=7), duration_ms=1200),
            self._event(
                "segment_job",
                "browser-a",
                timestamp=now - timedelta(minutes=6),
                duration_ms=4200,
                queue_wait_ms=80,
            ),
            # Failed inference still consumed capacity, so it must be included
            # in timing and queue statistics while lowering success rate.
            self._event(
                "segment_job",
                "browser-b",
                timestamp=now - timedelta(minutes=5),
                duration_ms=5200,
                queue_wait_ms=400,
                status="failed",
            ),
            self._event("agent_run", "browser-b", timestamp=now - timedelta(minutes=4), duration_ms=1400),
            self._event("page_view", "browser-b", timestamp=now - timedelta(minutes=3), duration_ms=6),
            # This event is out of the requested 24-hour window and must not
            # count as a visitor or a page view.
            self._event("page_view", "old-browser", timestamp=now - timedelta(hours=25), duration_ms=12),
        )
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write("this is not valid json\n")
        self._clear_metric_memory()

        response = self._ops_get(24)
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()

        self.assertIn("generated_at", payload)
        self.assertEqual(payload["window_hours"], 24)
        summary = payload["summary"]
        self.assertEqual(
            summary,
            {
                "active_sessions": 2,
                "page_views": 3,
                "uploads": 1,
                "grounding_requests": 1,
                "segment_jobs": 2,
                "segment_succeeded": 1,
                "segment_success_rate": 50.0,
                "agent_runs": 1,
            },
        )

        timings = payload["timings"]
        self._assert_timing(timings["upload_ms"], count=1, average=130, median=130, minimum=130, maximum=130)
        self._assert_timing(timings["grounding_ms"], count=1, average=1200, median=1200, minimum=1200, maximum=1200)
        self._assert_timing(timings["sam2_ms"], count=2, average=4700, median=4700, minimum=4200, maximum=5200)
        self._assert_timing(timings["queue_wait_ms"], count=2, average=240, median=240, minimum=80, maximum=400)
        self.assertGreaterEqual(timings["sam2_ms"]["p90_ms"], timings["sam2_ms"]["median_ms"])
        self.assertLessEqual(timings["sam2_ms"]["p90_ms"], timings["sam2_ms"]["max_ms"])

        self.assertEqual(set(payload["queue"]), {"depth", "capacity"})
        self.assertIsInstance(payload["recent_events"], list)
        self.assertTrue(payload["recent_events"])

    def _assert_timing(
        self,
        value: dict[str, Any],
        *,
        count: int,
        average: float,
        median: float,
        minimum: float,
        maximum: float,
    ) -> None:
        self.assertEqual(value["count"], count)
        self.assertEqual(value["average_ms"], average)
        self.assertEqual(value["median_ms"], median)
        self.assertEqual(value["min_ms"], minimum)
        self.assertEqual(value["max_ms"], maximum)
        self.assertIn("p90_ms", value)

    def test_workspace_and_upload_write_safe_durable_events(self) -> None:
        workspace = self.client.get("/workspace")
        self.assertEqual(workspace.status_code, 200)
        self._upload()

        self.assertTrue(self.events_path.is_file())
        events = [json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertTrue(any(event["kind"] == "page_view" for event in events))
        self.assertTrue(any(event["kind"] == "upload" for event in events))

        with self.client.session_transaction() as browser_session:
            owner_id = browser_session["autosem_owner_id"]
        for event in events:
            with self.subTest(event=event):
                self.assertEqual(
                    set(event),
                    {"timestamp", "kind", "session_hash", "duration_ms", "queue_wait_ms", "status"},
                )
                datetime.fromisoformat(event["timestamp"])
                self.assertIn(event["kind"], {"page_view", "upload", "grounding", "segment_job", "agent_run"})
                self.assertIsInstance(event["session_hash"], str)
                # The server may truncate an HMAC for storage efficiency; it
                # still must be a nontrivial opaque identifier, never the raw
                # browser owner id.
                self.assertGreaterEqual(len(event["session_hash"]), 16)
                self.assertNotEqual(event["session_hash"], owner_id)
                self.assertIsInstance(event["duration_ms"], (int, float))
                self.assertGreaterEqual(event["duration_ms"], 0)
                self.assertIsInstance(event["queue_wait_ms"], (int, float))
                self.assertGreaterEqual(event["queue_wait_ms"], 0)
                self.assertIsInstance(event["status"], str)


if __name__ == "__main__":
    unittest.main()
