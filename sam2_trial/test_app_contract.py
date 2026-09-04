import unittest
from pathlib import Path

from app import GroundingRecord, _parse_grounding_candidate_index
from grounding import GroundingCandidate, GroundingProposal


def _record() -> GroundingRecord:
    proposal = GroundingProposal(
        status="ambiguous",
        candidates=(
            GroundingCandidate(100, 100, 300, 300, 0.8, "first"),
            GroundingCandidate(500, 500, 800, 800, 0.7, "second"),
        ),
        note=None,
    )
    return GroundingRecord(
        grounding_id="a" * 32,
        image_id="b" * 32,
        description="two objects",
        model="qwen3-vl-flash",
        proposal=proposal,
        created_at="2026-08-28T00:00:00+00:00",
    )


class AppContractTests(unittest.TestCase):
    def _workspace_assets(self) -> tuple[str, str]:
        project_root = Path(__file__).resolve().parent
        return (
            (project_root / "templates" / "index.html").read_text(encoding="utf-8"),
            (project_root / "static" / "app.js").read_text(encoding="utf-8"),
        )

    def test_edited_preview_replaces_the_central_canvas(self) -> None:
        workspace, script = self._workspace_assets()

        canvas_panel = workspace.index('<section class="panel canvas-panel"')
        job_panel = workspace.index('<aside class="panel job-panel"')
        central_result = workspace.index('id="edit-result"')
        self.assertLess(canvas_panel, central_result)
        self.assertLess(central_result, job_panel)
        self.assertNotIn('id="edit-preview"', workspace)
        self.assertNotIn("editPreview.src", script)
        self.assertIn("await displayEditOnCanvas(result.preview_url);", script)
        self.assertIn('canvasShell.scrollIntoView({ behavior: "smooth", block: "center" });', script)

    def test_workspace_exposes_complete_mask_brush_history(self) -> None:
        workspace, script = self._workspace_assets()

        for element_id in (
            "mask-add-button",
            "mask-erase-button",
            "undo-mask-stroke",
            "redo-mask-stroke",
            "undo-edit",
            "redo-edit",
        ):
            self.assertIn(f'id="{element_id}"', workspace)
        self.assertIn("maskStrokeRedo", script)
        self.assertIn("function redoMaskStroke()", script)
        self.assertIn("function undoEditorChange()", script)
        self.assertIn("function redoEditorChange()", script)
        self.assertIn("captureEditorSnapshot", script)

    def test_mask_visibility_is_a_view_only_undoable_edit(self) -> None:
        _workspace, script = self._workspace_assets()

        self.assertIn("function editorSnapshotAffectsOutput(first, second)", script)
        self.assertIn("maskOverlayVisible: state.maskOverlayVisible", script)
        self.assertIn("const previous = captureEditorSnapshot();\n      state.maskOverlayVisible = !state.maskOverlayVisible;", script)
        self.assertIn("commitEditorMutation(previous, { invalidatePreview: false });", script)
        self.assertIn("if (outputChanged) invalidateEditPreview();", script)

    def test_custom_background_uses_an_owned_uploaded_image(self) -> None:
        workspace, script = self._workspace_assets()

        self.assertIn('<option value="image">上传背景图片</option>', workspace)
        self.assertIn('id="background-image-input"', workspace)
        self.assertIn('id="background-image-preview"', workspace)
        self.assertIn('fetch("/api/upload", { method: "POST", body: form })', script)
        self.assertIn('mode: selectedBackgroundMode', script)
        self.assertIn('image_id: selectedBackgroundMode === "image" ? state.backgroundImageId : undefined', script)
        self.assertIn('payload.background.mode === "image" && !payload.background.image_id', script)

    def test_legacy_qwen_request_defaults_to_first_candidate(self) -> None:
        self.assertEqual(_parse_grounding_candidate_index(None, _record()), 0)

    def test_selected_qwen_candidate_is_retained(self) -> None:
        self.assertEqual(_parse_grounding_candidate_index(1, _record()), 1)

    def test_out_of_range_qwen_candidate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _parse_grounding_candidate_index(2, _record())

    def test_candidate_index_without_grounding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _parse_grounding_candidate_index(0, None)


if __name__ == "__main__":
    unittest.main()
