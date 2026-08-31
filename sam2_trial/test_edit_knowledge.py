import json
import tempfile
import unittest
from pathlib import Path

from edit_knowledge import (
    KNOWLEDGE_PATH,
    KnowledgeError,
    load_editing_knowledge,
    retrieve_editing_knowledge,
)
from grounding import OneClickEditPlan, _constrain_plan_to_retrieved_capabilities


class EditingKnowledgeTests(unittest.TestCase):
    def test_retrieval_returns_only_relevant_trusted_operation_cards(self) -> None:
        retrieval = retrieve_editing_knowledge("保留左边的人物，背景透明并提亮")
        self.assertEqual(
            set(retrieval.matched_operation_ids),
            {"background.transparent", "subject.brightness"},
        )
        payload = retrieval.as_prompt_data()
        self.assertEqual(payload["catalog_version"], "2026-08-31.2")
        self.assertTrue(
            {"subject.single_visible", "policy.non_generative", "response.contract"}
            .issubset({card["id"] for card in payload["retrieved_capabilities"]})
        )

    def test_background_becomes_white_retrieves_and_allows_solid_color(self) -> None:
        retrieval = retrieve_editing_knowledge("保留奶酪，背景变白")
        self.assertEqual(retrieval.matched_operation_ids, ("background.color",))
        plan = OneClickEditPlan(
            status="ready",
            target="奶酪",
            selection={"edge_offset": 0, "feather_px": 0, "cleanup": True},
            background={"mode": "color", "color": "#ffffff", "blur_px": 0},
            subject={"brightness": 0, "saturation": 0, "blur_px": 0},
            summary="保留奶酪，背景变白。",
        )
        safe_plan = _constrain_plan_to_retrieved_capabilities(plan, retrieval)
        self.assertEqual(safe_plan.status, "ready")
        self.assertEqual(safe_plan.background, {"mode": "color", "color": "#ffffff", "blur_px": 0})

    def test_subject_becomes_white_does_not_imply_a_background_change(self) -> None:
        retrieval = retrieve_editing_knowledge("把奶酪变白")
        self.assertNotIn("background.color", retrieval.matched_operation_ids)

    def test_parameterised_background_colour_still_retrieves_solid_colour(self) -> None:
        retrieval = retrieve_editing_knowledge("保留杯子，背景换成红色")
        self.assertIn("background.color", retrieval.matched_operation_ids)

    def test_focus_subject_shorthand_retrieves_the_visible_recipe(self) -> None:
        retrieval = retrieve_editing_knowledge("让这个商品更显眼")
        self.assertEqual(
            set(retrieval.matched_operation_ids),
            {"background.blur", "subject.brightness"},
        )
        plan = OneClickEditPlan(
            status="ready",
            target="商品",
            selection={"edge_offset": 0, "feather_px": 0, "cleanup": True},
            background={"mode": "blur", "color": "#ffffff", "blur_px": 18},
            subject={"brightness": 8, "saturation": 0, "blur_px": 0},
            summary="突出商品。",
        )
        safe_plan = _constrain_plan_to_retrieved_capabilities(plan, retrieval)
        self.assertEqual(safe_plan.background["mode"], "blur")
        self.assertEqual(safe_plan.subject["brightness"], 8)

    def test_catalog_change_updates_retrieval_without_code_change(self) -> None:
        source = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        transparent = next(card for card in source["operations"] if card["id"] == "background.transparent")
        transparent["aliases"].append("透明化")
        source["catalog_version"] = "2026-08-31-rag-test"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
            retrieval = retrieve_editing_knowledge("把背景透明化", load_editing_knowledge(path))
        self.assertIn("background.transparent", retrieval.matched_operation_ids)
        self.assertEqual(retrieval.catalog_version, "2026-08-31-rag-test")

    def test_catalog_cannot_introduce_an_unknown_executable_capability(self) -> None:
        source = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        source["operations"][0]["id"] = "object.remove"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(KnowledgeError):
                load_editing_knowledge(path)

    def test_plan_with_effect_not_in_retrieved_knowledge_falls_back_to_needs_input(self) -> None:
        retrieval = retrieve_editing_knowledge("保留人物，背景透明")
        plan = OneClickEditPlan(
            status="ready",
            target="the person",
            selection={"edge_offset": 0, "feather_px": 0, "cleanup": True},
            background={"mode": "blur", "color": "#ffffff", "blur_px": 18},
            subject={"brightness": 0, "saturation": 0, "blur_px": 0},
            summary="保留人物，背景虚化。",
        )
        safe_plan = _constrain_plan_to_retrieved_capabilities(plan, retrieval)
        self.assertEqual(safe_plan.status, "needs_input")
        self.assertEqual(safe_plan.background["mode"], "original")

    def test_unrequested_subject_adjustment_is_removed_without_losing_requested_background(self) -> None:
        retrieval = retrieve_editing_knowledge("保留人物，背景透明")
        plan = OneClickEditPlan(
            status="ready",
            target="the person",
            selection={"edge_offset": 0, "feather_px": 0, "cleanup": True},
            background={"mode": "transparent", "color": "#ffffff", "blur_px": 0},
            subject={"brightness": 20, "saturation": 0, "blur_px": 0},
            summary="保留人物，背景透明并提亮。",
        )
        safe_plan = _constrain_plan_to_retrieved_capabilities(plan, retrieval)
        self.assertEqual(safe_plan.status, "ready")
        self.assertEqual(safe_plan.background["mode"], "transparent")
        self.assertEqual(safe_plan.subject["brightness"], 0)


if __name__ == "__main__":
    unittest.main()
