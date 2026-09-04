import json
import tempfile
import unittest
from pathlib import Path

from edit_knowledge import (
    ALLOWED_CARD_IDS,
    AUTOMATIC_OPERATION_CARD_IDS,
    EXPECTED_SCOPE_BY_ID,
    KNOWLEDGE_PATH,
    KnowledgeError,
    load_editing_knowledge,
    retrieve_editing_knowledge,
)
from grounding import OneClickEditPlan, _constrain_plan_to_retrieved_capabilities


class EditingKnowledgeTests(unittest.TestCase):
    def test_retrieval_prioritises_relevant_cards_but_exposes_full_capability_set(self) -> None:
        retrieval = retrieve_editing_knowledge("保留左边的人物，背景透明并提亮")
        self.assertEqual(
            set(retrieval.matched_operation_ids),
            {"background.transparent", "subject.brightness"},
        )
        payload = retrieval.as_prompt_data()
        self.assertEqual(payload["catalog_version"], "2026-09-04.2")
        self.assertEqual(
            set(payload["matched_operation_ids"]),
            {"background.transparent", "subject.brightness"},
        )
        payload_ids = {card["id"] for card in payload["retrieved_capabilities"]}
        self.assertEqual(payload_ids, ALLOWED_CARD_IDS)
        self.assertTrue(AUTOMATIC_OPERATION_CARD_IDS.issubset(payload_ids))
        self.assertTrue(
            {"subject.single_visible", "policy.non_generative", "response.contract"}
            .issubset(payload_ids)
        )

    def test_retrieval_ignores_punctuation_between_intent_words(self) -> None:
        retrieval = retrieve_editing_knowledge("保留左边人物，背景，透明")
        self.assertIn("background.transparent", retrieval.matched_operation_ids)

    def test_retrieval_limit_never_removes_capabilities_from_the_prompt(self) -> None:
        retrieval = retrieve_editing_knowledge(
            "边缘自然，手动调整，保留原背景，透明背景，纯色背景，背景虚化，"
            "主体提亮、更鲜艳并柔焦"
        )
        self.assertEqual(len(retrieval.matched_operation_ids), 9)
        self.assertEqual(retrieval.available_ids, ALLOWED_CARD_IDS)

    def test_every_allowed_card_has_a_fixed_scope(self) -> None:
        self.assertEqual(set(EXPECTED_SCOPE_BY_ID), ALLOWED_CARD_IDS)

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
        source["catalog_version"] = "2026-09-01-rag-test"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
            retrieval = retrieve_editing_knowledge("把背景透明化", load_editing_knowledge(path))
        self.assertIn("background.transparent", retrieval.matched_operation_ids)
        self.assertEqual(retrieval.catalog_version, "2026-09-01-rag-test")

    def test_catalog_cannot_introduce_an_unknown_executable_capability(self) -> None:
        source = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        source["operations"][0]["id"] = "object.remove"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(KnowledgeError):
                load_editing_knowledge(path)

    def test_catalog_cannot_promote_manual_capability_to_automatic(self) -> None:
        source = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        manual = next(
            card for card in source["operations"] if card["id"] == "selection.manual_strokes"
        )
        manual["scope"] = "automatic"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(KnowledgeError):
                load_editing_knowledge(path)

    def test_supported_background_effect_survives_a_lexical_miss(self) -> None:
        retrieval = retrieve_editing_knowledge("保留人物，让周围看起来朦胧些")
        self.assertNotIn("background.blur", retrieval.matched_operation_ids)
        plan = OneClickEditPlan(
            status="ready",
            target="the person",
            selection={"edge_offset": 0, "feather_px": 0, "cleanup": True},
            background={"mode": "blur", "color": "#ffffff", "blur_px": 18},
            subject={"brightness": 0, "saturation": 0, "blur_px": 0},
            summary="保留人物，背景虚化。",
        )
        safe_plan = _constrain_plan_to_retrieved_capabilities(plan, retrieval)
        self.assertEqual(safe_plan.status, "ready")
        self.assertEqual(safe_plan.background["mode"], "blur")

    def test_supported_subject_adjustment_survives_a_lexical_miss(self) -> None:
        retrieval = retrieve_editing_knowledge("保留人物，让人物看起来精神些")
        self.assertNotIn("subject.brightness", retrieval.matched_operation_ids)
        plan = OneClickEditPlan(
            status="ready",
            target="the person",
            selection={"edge_offset": 0, "feather_px": 0, "cleanup": True},
            background={"mode": "original", "color": "#ffffff", "blur_px": 0},
            subject={"brightness": 20, "saturation": 0, "blur_px": 0},
            summary="保留人物并提亮。",
        )
        safe_plan = _constrain_plan_to_retrieved_capabilities(plan, retrieval)
        self.assertEqual(safe_plan.status, "ready")
        self.assertEqual(safe_plan.subject["brightness"], 20)

    def test_unrequested_shadow_is_removed_from_model_plan(self) -> None:
        retrieval = retrieve_editing_knowledge("把图中圆形体弄出来，背景弄成白色")
        self.assertNotIn("effect.shadow", retrieval.matched_operation_ids)
        plan = OneClickEditPlan(
            status="ready",
            target="图像下半部中央的近圆形区域",
            selection={"edge_offset": 0, "feather_px": 4, "cleanup": True},
            background={"mode": "color", "color": "#ffffff", "blur_px": 0},
            subject={"brightness": 0, "saturation": 0, "blur_px": 0},
            effects={
                "outline_width_px": 0,
                "outline_color": "#ffffff",
                "outline_opacity": 0,
                "shadow_offset_x": 0,
                "shadow_offset_y": 8,
                "shadow_blur_px": 12,
                "shadow_color": "#000000",
                "shadow_opacity": 35,
            },
            summary="提取圆形区域并换成白色背景。",
        )
        safe_plan = _constrain_plan_to_retrieved_capabilities(plan, retrieval)
        self.assertEqual(safe_plan.effects["shadow_offset_y"], 0)
        self.assertEqual(safe_plan.effects["shadow_blur_px"], 0)
        self.assertEqual(safe_plan.effects["shadow_opacity"], 0)

    def test_explicit_shadow_request_keeps_model_plan(self) -> None:
        retrieval = retrieve_editing_knowledge("把杯子弄出来，加一点柔和阴影")
        self.assertIn("effect.shadow", retrieval.matched_operation_ids)
        plan = OneClickEditPlan(
            status="ready",
            target="杯子",
            selection={"edge_offset": 0, "feather_px": 2, "cleanup": True},
            background={"mode": "original", "color": "#ffffff", "blur_px": 0},
            subject={"brightness": 0, "saturation": 0, "blur_px": 0},
            effects={
                "outline_width_px": 0,
                "outline_color": "#ffffff",
                "outline_opacity": 0,
                "shadow_offset_x": 0,
                "shadow_offset_y": 8,
                "shadow_blur_px": 12,
                "shadow_color": "#000000",
                "shadow_opacity": 35,
            },
            summary="提取杯子并添加柔和阴影。",
        )
        safe_plan = _constrain_plan_to_retrieved_capabilities(plan, retrieval)
        self.assertEqual(safe_plan.effects["shadow_offset_y"], 8)
        self.assertEqual(safe_plan.effects["shadow_blur_px"], 12)
        self.assertEqual(safe_plan.effects["shadow_opacity"], 35)

    def test_multi_effect_request_keeps_every_matched_operation(self) -> None:
        retrieval = retrieve_editing_knowledge(
            "保留人物，背景透明，边缘自然，主体提亮、更鲜艳并柔焦"
        )
        self.assertTrue(
            {
                "background.transparent",
                "selection.edge_feather",
                "subject.brightness",
                "subject.saturation",
                "subject.blur",
            }.issubset(set(retrieval.matched_operation_ids))
        )

    def test_natural_language_retrieval_finds_outline_shadow_crop_and_grayscale_cards(self) -> None:
        retrieval = retrieve_editing_knowledge(
            "给商品加一圈边，有点悬浮感，4:5裁剪，再让背景没有颜色"
        )
        self.assertTrue(
            {
                "effect.outline",
                "effect.shadow",
                "crop.subject",
                "background.grayscale",
            }.issubset(set(retrieval.matched_operation_ids))
        )

    def test_shadow_retrieval_accepts_explicit_paraphrases_but_not_scientific_terms(self) -> None:
        for instruction in (
            "让商品像贴纸一样浮起来",
            "make the product look raised with a shadow underneath",
        ):
            with self.subTest(instruction=instruction):
                retrieval = retrieve_editing_knowledge(instruction)
                self.assertIn("effect.shadow", retrieval.matched_operation_ids)
        for instruction in ("定位悬浮细胞", "分割这个 3D 结构"):
            with self.subTest(instruction=instruction):
                retrieval = retrieve_editing_knowledge(instruction)
                self.assertNotIn("effect.shadow", retrieval.matched_operation_ids)

    def test_negated_shadow_never_opens_the_execution_gate(self) -> None:
        for instruction in ("把杯子抠出来，不要阴影", "cut it out without shadow"):
            with self.subTest(instruction=instruction):
                retrieval = retrieve_editing_knowledge(instruction)
                self.assertNotIn("effect.shadow", retrieval.matched_operation_ids)

    def test_negated_cutout_never_forces_a_transparent_background(self) -> None:
        for instruction in (
            "保留杯子，不要抠图，保留原背景",
            "keep the cup; do not remove the background",
        ):
            with self.subTest(instruction=instruction):
                retrieval = retrieve_editing_knowledge(instruction)
                self.assertNotIn(
                    "background.transparent", retrieval.matched_operation_ids
                )

    def test_negated_background_effects_do_not_open_execution_gates(self) -> None:
        cases = (
            ("保留杯子，背景不要虚化", "background.blur"),
            ("保留杯子，不要让背景虚化", "background.blur"),
            ("keep the cup; do not blur the background", "background.blur"),
            ("keep the cup; don't make the background blurry", "background.blur"),
            ("保留杯子，不要白底", "background.color"),
            ("保留杯子，不要把背景变白", "background.color"),
            ("保留杯子，别让背景变白", "background.color"),
            ("keep the cup; don't use a white background", "background.color"),
        )
        for instruction, operation_id in cases:
            with self.subTest(instruction=instruction):
                retrieval = retrieve_editing_knowledge(instruction)
                self.assertNotIn(operation_id, retrieval.matched_operation_ids)

    def test_later_colour_correction_reopens_the_colour_gate(self) -> None:
        for instruction in (
            "保留杯子，不要白底，背景改成蓝色",
            "keep the cup; don't use a white background, make it blue",
        ):
            with self.subTest(instruction=instruction):
                retrieval = retrieve_editing_knowledge(instruction)
                self.assertIn("background.color", retrieval.matched_operation_ids)


if __name__ == "__main__":
    unittest.main()
