import unittest

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
