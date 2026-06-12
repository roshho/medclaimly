from __future__ import annotations

import unittest

from src.demo_api.services import DemoService


class DemoServiceGuardrailsTest(unittest.TestCase):
    def test_build_retrieval_query_prioritizes_keys_and_includes_context(self) -> None:
        payload = {
            "denial_source": "drg_shortstay_upcoding",
            "drg_code": "207",
            "carc_code": "CO-4",
            "denial_reason": "Medical necessity documentation insufficient",
            "rationale": "Ventilator support exceeds 96 hours",
            "patient_context": "COPD exacerbation with pulmonary candidiasis",
        }
        context_summary = "Request reconsideration under LCD/NCD criteria."

        query = DemoService._build_retrieval_query(payload, context_summary)

        self.assertTrue(query.startswith("drg_shortstay_upcoding 207 CO-4"))
        self.assertIn("Medical necessity documentation insufficient", query)
        self.assertIn("Ventilator support exceeds 96 hours", query)
        self.assertIn("COPD exacerbation with pulmonary candidiasis", query)
        self.assertIn("Request reconsideration under LCD/NCD criteria.", query)

    def test_enforce_letter_guardrails_redacts_forbidden_tokens(self) -> None:
        draft = (
            "This letter references denial_risk_score and denial_source; "
            "it also mentions stage1_decision."
        )
        cleaned, reason = DemoService._enforce_letter_guardrails(draft)

        self.assertNotEqual(reason, "")
        self.assertNotIn("denial_risk_score", cleaned.lower())
        self.assertNotIn("denial_source", cleaned.lower())
        self.assertNotIn("stage1_decision", cleaned.lower())
        self.assertIn("[redacted]", cleaned)

    def test_contains_false_positive_caveat(self) -> None:
        self.assertTrue(
            DemoService._contains_false_positive_caveat(
                "This may be a false positive and needs manual review."
            )
        )
        self.assertFalse(
            DemoService._contains_false_positive_caveat(
                "Clinical documentation appears complete."
            )
        )


if __name__ == "__main__":
    unittest.main()
