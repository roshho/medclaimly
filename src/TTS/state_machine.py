#!/usr/bin/env python3
"""Conversation state machine for provider-to-payer call scenarios.

Tracks which phase of the call we are in so the LLM system prompt
can include phase-appropriate instructions.  States are *advisory* —
they guide the LLM but do not prevent it from answering out-of-order
questions.
"""
from __future__ import annotations

import re
from enum import Enum, auto
from typing import Any


# ======================================================================
# State definitions
# ======================================================================

class S2State(Enum):
    """Scenario 2 — IVR claim status + representative detail."""
    IVR_GREETING = auto()
    IVR_AUTH_NPI = auto()
    IVR_CLAIM_LOOKUP = auto()
    IVR_STATUS_CODES = auto()
    TRANSFER_TO_REP = auto()
    REP_DETAIL = auto()
    REP_DOCUMENTATION = auto()
    REP_REFERENCE = auto()
    CALL_CLOSE = auto()


class S3State(Enum):
    """Scenario 3 — Appeal receipt confirmation."""
    IVR_GREETING = auto()
    IVR_AUTH_NPI = auto()
    TRANSFER_TO_REP = auto()
    REP_CONFIRM_RECEIPT = auto()
    REP_CLINICAL_REVIEW = auto()
    REP_TRACKING = auto()
    REP_TIMELINE = auto()
    CALL_CLOSE = auto()


# ======================================================================
# Hints per state — injected into the LLM system prompt
# ======================================================================

S2_HINTS: dict[S2State, str] = {
    S2State.IVR_GREETING: (
        "The payer IVR has greeted you. You should wait for the authentication prompt."
    ),
    S2State.IVR_AUTH_NPI: (
        "The IVR is asking for your NPI. Provide your 10-digit NPI to authenticate."
    ),
    S2State.IVR_CLAIM_LOOKUP: (
        "The IVR wants the claim control number. Provide the 15-digit claim ID."
    ),
    S2State.IVR_STATUS_CODES: (
        "The IVR has returned CARC/RARC denial codes. You should note these codes "
        "and request to speak with a representative (press 0) for detailed information."
    ),
    S2State.TRANSFER_TO_REP: (
        "You are being transferred to a live claims representative. "
        "Introduce your issue: identify the claim, beneficiary, and denial codes. "
        "Ask which service lines were affected."
    ),
    S2State.REP_DETAIL: (
        "You are speaking with a representative. Ask about the specific denial reason, "
        "which service lines are affected, and why the LOS was considered inconsistent."
    ),
    S2State.REP_DOCUMENTATION: (
        "Ask the representative what documentation is needed for a written appeal. "
        "Confirm the filing deadline (120 days)."
    ),
    S2State.REP_REFERENCE: (
        "The representative has given the info you need. Request a call reference number "
        "for this inquiry and confirm it."
    ),
    S2State.CALL_CLOSE: (
        "The call is wrapping up. Thank the representative and confirm you have "
        "the reference number, denial codes, and documentation requirements."
    ),
}

S3_HINTS: dict[S3State, str] = {
    S3State.IVR_GREETING: (
        "The payer appeals IVR has greeted you. Wait for the authentication prompt."
    ),
    S3State.IVR_AUTH_NPI: (
        "Provide your 10-digit NPI to authenticate with the appeals line."
    ),
    S3State.TRANSFER_TO_REP: (
        "Request transfer to a live appeals representative (press 0). "
        "Appeals cannot be submitted via IVR — this call confirms receipt of a written appeal."
    ),
    S3State.REP_CONFIRM_RECEIPT: (
        "You are speaking with an appeals rep. State the claim number, beneficiary ID, "
        "prior call reference, and that you faxed a written appeal. "
        "Ask the rep to confirm the fax was received."
    ),
    S3State.REP_CLINICAL_REVIEW: (
        "The representative is reviewing documents. If asked, provide clinical justification "
        "based on patient evidence: ventilator duration, diagnoses, comorbidity impact on LOS. "
        "NEVER reference model predictions, denial scores, or classifier outputs."
    ),
    S3State.REP_TRACKING: (
        "The representative should assign an appeal tracking number. "
        "Confirm the tracking number and ask about the review timeline."
    ),
    S3State.REP_TIMELINE: (
        "Confirm the review period (30 days for Medicare redetermination), "
        "determination method (written by mail), and how to check status later."
    ),
    S3State.CALL_CLOSE: (
        "Wrap up the call. Confirm the tracking number and timeline."
    ),
}


# ======================================================================
# State machine
# ======================================================================

class CallStateMachine:
    """Track the current phase of a provider-to-payer call."""

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        if scenario == "2":
            self.state: S2State | S3State = S2State.IVR_GREETING
        else:
            self.state = S3State.IVR_GREETING

        self.history: list[dict[str, str]] = []  # [{state, user_input, response}]

    # ------------------------------------------------------------------
    # Advance based on user (payer) input
    # ------------------------------------------------------------------

    def advance(self, user_input: str) -> None:
        """Advance state based on the payer's latest prompt."""
        line = re.sub(r"\s+", " ", user_input.strip().lower())

        if self.scenario == "2":
            self._advance_s2(line)
        else:
            self._advance_s3(line)

    def _advance_s2(self, line: str) -> None:
        """Scenario 2 state transitions."""
        cur = self.state
        assert isinstance(cur, S2State)

        if cur == S2State.IVR_GREETING and re.search(r"npi|provider\s*id|authenticate", line):
            self.state = S2State.IVR_AUTH_NPI
        elif cur == S2State.IVR_AUTH_NPI and re.search(r"claim|control\s*number|icn", line):
            self.state = S2State.IVR_CLAIM_LOOKUP
        elif cur in (S2State.IVR_AUTH_NPI, S2State.IVR_CLAIM_LOOKUP) and re.search(
            r"status|denied|carc|rarc|denial", line
        ):
            self.state = S2State.IVR_STATUS_CODES
        elif cur == S2State.IVR_STATUS_CODES and re.search(r"transfer|rep|agent|press\s*0|speak", line):
            self.state = S2State.TRANSFER_TO_REP
        elif cur in (S2State.TRANSFER_TO_REP, S2State.IVR_STATUS_CODES) and re.search(
            r"how\s*can\s*i\s*help|sarah|department|service\s*line|detail|which\s*line", line
        ):
            self.state = S2State.REP_DETAIL
        elif cur == S2State.REP_DETAIL and re.search(r"document|submit|what.*need|appeal|written", line):
            self.state = S2State.REP_DOCUMENTATION
        elif cur == S2State.REP_DOCUMENTATION and re.search(r"reference|rn-|call\s*ref|confirmation", line):
            self.state = S2State.REP_REFERENCE
        elif re.search(r"goodbye|end\s*call|hang\s*up|that.*all", line):
            self.state = S2State.CALL_CLOSE

        # Also allow skipping ahead from any state
        if re.search(r"transfer|press\s*0|speak.*(rep|agent)", line) and cur.value < S2State.TRANSFER_TO_REP.value:
            self.state = S2State.TRANSFER_TO_REP
        if re.search(r"goodbye|end\s*call|hang\s*up", line):
            self.state = S2State.CALL_CLOSE

    def _advance_s3(self, line: str) -> None:
        """Scenario 3 state transitions."""
        cur = self.state
        assert isinstance(cur, S3State)

        if cur == S3State.IVR_GREETING and re.search(r"npi|provider\s*id|authenticate", line):
            self.state = S3State.IVR_AUTH_NPI
        elif cur == S3State.IVR_AUTH_NPI and re.search(r"transfer|rep|agent|press\s*0|speak", line):
            self.state = S3State.TRANSFER_TO_REP
        elif cur in (S3State.TRANSFER_TO_REP, S3State.IVR_AUTH_NPI) and re.search(
            r"how\s*can\s*i\s*help|mark|appeal|fax|receipt|confirm", line
        ):
            self.state = S3State.REP_CONFIRM_RECEIPT
        elif cur == S3State.REP_CONFIRM_RECEIPT and re.search(
            r"clinical|justification|medical\s*necessity|document|review", line
        ):
            self.state = S3State.REP_CLINICAL_REVIEW
        elif cur in (S3State.REP_CLINICAL_REVIEW, S3State.REP_CONFIRM_RECEIPT) and re.search(
            r"track|rn-|assign|appeal.*number|reference", line
        ):
            self.state = S3State.REP_TRACKING
        elif cur == S3State.REP_TRACKING and re.search(r"timeline|how\s*long|30\s*day|when|review|status", line):
            self.state = S3State.REP_TIMELINE
        elif re.search(r"goodbye|end\s*call|hang\s*up|that.*all", line):
            self.state = S3State.CALL_CLOSE

        # Allow skipping ahead
        if re.search(r"transfer|press\s*0|speak.*(rep|agent)", line) and cur.value < S3State.TRANSFER_TO_REP.value:
            self.state = S3State.TRANSFER_TO_REP
        if re.search(r"goodbye|end\s*call|hang\s*up", line):
            self.state = S3State.CALL_CLOSE

    # ------------------------------------------------------------------
    # Context for LLM prompt
    # ------------------------------------------------------------------

    def get_state_context(self) -> str:
        """Return a text description of the current call phase for the LLM."""
        if self.scenario == "2":
            assert isinstance(self.state, S2State)
            hint = S2_HINTS.get(self.state, "Continue the conversation naturally.")
        else:
            assert isinstance(self.state, S3State)
            hint = S3_HINTS.get(self.state, "Continue the conversation naturally.")
        return f"[Call Phase: {self.state.name}] {hint}"

    @property
    def state_label(self) -> str:
        return self.state.name

    def record_turn(self, user_input: str, response: str) -> None:
        """Record a conversation turn in history."""
        self.history.append(
            {
                "state": self.state.name,
                "user_input": user_input,
                "response": response,
            }
        )
