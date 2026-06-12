#!/usr/bin/env python3
"""LLM clients via the OpenAI-compatible chat API.

DeepSeek is the primary provider; Grok (x.ai) is the backup. Both speak the
OpenAI chat-completions protocol, so a single thin wrapper covers each by
overriding ``base_url`` and the default model.

Also builds per-turn system prompts from claim context, call state, and RAG
policy excerpts.
"""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"

GROK_BASE_URL = "https://api.x.ai/v1"
GROK_DEFAULT_MODEL = "grok-2-latest"

# Backwards-compatible alias used by older imports.
DEFAULT_MODEL = DEEPSEEK_DEFAULT_MODEL


# ======================================================================
# System prompt templates
# ======================================================================

SYSTEM_PROMPT_SCENARIO_2 = """\
You are a provider office assistant calling a Medicare payer to inquire about a denied claim.
You are speaking either to an automated IVR system or to a live claims representative.

YOUR ROLE: You are the PROVIDER side of the call. The user is playing the PAYER/IVR.

CLAIM CONTEXT:
{claim_context}

CURRENT CALL PHASE:
{state_context}

RELEVANT MEDICARE POLICY EXCERPTS:
{policy_excerpts}

RULES — follow these strictly:
1. Respond in 1-3 sentences. Be concise, professional, and direct.
2. When in the IVR phase, provide requested identifiers (NPI, claim number) without elaboration.
3. When speaking with a representative, be more conversational but still focused.
4. NEVER mention "denial probability", "model predictions", "classifier", "XGBoost", "denial confidence", or any ML/AI terminology.
5. When discussing medical necessity, reference ONLY clinical evidence: diagnoses, treatments, ventilator duration, comorbidities.
6. If the payer asks for clinical justification, cite the patient's actual conditions and treatment, not statistical scores.
7. When relevant, reference Medicare policy or coverage determinations from the policy excerpts provided.
8. Use proper CARC/RARC codes when referencing denial reasons.
9. Do not invent claim details not present in the CLAIM CONTEXT.
10. If the payer says something unexpected, respond naturally while staying in character.
"""

SYSTEM_PROMPT_SCENARIO_3 = """\
You are a provider office assistant calling a Medicare payer to confirm receipt of a written appeal.
The appeal was already submitted by fax. This call is to verify the fax was received, confirm submitted documents, and obtain an appeal tracking number.

YOUR ROLE: You are the PROVIDER side of the call. The user is playing the PAYER/IVR.

CLAIM CONTEXT:
{claim_context}

CURRENT CALL PHASE:
{state_context}

RELEVANT MEDICARE POLICY EXCERPTS:
{policy_excerpts}

RULES — follow these strictly:
1. Respond in 1-3 sentences. Be concise, professional, and direct.
2. Appeals are NEVER submitted through IVR. You already faxed the appeal. You are calling to confirm receipt.
3. When in the IVR phase, provide NPI and navigate menus. Request transfer to a representative.
4. When speaking with a representative, identify the claim, beneficiary, prior call reference, and confirm the fax.
5. NEVER mention "denial probability", "model predictions", "classifier", "XGBoost", "denial confidence", or any ML/AI terminology.
6. Clinical justification must reference ONLY patient evidence: ventilator support exceeding 96 hours, acute exacerbation of chronic obstructive asthma, pulmonary candidiasis, atrial fibrillation, and their effect on length of stay.
7. When relevant, reference specific Medicare policy or coverage determinations from the policy excerpts provided.
8. Confirm the appeal tracking number and review timeline when provided by the representative.
9. Do not invent claim details not present in the CLAIM CONTEXT.
10. If the payer says something unexpected, respond naturally while staying in character.
"""


class DeepSeekClient:
    """Wrapper for DeepSeek chat completions via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEEPSEEK_DEFAULT_MODEL,
        base_url: str = DEEPSEEK_BASE_URL,
    ) -> None:
        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Pass --api-key or set DEEPSEEK_API_KEY env var."
            )
        self.model = model
        self.client = OpenAI(api_key=resolved_key, base_url=base_url)
        
        # Validate the API key with a test call
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                max_tokens=1,
            )
        except Exception as e:
            raise ValueError(f"DeepSeek API key validation failed: {e}")

    def generate_response(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        temperature: float = 0.7,
    ) -> str:
        """Call DeepSeek chat completion and return the assistant message.

        conversation_history is a list of {"role": "user"|"assistant", "content": "..."}.
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=300,
            )
            content = response.choices[0].message.content
            return content.strip() if content else "(No response generated.)"
        except Exception as exc:
            return f"(LLM error: {exc})"


class GrokClient:
    """Backup provider: Grok (x.ai) via the OpenAI-compatible chat API.

    Drop-in alternative to :class:`DeepSeekClient` with the same
    ``generate_response`` contract, used when DeepSeek is unavailable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = GROK_BASE_URL,
    ) -> None:
        resolved_key = api_key or os.environ.get("GROK_API_KEY", "") or os.environ.get("XAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Pass --api-key or set GROK_API_KEY (or XAI_API_KEY) env var."
            )
        self.model = model or os.environ.get("GROK_MODEL", GROK_DEFAULT_MODEL)
        self.client = OpenAI(api_key=resolved_key, base_url=base_url)

    def generate_response(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        temperature: float = 0.7,
    ) -> str:
        """Call Grok chat completion and return the assistant message.

        conversation_history is a list of {"role": "user"|"assistant", "content": "..."}.
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=300,
            )
            content = response.choices[0].message.content
            return content.strip() if content else "(No response generated.)"
        except Exception as exc:
            return f"(LLM error: {exc})"


def build_system_prompt(
    scenario: str,
    claim_context: dict[str, Any],
    state_context: str,
    policy_excerpts: str,
) -> str:
    """Assemble the full system prompt for a given turn."""
    forbidden_keys = {
        "classifier",
        "denial_risk_score",
        "denial_risk_pct",
        "denial_source",
        "stage1_decision",
        "caveat_notes",
    }
    # Serialize claim context after removing structured/internal fields
    # that must not be exposed in prompt context.
    sanitized = {k: v for k, v in claim_context.items() if k not in forbidden_keys}
    claim_json = json.dumps(sanitized, indent=2, default=str)

    template = SYSTEM_PROMPT_SCENARIO_2 if scenario == "2" else SYSTEM_PROMPT_SCENARIO_3

    return template.format(
        claim_context=claim_json,
        state_context=state_context,
        policy_excerpts=policy_excerpts or "(No policy excerpts retrieved for this turn.)",
    )
