from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    denial_type: str = Field(..., description="High-level denial type.")
    denial_reason: str = Field(..., description="Primary denial reason.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    denial_code: str = Field(..., description="Synthetic code for demo traceability.")
    rationale: str
    extracted_text_preview: str


class GenerateAppealRequest(BaseModel):
    denial_type: str
    denial_reason: str
    denial_code: str
    rationale: str = ""
    patient_context: str = ""
    denial_source: str = ""
    drg_code: str = ""
    carc_code: str = ""
    caveat_notes: str = ""
    llm_provider: str = Field(
        default="deepseek",
        description="Target LLM provider: deepseek (primary) or grok (backup).",
    )


class GenerateAppealResponse(BaseModel):
    appeal_letter: str
    policy_excerpts: str
    generator: str
    provider_requested: str
    provider_used: str
    fallback_reason: str = ""
    context_summary: str = ""
    context_snapshot: dict[str, Any] = Field(default_factory=dict)


class VoiceExplainRequest(BaseModel):
    text: str
    voice_name: str = "en-US-JennyNeural"


class VoiceExplainResponse(BaseModel):
    mime_type: str = "audio/mpeg"
    audio_base64: str


class ClaimScoreResult(BaseModel):
    row_index: int
    claim_id: str
    inference_mode: str = Field(
        ..., description="model for Stage2 XGBoost scoring, fallback_llm for LLM fallback, rejected for invalid rows"
    )
    status: str = Field(..., description="scored, fallback, or rejected")
    denial_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    predicted_denial_flag: int | None = Field(default=None, ge=0, le=1)
    threshold_used: float | None = Field(default=None, ge=0.0, le=1.0)
    denial_reason: str
    denial_code: str
    rationale: str
    preemptive_note: str
    error: str = ""


class ScoreClaimsCsvResponse(BaseModel):
    sample_id: int
    model_version: str
    rows_total: int
    rows_scored: int
    rows_fallback: int
    rows_rejected: int
    results: list[ClaimScoreResult]
    claim_rows: list[dict[str, Any]] = Field(default_factory=list)
    schema_validation: dict[str, Any] = Field(default_factory=dict)


class AutoAppealFromLetterResponse(BaseModel):
    classification: ClassificationResult
    appeal_letter: str
    evidence_checklist: list[str]
    policy_excerpts: str
    generator: str
    provider_requested: str
    provider_used: str
    fallback_reason: str = ""
    context_summary: str
    context_snapshot: dict[str, Any] = Field(default_factory=dict)


class ProviderHealthStatus(BaseModel):
    provider: str
    configured: bool
    healthy: bool
    detail: str = ""


class ProviderHealthResponse(BaseModel):
    providers: list[ProviderHealthStatus]


class VoiceCallStartRequest(BaseModel):
    scenario: str = Field(default="2", description="Call scenario id: 2 or 3")
    llm_provider: str = Field(default="deepseek")
    tts_backend: str = Field(default="elevenlabs", description="elevenlabs or edge-tts")
    claim_context: dict[str, Any] = Field(default_factory=dict)


class VoiceCallTurnRequest(BaseModel):
    user_text: str
    llm_provider: str | None = Field(default=None)
    tts_backend: str | None = Field(default=None)


class VoiceCallTurnResponse(BaseModel):
    call_id: str
    current_state: str
    user_text: str = ""
    assistant_text: str
    audio_base64: str = ""
    mime_type: str = "audio/mpeg"
    fallback_reason: str = ""
    transcript_turns: int = 0


class VoiceCallSummaryResponse(BaseModel):
    call_id: str
    scenario: str
    current_state: str
    status_case: str
    key_summary: str
    missing_information: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    transcript_excerpt: list[dict[str, str]] = Field(default_factory=list)
    llm_provider_requested: str = ""
    llm_provider_used: str = ""
