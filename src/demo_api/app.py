from __future__ import annotations

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import Request

from .schemas import (
    AutoAppealFromLetterResponse,
    GenerateAppealRequest,
    GenerateAppealResponse,
    ProviderHealthResponse,
    ScoreClaimsCsvResponse,
    VoiceCallStartRequest,
    VoiceCallSummaryResponse,
    VoiceCallTurnRequest,
    VoiceCallTurnResponse,
    VoiceExplainRequest,
    VoiceExplainResponse,
)
from .services import DemoService

app = FastAPI(title="MedClaimly Hackathon API", version="0.1.0")
service = DemoService()

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 60
_RATE_BUCKETS: dict[str, list[float]] = {}


def _enforce_rate_limit(request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _RATE_BUCKETS.setdefault(client_host, [])
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    bucket[:] = [ts for ts in bucket if ts >= cutoff]
    if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in 1 minute.")
    bucket.append(now)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    _enforce_rate_limit(request)
    return {"status": "ok"}


@app.get("/llm_provider_health", response_model=ProviderHealthResponse)
def llm_provider_health(request: Request) -> ProviderHealthResponse:
    _enforce_rate_limit(request)
    return service.get_provider_health()


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> dict:
    _enforce_rate_limit(request)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    result = service.classify_upload(
        data=data,
        filename=file.filename or "unknown",
    )
    return result.model_dump()


@app.post("/score_claims_csv", response_model=ScoreClaimsCsvResponse)
async def score_claims_csv(
    request: Request,
    file: UploadFile = File(...),
) -> ScoreClaimsCsvResponse:
    _enforce_rate_limit(request)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported for this endpoint.")
    try:
        return service.score_claims_csv(
            data=data,
            filename=file.filename or "claims.csv",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/auto_appeal_from_letter", response_model=AutoAppealFromLetterResponse)
async def auto_appeal_from_letter(
    request: Request,
    file: UploadFile = File(...),
    patient_context: str = "",
    llm_provider: str = "deepseek",
    denial_source: str = "",
    drg_code: str = "",
    carc_code: str = "",
    caveat_notes: str = "",
) -> AutoAppealFromLetterResponse:
    _enforce_rate_limit(request)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return service.auto_appeal_from_letter(
        data=data,
        filename=file.filename or "denial_letter.txt",
        patient_context=patient_context,
        llm_provider=llm_provider,
        denial_source=denial_source,
        drg_code=drg_code,
        carc_code=carc_code,
        caveat_notes=caveat_notes,
    )


@app.post("/generate_appeal", response_model=GenerateAppealResponse)
def generate_appeal(request: Request, payload: GenerateAppealRequest) -> GenerateAppealResponse:
    _enforce_rate_limit(request)
    (
        letter,
        policy_excerpts,
        generator,
        provider_requested,
        provider_used,
        fallback_reason,
        context_summary,
        context_snapshot,
    ) = service.generate_appeal(payload.model_dump())
    return GenerateAppealResponse(
        appeal_letter=letter,
        policy_excerpts=policy_excerpts,
        generator=generator,
        provider_requested=provider_requested,
        provider_used=provider_used,
        fallback_reason=fallback_reason,
        context_summary=context_summary,
        context_snapshot=context_snapshot,
    )


@app.post("/voice_explain", response_model=VoiceExplainResponse)
def voice_explain(request: Request, payload: VoiceExplainRequest) -> VoiceExplainResponse:
    _enforce_rate_limit(request)
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    audio_base64 = service.voice_explain(text=payload.text, voice_name=payload.voice_name)
    return VoiceExplainResponse(audio_base64=audio_base64)


@app.post("/voice_call/start", response_model=VoiceCallTurnResponse)
def voice_call_start(request: Request, payload: VoiceCallStartRequest) -> VoiceCallTurnResponse:
    _enforce_rate_limit(request)
    try:
        return service.start_voice_call(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/voice_call/{call_id}/turn", response_model=VoiceCallTurnResponse)
def voice_call_turn(request: Request, call_id: str, payload: VoiceCallTurnRequest) -> VoiceCallTurnResponse:
    _enforce_rate_limit(request)
    if not payload.user_text.strip():
        raise HTTPException(status_code=400, detail="user_text must not be empty")
    try:
        return service.process_voice_call_turn(
            call_id=call_id,
            user_text=payload.user_text,
            llm_provider=payload.llm_provider,
            tts_backend=payload.tts_backend,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/voice_call/{call_id}/turn_audio", response_model=VoiceCallTurnResponse)
async def voice_call_turn_audio(
    request: Request,
    call_id: str,
    file: UploadFile = File(...),
    llm_provider: str | None = None,
    tts_backend: str | None = None,
) -> VoiceCallTurnResponse:
    _enforce_rate_limit(request)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty")
    try:
        return service.process_voice_call_audio_turn(
            call_id=call_id,
            audio_data=data,
            filename=file.filename or "audio.wav",
            llm_provider=llm_provider,
            tts_backend=tts_backend,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/voice_call/{call_id}/summary", response_model=VoiceCallSummaryResponse)
def voice_call_summary(
    request: Request,
    call_id: str,
    llm_provider: str = "deepseek",
) -> VoiceCallSummaryResponse:
    _enforce_rate_limit(request)
    try:
        return service.get_voice_call_summary(call_id=call_id, llm_provider=llm_provider)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
