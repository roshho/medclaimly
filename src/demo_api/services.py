from __future__ import annotations

import asyncio
import base64
import datetime as dt
import io
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.TTS.llm_client import DeepSeekClient, GrokClient, build_system_prompt
from src.TTS.state_machine import CallStateMachine
from src.TTS.policy_rag import PolicyRAG

from .schemas import (
    AutoAppealFromLetterResponse,
    ClaimScoreResult,
    ClassificationResult,
    ProviderHealthResponse,
    ProviderHealthStatus,
    ScoreClaimsCsvResponse,
    VoiceCallStartRequest,
    VoiceCallSummaryResponse,
    VoiceCallTurnResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "rules" / "policy_corpus"
FOCUSED_CORPUS_DIR = PROJECT_ROOT / "output" / "policy_corpus_drg207_focus"
DEFAULT_SAMPLE_ID = int(os.environ.get("STAGE2_SAMPLE_ID", "1"))
# DeepSeek is the primary LLM; Grok (x.ai) is the automatic backup.
DEFAULT_LLM_PROVIDER = "deepseek"
LLM_BACKUP_PROVIDER = "grok"
SUPPORTED_LLM_PROVIDERS = {"deepseek", "grok"}
SUPPORTED_TTS_BACKENDS = {"elevenlabs", "edge-tts"}
ELEVENLABS_STT_MODEL = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1")
REQUIRED_TTS_CSV_COLUMNS = [
    "claim_id",
    "drg_code",
    "drg_description",
    "primary_diagnosis_code",
    "primary_diagnosis_description",
    "carc_code",
    "carc_description",
    "rarc_code",
    "rarc_description",
]


class _Stage2Artifacts:
    def __init__(
        self,
        sample_id: int,
        model: Any,
        threshold: float,
        feature_columns: list[str],
        model_version: str,
    ) -> None:
        self.sample_id = sample_id
        self.model = model
        self.threshold = threshold
        self.feature_columns = feature_columns
        self.model_version = model_version


def _best_effort_text_extract(data: bytes, filename: str) -> str:
    # Quick hackathon extractor: plain text decode with cleanup.
    if not data:
        return ""
    decoded = data.decode("utf-8", errors="ignore")
    if not decoded.strip():
        decoded = data.decode("latin-1", errors="ignore")
    cleaned = re.sub(r"\s+", " ", decoded).strip()
    if cleaned:
        return cleaned
    return f"Uploaded file: {filename}"


def _heuristic_classification(text: str) -> ClassificationResult:
    haystack = text.lower()

    if any(token in haystack for token in ["mue", "units exceeded", "medically unlikely"]):
        return ClassificationResult(
            denial_type="D",
            denial_reason="Medically Unlikely Edit (MUE) exceeded",
            confidence=0.9,
            denial_code="D-MUE-001",
            rationale="Claim suggests unit/day limits exceeded for billed HCPCS/CPT.",
            extracted_text_preview=text[:350],
        )

    if any(token in haystack for token in ["prior authorization", "missing auth", "no authorization"]):
        return ClassificationResult(
            denial_type="D",
            denial_reason="Missing prior authorization",
            confidence=0.88,
            denial_code="D-AUTH-002",
            rationale="Narrative indicates missing or invalid authorization for billed services.",
            extracted_text_preview=text[:350],
        )

    if any(token in haystack for token in ["medical necessity", "not reasonable and necessary", "ncd", "lcd"]):
        return ClassificationResult(
            denial_type="D",
            denial_reason="Medical necessity documentation insufficient",
            confidence=0.84,
            denial_code="D-MN-003",
            rationale="Coverage language suggests documentation did not meet LCD/NCD requirements.",
            extracted_text_preview=text[:350],
        )

    if any(token in haystack for token in ["non-covered", "noncovered", "carc co-4"]):
        return ClassificationResult(
            denial_type="D",
            denial_reason="Non-covered service or coding mismatch",
            confidence=0.82,
            denial_code="D-COVERAGE-004",
            rationale="Denial appears to involve non-covered service definitions or coding mismatches.",
            extracted_text_preview=text[:350],
        )

    return ClassificationResult(
        denial_type="D",
        denial_reason="Insufficient claim evidence - manual review needed",
        confidence=0.6,
        denial_code="D-REVIEW-000",
        rationale="No strong denial pattern found in uploaded text; fallback classification applied.",
        extracted_text_preview=text[:350],
    )


class DemoService:
    def __init__(self, corpus_dir: Path | None = None) -> None:
        self.corpus_dir = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
        rag_dirs: list[Path] = [self.corpus_dir]
        if FOCUSED_CORPUS_DIR.exists():
            rag_dirs.append(FOCUSED_CORPUS_DIR)
        self._rag = PolicyRAG(corpus_dirs=rag_dirs)
        self._artifacts_cache: dict[int, _Stage2Artifacts] = {}
        self._voice_call_sessions: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _normalize_scenario(scenario: str) -> str:
        candidate = str(scenario).strip()
        return candidate if candidate in {"2", "3"} else "2"

    def _load_default_claim_context(self, scenario: str) -> dict[str, Any]:
        sample_path = PROJECT_ROOT / "src" / "TTS" / "sample_claim_context.json"
        if not sample_path.exists():
            return {}
        try:
            payload = json.loads(sample_path.read_text(encoding="utf-8"))
            return payload.get(f"scenario_{scenario}", {}) if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _build_retrieval_query(payload: dict[str, Any], context_summary: str) -> str:
        # Prioritize structured denial keys before free-text context.
        priority_bits = [
            str(payload.get("denial_source", "")).strip(),
            str(payload.get("drg_code", "")).strip(),
            str(payload.get("carc_code", "")).strip(),
        ]
        context_bits = [
            str(payload.get("denial_reason", "")).strip(),
            str(payload.get("rationale", "")).strip(),
            str(payload.get("patient_context", "")).strip(),
            context_summary.strip(),
        ]
        return " ".join([x for x in priority_bits + context_bits if x])

    @staticmethod
    def _contains_false_positive_caveat(text: str) -> bool:
        return bool(re.search(r"\bfalse[\s-]*positive\b", text, flags=re.IGNORECASE))

    @staticmethod
    def _enforce_letter_guardrails(letter: str) -> tuple[str, str]:
        forbidden_tokens = [
            "denial_risk_score",
            "denial_risk_pct",
            "denial_source",
            "stage1_decision",
            "caveat_notes",
        ]
        sanitized = letter
        triggered: list[str] = []
        for token in forbidden_tokens:
            pattern = re.compile(rf"\b{re.escape(token)}\b", flags=re.IGNORECASE)
            if pattern.search(sanitized):
                triggered.append(token)
                sanitized = pattern.sub("[redacted]", sanitized)

        still_present = [
            token for token in forbidden_tokens if re.search(rf"\b{re.escape(token)}\b", sanitized, flags=re.IGNORECASE)
        ]
        if still_present:
            fallback_letter = (
                "Subject: Request for Reconsideration of Medicare Denial\n\n"
                "To Claims Review Department,\n\n"
                "Please reconsider this denied claim based on the attached clinical documentation and "
                "applicable Medicare policy criteria.\n\n"
                "The medical record supports necessity, coding alignment, and service appropriateness for the date(s) of service. "
                "We request full reprocessing and reversal of the denial.\n\n"
                "Sincerely,\nProvider Appeals Team"
            )
            reason = "Output guardrail triggered: forbidden structured fields leaked in draft; replaced with safe template."
            return fallback_letter, reason

        if triggered:
            reason = (
                "Output guardrail triggered: removed forbidden structured fields from draft: "
                + ", ".join(sorted(set(triggered)))
            )
            return sanitized, reason
        return sanitized, ""

    def classify_upload(self, data: bytes, filename: str) -> ClassificationResult:
        extracted = _best_effort_text_extract(data, filename)
        return _heuristic_classification(extracted)

    @staticmethod
    def _resolve_provider(provider: str | None) -> str:
        candidate = (provider or DEFAULT_LLM_PROVIDER).strip().lower()
        return candidate if candidate in SUPPORTED_LLM_PROVIDERS else DEFAULT_LLM_PROVIDER

    @staticmethod
    def _validate_tts_csv_schema(df: pd.DataFrame) -> dict[str, Any]:
        actual_cols = [str(c) for c in df.columns]
        missing_columns = [c for c in REQUIRED_TTS_CSV_COLUMNS if c not in actual_cols]

        row_errors: list[str] = []
        invalid_rows = 0
        for idx, row in df.iterrows():
            claim_id = str(row.get("claim_id", "")).strip()
            drg_code = str(row.get("drg_code", "")).strip()
            primary_dx = str(row.get("primary_diagnosis_code", "")).strip()
            if not claim_id:
                invalid_rows += 1
                row_errors.append(f"row {idx}: missing claim_id")
            if not drg_code:
                invalid_rows += 1
                row_errors.append(f"row {idx}: missing drg_code")
            if not primary_dx:
                invalid_rows += 1
                row_errors.append(f"row {idx}: missing primary_diagnosis_code")

        # Deduplicate repeated row errors.
        row_errors = sorted(set(row_errors))
        blocking_errors: list[str] = []
        if missing_columns:
            blocking_errors.append(
                "Missing required CSV columns: " + ", ".join(missing_columns)
            )

        return {
            "schema_version": "tts_v1",
            "required_columns": REQUIRED_TTS_CSV_COLUMNS,
            "missing_columns": missing_columns,
            "valid_rows": max(len(df) - invalid_rows, 0),
            "invalid_rows": invalid_rows,
            "row_errors": row_errors[:100],
            "blocking_errors": blocking_errors,
            "is_valid": len(blocking_errors) == 0,
        }

    def get_provider_health(self) -> ProviderHealthResponse:
        has_grok_key = bool(os.environ.get("GROK_API_KEY", "") or os.environ.get("XAI_API_KEY", ""))
        statuses = [
            ProviderHealthStatus(
                provider="deepseek",
                configured=bool(os.environ.get("DEEPSEEK_API_KEY", "")),
                healthy=bool(os.environ.get("DEEPSEEK_API_KEY", "")),
                detail="Primary LLM. Requires DEEPSEEK_API_KEY",
            ),
            ProviderHealthStatus(
                provider="grok",
                configured=has_grok_key,
                healthy=has_grok_key,
                detail="Backup LLM (x.ai). Requires GROK_API_KEY (or XAI_API_KEY)",
            ),
            ProviderHealthStatus(
                provider="elevenlabs",
                configured=bool(os.environ.get("ELEVENLABS_API_KEY", "")),
                healthy=bool(os.environ.get("ELEVENLABS_API_KEY", "")),
                detail="Voice STT + TTS. Requires ELEVENLABS_API_KEY",
            ),
        ]
        return ProviderHealthResponse(providers=statuses)

    @staticmethod
    def _build_auto_context(
        denial_type: str,
        denial_reason: str,
        denial_code: str,
        rationale: str,
        patient_context: str,
        extracted_preview: str = "",
    ) -> str:
        base_segments = [
            f"Denial type: {denial_type}",
            f"Denial reason: {denial_reason}",
            f"Denial code: {denial_code}",
            f"Key rationale: {rationale}",
        ]
        if patient_context.strip():
            base_segments.append(f"Additional patient/claim context: {patient_context.strip()}")
        if extracted_preview.strip():
            base_segments.append(f"Extracted denial letter preview: {extracted_preview[:300]}")
        return " | ".join(base_segments)

    def _llm_summarize_context(self, provider: str, context_text: str) -> str:
        provider_name = self._resolve_provider(provider)
        prompt = (
            "Summarize the following claim/denial context for an appeal letter in 2 concise sentences. "
            "Focus on medical necessity, coding/authorization signals, and requested payer action.\n\n"
            f"Context:\n{context_text}"
        )
        summary, used = self._call_llm_provider(provider_name, prompt)
        if summary.strip() and not summary.startswith("(LLM unavailable"):
            return f"{summary.strip()} [context_provider={used}]"
        return context_text

    @staticmethod
    def _is_usable_llm_text(text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        # DeepSeek/Grok wrappers signal failures with a parenthetical marker.
        return not stripped.startswith(("(LLM error", "(LLM unavailable", "(No response"))

    def _invoke_provider(self, provider: str, prompt: str) -> tuple[str, str]:
        """Call a single provider. Returns (text, used) or ("", "<name>-unavailable")."""
        system_prompt = "You are a concise medical appeals drafting assistant."
        try:
            if provider == "deepseek":
                api_key = os.environ.get("DEEPSEEK_API_KEY", "")
                if not api_key:
                    return "", "deepseek-unavailable"
                client = DeepSeekClient(api_key=api_key)
            elif provider == "grok":
                api_key = os.environ.get("GROK_API_KEY", "") or os.environ.get("XAI_API_KEY", "")
                if not api_key:
                    return "", "grok-unavailable"
                client = GrokClient(api_key=api_key)
            else:
                return "", f"{provider}-unavailable"

            text = client.generate_response(
                system_prompt=system_prompt,
                conversation_history=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            if self._is_usable_llm_text(text):
                return text, provider
            return "", f"{provider}-unavailable"
        except Exception:
            return "", f"{provider}-unavailable"

    def _call_llm_provider(self, provider: str, prompt: str) -> tuple[str, str]:
        """Call the requested provider, automatically failing over to the backup.

        DeepSeek is primary and Grok is the backup; if the requested provider is
        unavailable the other one is tried before giving up.
        """
        primary = self._resolve_provider(provider)
        backup = LLM_BACKUP_PROVIDER if primary == DEFAULT_LLM_PROVIDER else DEFAULT_LLM_PROVIDER

        text, used = self._invoke_provider(primary, prompt)
        if text:
            return text, used

        backup_text, backup_used = self._invoke_provider(backup, prompt)
        if backup_text:
            return backup_text, f"{backup_used} (failover from {primary})"

        return "", f"{primary}-unavailable"

    def _load_stage2_artifacts(self, sample_id: int) -> _Stage2Artifacts | None:
        if sample_id in self._artifacts_cache:
            return self._artifacts_cache[sample_id]

        model_path = PROJECT_ROOT / "output" / f"sample{sample_id:02d}_stage2_model.json"
        metrics_path = PROJECT_ROOT / "output" / f"sample{sample_id:02d}_stage2_metrics.csv"
        features_path = PROJECT_ROOT / "output" / f"sample{sample_id:02d}_stage2_features.parquet"

        if not model_path.exists():
            return None

        try:
            from xgboost import XGBClassifier
        except Exception:
            return None

        model = XGBClassifier()
        model.load_model(str(model_path))

        threshold = 0.5
        if metrics_path.exists():
            try:
                metrics_df = pd.read_csv(metrics_path)
                if not metrics_df.empty and "best_f1_threshold" in metrics_df.columns:
                    threshold = float(metrics_df.iloc[0]["best_f1_threshold"])
            except Exception:
                threshold = 0.5

        # Prefer feature names embedded in the saved model so the demo does not
        # depend on the large training-features parquet being present.
        feature_columns: list[str] = []
        try:
            booster_features = model.get_booster().feature_names
            if booster_features:
                feature_columns = list(booster_features)
        except Exception:
            feature_columns = []

        # Fallback: recover column order from the features parquet if available.
        if not feature_columns and features_path.exists():
            try:
                df_features = pd.read_parquet(features_path)
                drop_cols = {
                    "CLM_ID",
                    "DESYNPUF_ID",
                    "sample_id",
                    "CLM_FROM_DT",
                    "CLM_THRU_DT",
                    "CLM_DRG_CD",
                    "is_denied",
                }
                feature_columns = [c for c in df_features.columns if c not in drop_cols]
            except Exception:
                feature_columns = []

        if not feature_columns:
            return None

        artifact = _Stage2Artifacts(
            sample_id=sample_id,
            model=model,
            threshold=threshold,
            feature_columns=feature_columns,
            model_version=model_path.name,
        )
        self._artifacts_cache[sample_id] = artifact
        return artifact

    @staticmethod
    def _count_populated(row: pd.Series, prefix: str, start: int, end: int) -> int:
        count = 0
        for i in range(start, end + 1):
            col = f"{prefix}_{i}"
            if col in row and str(row[col]).strip() not in {"", "nan", "None"}:
                count += 1
        return count

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            text = str(value).strip()
            if text in {"", "nan", "None"}:
                return default
            return float(text)
        except Exception:
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(str(value).strip()))
        except Exception:
            return default

    @staticmethod
    def _month_from_yyyymmdd(value: Any) -> int:
        text = str(value).strip()
        if len(text) >= 6 and text.isdigit():
            month = int(text[4:6])
            if 1 <= month <= 12:
                return month
        return 0

    def _build_feature_map_from_csv_row(self, row: pd.Series) -> dict[str, float | int]:
        icd9_diag_count = self._count_populated(row, "ICD9_DGNS_CD", 1, 10)
        line_icd9_diag_count = self._count_populated(row, "LINE_ICD9_DGNS_CD", 1, 13)
        hcpcs_count = self._count_populated(row, "HCPCS_CD", 1, 45)
        total_diag_count = icd9_diag_count + line_icd9_diag_count

        utilization_days = self._safe_float(row.get("CLM_UTLZTN_DAY_CNT", 0.0), 0.0)
        drg_weight = self._safe_float(row.get("drg_weight", row.get("DRG_WEIGHT", 0.0)), 0.0)
        drg_los_reference = self._safe_float(
            row.get("drg_los_reference", row.get("DRG_LOS_REFERENCE", 0.0)),
            0.0,
        )

        los_outlier_flag = int(
            drg_los_reference > 0 and utilization_days > 1.5 * drg_los_reference
        )
        short_stay_flag = int(
            drg_los_reference > 0 and utilization_days < 0.5 * drg_los_reference
        )

        claim_month = self._month_from_yyyymmdd(row.get("CLM_FROM_DT", ""))

        bene_hi_coverage_months = self._safe_int(row.get("BENE_HI_CVRAGE_TOT_MONS", 0), 0)
        bene_smi_coverage_months = self._safe_int(row.get("BENE_SMI_CVRAGE_TOT_MONS", 0), 0)
        bene_medreimb_ip = self._safe_float(row.get("MEDREIMB_IP", 0.0), 0.0)
        bene_medreimb_car = self._safe_float(row.get("MEDREIMB_CAR", 0.0), 0.0)
        bene_benres_car = self._safe_float(row.get("BENRES_CAR", 0.0), 0.0)
        bene_sex = self._safe_int(row.get("BENE_SEX_IDENT_CD", 0), 0)

        chronic_flags = [
            "SP_DIABETES",
            "SP_CHF",
            "SP_CNCR",
            "SP_COPD",
            "SP_CHRNKIDN",
            "SP_OSTEOPRS",
            "SP_RA_OA",
            "SP_STRKETIA",
        ]
        chronic_condition_count = 0
        for flag in chronic_flags:
            raw = str(row.get(flag, "0")).strip().lower()
            if raw in {"1", "y", "yes", "true"}:
                chronic_condition_count += 1

        bene_age_at_claim = 0
        birth_text = str(row.get("BENE_BIRTH_DT", "")).strip()
        from_text = str(row.get("CLM_FROM_DT", "")).strip()
        if len(birth_text) >= 4 and len(from_text) >= 4 and birth_text[:4].isdigit() and from_text[:4].isdigit():
            bene_age_at_claim = max(int(from_text[:4]) - int(birth_text[:4]), 0)

        return {
            "icd9_diag_count": icd9_diag_count,
            "line_icd9_diag_count": line_icd9_diag_count,
            "hcpcs_count": hcpcs_count,
            "total_diag_count": total_diag_count,
            "claim_month": claim_month,
            "utilization_days": utilization_days,
            "drg_weight": drg_weight,
            "drg_los_reference": drg_los_reference,
            "los_outlier_flag": los_outlier_flag,
            "short_stay_flag": short_stay_flag,
            "bene_hi_coverage_months": bene_hi_coverage_months,
            "bene_smi_coverage_months": bene_smi_coverage_months,
            "bene_medreimb_ip": bene_medreimb_ip,
            "bene_medreimb_car": bene_medreimb_car,
            "bene_benres_car": bene_benres_car,
            "bene_age_at_claim": bene_age_at_claim,
            "bene_sex": bene_sex,
            "chronic_condition_count": chronic_condition_count,
        }

    @staticmethod
    def _top_risk_signals(feature_map: dict[str, float | int]) -> list[str]:
        signals: list[str] = []
        if float(feature_map.get("short_stay_flag", 0)) > 0:
            signals.append("short LOS compared with DRG baseline")
        if float(feature_map.get("los_outlier_flag", 0)) > 0:
            signals.append("LOS outlier for DRG profile")
        if float(feature_map.get("hcpcs_count", 0)) >= 3:
            signals.append("high HCPCS density")
        if float(feature_map.get("total_diag_count", 0)) <= 1 and float(feature_map.get("hcpcs_count", 0)) >= 2:
            signals.append("procedure-to-diagnosis imbalance")
        if float(feature_map.get("chronic_condition_count", 0)) == 0 and float(feature_map.get("utilization_days", 0)) >= 3:
            signals.append("extended utilization with limited chronic context")
        if not signals:
            signals.append("pattern resembles previously denied cohorts")
        return signals[:3]

    @staticmethod
    def _preemptive_note(signals: list[str]) -> str:
        base = (
            "Preemptive documentation: include explicit medical necessity statement, diagnosis-to-procedure linkage, "
            "supporting progress notes/labs/imaging, and authorization evidence when applicable."
        )
        signal_text = "; ".join(signals)
        return f"{base} Priority focus: {signal_text}."

    @staticmethod
    def _denial_reason_from_probability(probability: float, signals: list[str]) -> tuple[str, str, str]:
        if probability >= 0.8:
            return (
                "Medical necessity documentation insufficient",
                "D-MN-003",
                f"High denial risk ({probability:.3f}) driven by {', '.join(signals)}.",
            )
        if probability >= 0.6:
            return (
                "Non-covered service or coding mismatch",
                "D-COVERAGE-004",
                f"Moderate-high denial risk ({probability:.3f}) with key signals: {', '.join(signals)}.",
            )
        if probability >= 0.45:
            return (
                "Missing prior authorization",
                "D-AUTH-002",
                f"Borderline denial risk ({probability:.3f}); verify authorization and coding alignment.",
            )
        return (
            "Insufficient claim evidence - manual review needed",
            "D-REVIEW-000",
            f"Lower denial risk ({probability:.3f}) but manual validation is still recommended.",
        )

    def score_claims_csv(
        self,
        data: bytes,
        filename: str,
        sample_id: int | None = None,
    ) -> ScoreClaimsCsvResponse:
        resolved_sample_id = sample_id if sample_id is not None else DEFAULT_SAMPLE_ID
        artifacts = self._load_stage2_artifacts(resolved_sample_id)
        model_version = artifacts.model_version if artifacts else "unavailable"

        try:
            df = pd.read_csv(io.BytesIO(data))
        except Exception as exc:
            raise ValueError(f"Unable to parse CSV upload '{filename}': {exc}") from exc

        schema_validation = self._validate_tts_csv_schema(df)
        if not schema_validation.get("is_valid", False):
            return ScoreClaimsCsvResponse(
                sample_id=resolved_sample_id,
                model_version=model_version,
                rows_total=len(df),
                rows_scored=0,
                rows_fallback=0,
                rows_rejected=len(df),
                results=[],
                claim_rows=df.fillna("").to_dict(orient="records"),
                schema_validation=schema_validation,
            )

        results: list[ClaimScoreResult] = []

        for idx, row in df.iterrows():
            claim_id = str(row.get("CLM_ID", row.get("claim_id", f"row-{idx}")))
            row_text = " ".join([str(v) for v in row.values if str(v).strip() not in {"", "nan", "None"}])

            if not row_text.strip():
                results.append(
                    ClaimScoreResult(
                        row_index=int(idx),
                        claim_id=claim_id,
                        inference_mode="rejected",
                        status="rejected",
                        denial_reason="Empty row",
                        denial_code="D-ROW-EMPTY",
                        rationale="No usable row values found.",
                        preemptive_note="Populate required claim fields before scoring.",
                        error="No data in row",
                    )
                )
                continue

            feature_map = self._build_feature_map_from_csv_row(row)
            signals = self._top_risk_signals(feature_map)

            if artifacts is None:
                fallback = self.classify_upload(
                    data=row_text.encode("utf-8", errors="ignore"),
                    filename=f"{filename}:row-{idx}",
                )
                results.append(
                    ClaimScoreResult(
                        row_index=int(idx),
                        claim_id=claim_id,
                        inference_mode="fallback_llm",
                        status="fallback",
                        denial_reason=fallback.denial_reason,
                        denial_code=fallback.denial_code,
                        rationale=fallback.rationale,
                        preemptive_note=self._preemptive_note(signals),
                        error="Stage2 model artifacts unavailable",
                    )
                )
                continue

            try:
                payload = {col: 0.0 for col in artifacts.feature_columns}
                for key, value in feature_map.items():
                    if key in payload:
                        payload[key] = value

                X = pd.DataFrame([payload], columns=artifacts.feature_columns)
                for col in X.columns:
                    if X[col].dtype == "bool":
                        X[col] = X[col].astype(int)
                    elif not np.issubdtype(X[col].dtype, np.number):
                        X[col] = pd.to_numeric(X[col], errors="coerce")
                X = X.fillna(0.0)

                prob = float(artifacts.model.predict_proba(X)[:, 1][0])
                pred_flag = int(prob >= artifacts.threshold)
                reason, code, rationale = self._denial_reason_from_probability(prob, signals)

                results.append(
                    ClaimScoreResult(
                        row_index=int(idx),
                        claim_id=claim_id,
                        inference_mode="model",
                        status="scored",
                        denial_probability=prob,
                        predicted_denial_flag=pred_flag,
                        threshold_used=artifacts.threshold,
                        denial_reason=reason,
                        denial_code=code,
                        rationale=rationale,
                        preemptive_note=self._preemptive_note(signals),
                    )
                )
            except Exception as exc:
                fallback = self.classify_upload(
                    data=row_text.encode("utf-8", errors="ignore"),
                    filename=f"{filename}:row-{idx}",
                )
                results.append(
                    ClaimScoreResult(
                        row_index=int(idx),
                        claim_id=claim_id,
                        inference_mode="fallback_llm",
                        status="fallback",
                        denial_reason=fallback.denial_reason,
                        denial_code=fallback.denial_code,
                        rationale=fallback.rationale,
                        preemptive_note=self._preemptive_note(signals),
                        error=f"Model inference failed: {exc}",
                    )
                )

        rows_total = len(results)
        rows_scored = len([r for r in results if r.status == "scored"])
        rows_fallback = len([r for r in results if r.status == "fallback"])
        rows_rejected = len([r for r in results if r.status == "rejected"])

        return ScoreClaimsCsvResponse(
            sample_id=resolved_sample_id,
            model_version=model_version,
            rows_total=rows_total,
            rows_scored=rows_scored,
            rows_fallback=rows_fallback,
            rows_rejected=rows_rejected,
            results=results,
            claim_rows=df.fillna("").to_dict(orient="records"),
            schema_validation=schema_validation,
        )

    def generate_appeal(self, payload: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, dict[str, Any]]:
        llm_provider = self._resolve_provider(str(payload.get("llm_provider", "deepseek")))
        context_summary = self._build_auto_context(
            denial_type=str(payload.get("denial_type", "D")),
            denial_reason=str(payload.get("denial_reason", "")),
            denial_code=str(payload.get("denial_code", "")),
            rationale=str(payload.get("rationale", "")),
            patient_context=str(payload.get("patient_context", "")),
        )
        context_summary = self._llm_summarize_context(llm_provider, context_summary)

        query = self._build_retrieval_query(payload, context_summary)
        retrieval = self._rag.retrieve(query or "medicare denial appeal", top_k=3)
        policy_excerpts = self._rag.format_for_prompt(retrieval)

        user_text = (
            "Draft a concise formal Medicare appeal letter (max 5 short paragraphs + action request).\n"
            "Do not include or mention internal structured field names in the final letter text: "
            "denial_risk_score, denial_risk_pct, denial_source, stage1_decision, caveat_notes.\n"
            f"Denial type: {payload.get('denial_type', '')}\n"
            f"Denial reason: {payload.get('denial_reason', '')}\n"
            f"Denial code: {payload.get('denial_code', '')}\n"
            f"Denial source (for retrieval context only): {payload.get('denial_source', '')}\n"
            f"DRG code (for retrieval context only): {payload.get('drg_code', '')}\n"
            f"CARC code (for retrieval context only): {payload.get('carc_code', '')}\n"
            f"Rationale: {payload.get('rationale', '')}\n"
            f"Patient context: {payload.get('patient_context', '')}\n"
            f"Auto context summary: {context_summary}\n\n"
            f"Policy excerpts:\n{policy_excerpts}"
        )
        letter, provider_used = self._call_llm_provider(llm_provider, user_text)
        guarded_letter, guardrail_reason = self._enforce_letter_guardrails(letter)
        caveat_notes = str(payload.get("caveat_notes", ""))
        caveat_warning = self._contains_false_positive_caveat(caveat_notes)
        has_patient_context = bool(str(payload.get("patient_context", "")).strip())
        context_snapshot = {
            "primary_reason_source": "request_payload",
            "supporting_context_source": "patient_context" if has_patient_context else "none",
            "requested_provider": llm_provider,
            "retrieval_query": query,
            "retrieval_keys": {
                "denial_source": str(payload.get("denial_source", "")),
                "drg_code": str(payload.get("drg_code", "")),
                "carc_code": str(payload.get("carc_code", "")),
            },
            "denial_source": str(payload.get("denial_source", "")),
            "drg_code": str(payload.get("drg_code", "")),
            "carc_code": str(payload.get("carc_code", "")),
            "caveat_notes": caveat_notes,
            "caveat_warning": "true" if caveat_warning else "false",
        }
        if guarded_letter.strip() and not guarded_letter.startswith("(LLM unavailable"):
            fallback_reason = guardrail_reason
            return (
                guarded_letter,
                policy_excerpts,
                f"{provider_used}+rag",
                llm_provider,
                provider_used,
                fallback_reason,
                context_summary,
                context_snapshot,
            )

        letter = (
            "Subject: Request for Reconsideration of Medicare Denial\n\n"
            "To Claims Review Department,\n\n"
            f"We are requesting reconsideration for denial code {payload.get('denial_code', 'N/A')} "
            f"regarding {payload.get('denial_reason', 'this claim')}. "
            "The submitted claim documentation supports medical necessity and coverage criteria under "
            "applicable Medicare policy.\n\n"
            f"Clinical/administrative rationale: {payload.get('rationale', 'See attached records.')}\n\n"
            "Requested action: Reprocess and overturn the denial based on attached records and policy support.\n\n"
            "Sincerely,\nProvider Appeals Team"
        )
        fallback_reason = f"Requested provider '{llm_provider}' unavailable; used template fallback."
        return (
            letter,
            policy_excerpts,
            "template",
            llm_provider,
            "template",
            fallback_reason,
            context_summary,
            context_snapshot,
        )

    def _build_evidence_checklist(
        self,
        classification: ClassificationResult,
        patient_context: str,
        policy_excerpts: str,
    ) -> list[str]:
        reason = classification.denial_reason.lower()
        checklist = [
            "Attach physician signed clinical summary with diagnosis-to-procedure linkage.",
            "Attach supporting records (labs/imaging/progress notes) for dates of service.",
            "Include claim identifiers, denial code, and requested action in cover note.",
        ]
        if "authorization" in reason:
            checklist.append("Attach prior authorization number, approval window, and referral records.")
        if "medical necessity" in reason:
            checklist.append("Map chart evidence directly to LCD/NCD medical necessity criteria.")
        if "non-covered" in reason or "coding" in reason:
            checklist.append("Include corrected coding rationale (HCPCS/CPT/diagnosis alignment and modifiers).")
        if "mue" in reason:
            checklist.append("Justify units with operative/procedure detail and same-day service documentation.")
        if patient_context.strip():
            checklist.append("Reference patient-specific timeline and prior treatment context in appeal body.")
        if policy_excerpts.strip():
            checklist.append("Cite relevant policy excerpt IDs and quote key coverage language.")
        return checklist

    def auto_appeal_from_letter(
        self,
        data: bytes,
        filename: str,
        patient_context: str = "",
        llm_provider: str = DEFAULT_LLM_PROVIDER,
        denial_source: str = "",
        drg_code: str = "",
        carc_code: str = "",
        caveat_notes: str = "",
    ) -> AutoAppealFromLetterResponse:
        classification = self.classify_upload(data=data, filename=filename)
        auto_context = self._build_auto_context(
            denial_type=classification.denial_type,
            denial_reason=classification.denial_reason,
            denial_code=classification.denial_code,
            rationale=classification.rationale,
            patient_context=patient_context,
            extracted_preview=classification.extracted_text_preview,
        )
        auto_context = self._llm_summarize_context(llm_provider, auto_context)
        payload = {
            "denial_type": classification.denial_type,
            "denial_reason": classification.denial_reason,
            "denial_code": classification.denial_code,
            "rationale": classification.rationale,
            "patient_context": patient_context or auto_context,
            "denial_source": denial_source,
            "drg_code": drg_code,
            "carc_code": carc_code,
            "caveat_notes": caveat_notes,
            "llm_provider": llm_provider,
        }
        (
            letter,
            policy_excerpts,
            generator,
            provider_requested,
            provider_used,
            fallback_reason,
            context_summary,
            context_snapshot,
        ) = self.generate_appeal(payload)
        checklist = self._build_evidence_checklist(
            classification=classification,
            patient_context=patient_context or context_summary,
            policy_excerpts=policy_excerpts,
        )
        context_snapshot["primary_reason_source"] = "denial_letter"
        context_snapshot["primary_denial_reason"] = classification.denial_reason
        context_snapshot["primary_denial_code"] = classification.denial_code
        if patient_context.strip():
            context_snapshot["supporting_context_source"] = "csv_context"
        else:
            context_snapshot["supporting_context_source"] = "none"
        context_snapshot["caveat_notes"] = caveat_notes
        context_snapshot["caveat_warning"] = "true" if self._contains_false_positive_caveat(caveat_notes) else "false"
        return AutoAppealFromLetterResponse(
            classification=classification,
            appeal_letter=letter,
            evidence_checklist=checklist,
            policy_excerpts=policy_excerpts,
            generator=generator,
            provider_requested=provider_requested,
            provider_used=provider_used,
            fallback_reason=fallback_reason,
            context_summary=context_summary,
            context_snapshot=context_snapshot,
        )

    def _voice_session_or_raise(self, call_id: str) -> dict[str, Any]:
        session = self._voice_call_sessions.get(call_id)
        if session is None:
            raise ValueError(f"Unknown call_id '{call_id}'")
        return session

    def _build_voice_retrieval_query(self, claim_context: dict[str, Any], payer_text: str) -> str:
        claim = claim_context.get("claim", {}) if isinstance(claim_context, dict) else {}
        ivr = claim_context.get("ivr_status", {}) if isinstance(claim_context, dict) else {}
        classifier = claim_context.get("classifier", {}) if isinstance(claim_context, dict) else {}
        pieces = [
            str(classifier.get("denial_source", "")).strip(),
            str(claim.get("drg_code", "")).strip(),
            str(ivr.get("carc_code", "")).strip(),
            str(claim.get("drg_description", "")).strip(),
            payer_text.strip(),
        ]
        return " ".join([p for p in pieces if p])

    def _synthesize_elevenlabs_tts_bytes(self, text: str) -> bytes:
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is not configured")
        try:
            from elevenlabs import ElevenLabs as ElevenLabsClient
        except Exception as exc:
            raise ValueError(f"elevenlabs package unavailable: {exc}") from exc

        voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "iP95p4xoKVk53GoZ742B")
        model_id = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
        client = ElevenLabsClient(api_key=api_key)
        chunks = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format="mp3_44100_128",
        )
        return b"".join(chunks)

    def _synthesize_tts_base64(self, text: str, tts_backend: str) -> tuple[str, str, str]:
        """Returns (audio_base64, error_message, mime_type).

        ElevenLabs is the primary backend; edge-tts is a zero-config fallback.
        """
        backend = (tts_backend or "elevenlabs").strip().lower()
        if backend not in SUPPORTED_TTS_BACKENDS:
            backend = "elevenlabs"

        try:
            if backend == "elevenlabs":
                audio_bytes = self._synthesize_elevenlabs_tts_bytes(text=text)
                return base64.b64encode(audio_bytes).decode("ascii"), "", "audio/mpeg"
            else:  # edge-tts
                voice_name = os.environ.get("EDGE_TTS_VOICE", "en-US-JennyNeural")
                audio_bytes = asyncio.run(self._synthesize_edge_tts(text=text, voice_name=voice_name))
                return base64.b64encode(audio_bytes).decode("ascii"), "", "audio/mpeg"
        except Exception as exc:
            return "", f"TTS backend '{backend}' unavailable: {exc}", ""

    def _transcribe_with_elevenlabs(self, audio_data: bytes, filename: str) -> str:
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is not configured")

        content_type = "audio/wav"
        lower_name = filename.lower()
        if lower_name.endswith(".mp3"):
            content_type = "audio/mpeg"
        elif lower_name.endswith(".m4a"):
            content_type = "audio/mp4"
        elif lower_name.endswith(".webm"):
            content_type = "audio/webm"

        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": api_key},
            files={"file": (filename, audio_data, content_type)},
            data={"model_id": ELEVENLABS_STT_MODEL},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("No transcription text returned from ElevenLabs STT")
        return text

    def _generate_voice_reply(
        self,
        session: dict[str, Any],
        user_text: str,
        llm_provider: str,
    ) -> tuple[str, str, str]:
        machine: CallStateMachine = session["machine"]
        machine.advance(user_text)
        state_context = machine.get_state_context()
        claim_context = session.get("claim_context", {})
        retrieval_query = self._build_voice_retrieval_query(claim_context=claim_context, payer_text=user_text)
        retrieval = self._rag.retrieve(retrieval_query or "medicare denial call", top_k=2)
        policy_excerpts = self._rag.format_for_prompt(retrieval)
        system_prompt = build_system_prompt(
            scenario=session["scenario"],
            claim_context=claim_context,
            state_context=state_context,
            policy_excerpts=policy_excerpts,
        )
        transcript_lines: list[str] = []
        for turn in session.get("transcript", [])[-6:]:
            transcript_lines.append(f"PAYER: {turn.get('payer', '')}")
            transcript_lines.append(f"PROVIDER: {turn.get('provider', '')}")
        transcript_text = "\n".join(transcript_lines)
        prompt = (
            f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n"
            f"RECENT TRANSCRIPT:\n{transcript_text}\n\n"
            f"LATEST PAYER MESSAGE:\n{user_text}\n\n"
            "Return only the provider-side response to say next."
        )
        reply_text, provider_used = self._call_llm_provider(llm_provider, prompt)
        reply_text = reply_text.strip()
        fallback_reason = ""
        if not reply_text or reply_text.startswith("(LLM unavailable"):
            fallback_reason = f"Provider '{llm_provider}' unavailable; used deterministic fallback response."
            reply_text = (
                "Thank you. I am documenting this update and would like to confirm any additional "
                "documentation required for adjudication or appeal review."
            )

        machine.record_turn(user_input=user_text, response=reply_text)
        session.setdefault("transcript", []).append(
            {
                "state": machine.state_label,
                "payer": user_text,
                "provider": reply_text,
                "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
        )
        session["updated_at"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        session["provider_used_last"] = provider_used
        return reply_text, provider_used, fallback_reason

    def start_voice_call(self, payload: VoiceCallStartRequest) -> VoiceCallTurnResponse:
        scenario = self._normalize_scenario(payload.scenario)
        claim_context = payload.claim_context or self._load_default_claim_context(scenario=scenario)
        call_id = str(uuid.uuid4())
        machine = CallStateMachine(scenario=scenario)
        session: dict[str, Any] = {
            "call_id": call_id,
            "scenario": scenario,
            "machine": machine,
            "claim_context": claim_context,
            "transcript": [],
            "llm_provider_requested": payload.llm_provider,
            "llm_provider": payload.llm_provider,
            "tts_backend": payload.tts_backend,
            "provider_used_last": "",
            "created_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "updated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        self._voice_call_sessions[call_id] = session

        opener = (
            "Hello, this is Riverside Medical Group calling to review claim status and denial details."
            if scenario == "2"
            else "Hello, this is Riverside Medical Group calling to confirm receipt of a written appeal submission."
        )
        reply_text, provider_used, fallback_reason = self._generate_voice_reply(
            session=session,
            user_text=opener,
            llm_provider=payload.llm_provider,
        )
        audio_base64, tts_reason, mime_type = self._synthesize_tts_base64(reply_text, payload.tts_backend)
        merged_reason = "; ".join([x for x in [fallback_reason, tts_reason] if x])
        return VoiceCallTurnResponse(
            call_id=call_id,
            current_state=machine.state_label,
            user_text=opener,
            assistant_text=reply_text,
            audio_base64=audio_base64,
            mime_type=mime_type or "audio/mpeg",
            fallback_reason=merged_reason,
            transcript_turns=len(session.get("transcript", [])),
        )

    def process_voice_call_turn(
        self,
        call_id: str,
        user_text: str,
        llm_provider: str | None,
        tts_backend: str | None,
    ) -> VoiceCallTurnResponse:
        session = self._voice_session_or_raise(call_id)
        resolved_provider = llm_provider or str(session.get("llm_provider", "deepseek"))
        resolved_tts_backend = tts_backend or str(session.get("tts_backend", "elevenlabs"))
        reply_text, _provider_used, fallback_reason = self._generate_voice_reply(
            session=session,
            user_text=user_text,
            llm_provider=resolved_provider,
        )
        session["llm_provider"] = resolved_provider
        session["tts_backend"] = resolved_tts_backend
        audio_base64, tts_reason, mime_type = self._synthesize_tts_base64(reply_text, resolved_tts_backend)
        merged_reason = "; ".join([x for x in [fallback_reason, tts_reason] if x])
        machine: CallStateMachine = session["machine"]
        return VoiceCallTurnResponse(
            call_id=call_id,
            current_state=machine.state_label,
            user_text=user_text,
            assistant_text=reply_text,
            audio_base64=audio_base64,
            mime_type=mime_type or "audio/mpeg",
            fallback_reason=merged_reason,
            transcript_turns=len(session.get("transcript", [])),
        )

    def process_voice_call_audio_turn(
        self,
        call_id: str,
        audio_data: bytes,
        filename: str,
        llm_provider: str | None,
        tts_backend: str | None,
    ) -> VoiceCallTurnResponse:
        session = self._voice_session_or_raise(call_id)
        user_text = self._transcribe_with_elevenlabs(audio_data=audio_data, filename=filename)
        return self.process_voice_call_turn(
            call_id=call_id,
            user_text=user_text,
            llm_provider=llm_provider,
            tts_backend=tts_backend,
        )

    def get_voice_call_summary(self, call_id: str, llm_provider: str) -> VoiceCallSummaryResponse:
        session = self._voice_session_or_raise(call_id)
        transcript = session.get("transcript", [])
        transcript_text = "\n".join(
            [f"[{t.get('state')}] PAYER: {t.get('payer')} | PROVIDER: {t.get('provider')}" for t in transcript]
        )

        lower_text = transcript_text.lower()
        status_case = "in_progress"
        if any(x in lower_text for x in ["denied", "denial"]):
            status_case = "denied_or_pending_appeal"
        if any(x in lower_text for x in ["appeal tracking", "tracking number", "rn-"]):
            status_case = "appeal_tracking_confirmed"
        if any(x in lower_text for x in ["approved", "paid"]):
            status_case = "approved"

        missing_information: list[str] = []
        if "carc" not in lower_text:
            missing_information.append("Confirm CARC denial code")
        if "rarc" not in lower_text:
            missing_information.append("Confirm RARC denial code")
        if "reference" not in lower_text and "rn-" not in lower_text:
            missing_information.append("Obtain payer call reference number")
        if "document" not in lower_text and "records" not in lower_text:
            missing_information.append("Confirm required supporting documentation list")

        next_actions = [
            "Document call details in RCM note",
            "Attach payer reference/tracking IDs to the claim file",
            "Collect missing clinical or coding documentation before follow-up",
        ]

        summary_prompt = (
            "Summarize this payer call for physician/RCM handoff in 3-5 concise sentences. "
            "Include current status, key denial/appeal details, and what information is still needed.\n\n"
            f"Transcript:\n{transcript_text}"
        )
        summary_text, provider_used = self._call_llm_provider(llm_provider, summary_prompt)
        if not summary_text.strip() or summary_text.startswith("(LLM unavailable"):
            provider_used = "template"
            summary_text = (
                "Call completed with payer interaction captured in transcript. "
                "Review denial/appeal identifiers, verify documentation requirements, and proceed with the next appeal step."
            )

        machine: CallStateMachine = session["machine"]
        excerpt = transcript[-6:] if len(transcript) > 6 else transcript
        return VoiceCallSummaryResponse(
            call_id=call_id,
            scenario=session.get("scenario", "2"),
            current_state=machine.state_label,
            status_case=status_case,
            key_summary=summary_text.strip(),
            missing_information=missing_information,
            next_actions=next_actions,
            transcript_excerpt=excerpt,
            llm_provider_requested=llm_provider,
            llm_provider_used=provider_used,
        )

    async def _synthesize_edge_tts(self, text: str, voice_name: str) -> bytes:
        import edge_tts

        communicator = edge_tts.Communicate(text=text, voice=voice_name)
        audio_chunks: list[bytes] = []
        async for event in communicator.stream():
            if event["type"] == "audio":
                audio_chunks.append(event["data"])
        return b"".join(audio_chunks)

    def voice_explain(self, text: str, voice_name: str) -> str:
        audio_bytes = asyncio.run(self._synthesize_edge_tts(text=text, voice_name=voice_name))
        return base64.b64encode(audio_bytes).decode("ascii")
