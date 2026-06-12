from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components


def _resolve_api_base_url() -> str:
    default = os.environ.get("API_BASE_URL", "http://localhost:8000")
    try:
        return st.secrets.get("API_BASE_URL", default)
    except Exception:
        return default


API_BASE_URL = _resolve_api_base_url()
LLM_PROVIDER = "deepseek"
TTS_BACKEND = "elevenlabs"
ROLE_OPTIONS = ["Clinical reviewer", "Coding analyst"]
VOICE_SCENARIO_OPTIONS = {
    "Scenario 2 - Claim Status Inquiry": "2",
    "Scenario 3 - Appeal Receipt Confirmation": "3",
}


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def _get_json(path: str) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=120)
    response.raise_for_status()
    return response.json()


def _post_multipart(
    path: str,
    file_name: str,
    content: bytes,
    mime_type: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    files = {"file": (file_name, content, mime_type)}
    response = requests.post(f"{API_BASE_URL}{path}", files=files, data=data or {}, timeout=240)
    response.raise_for_status()
    return response.json()


def _build_context_from_claim_row(row: dict[str, Any]) -> str:
    return " | ".join(
        [
            f"Claim ID: {row.get('claim_id', 'N/A')}",
            f"DRG: {row.get('drg_code', 'N/A')} ({row.get('drg_description', '')})",
            f"Primary Dx: {row.get('primary_diagnosis_code', '')} {row.get('primary_diagnosis_description', '')}",
            f"Secondary Dx: {row.get('secondary_diagnosis_code', '')} {row.get('secondary_diagnosis_description', '')}",
            f"Tertiary Dx: {row.get('tertiary_diagnosis_code', '')} {row.get('tertiary_diagnosis_description', '')}",
            f"CARC/RARC: {row.get('carc_code', '')}/{row.get('rarc_code', '')}",
            f"LOS days: {row.get('length_of_stay_days', '')}",
            f"Caveat: {row.get('caveat_notes', '')}",
        ]
    )


def _contains_false_positive(text: str) -> bool:
    return bool(re.search(r"\bfalse[\s-]*positive\b", text or "", flags=re.IGNORECASE))


def _panel_order(role: str) -> list[str]:
    if role == "Coding analyst":
        return ["classification", "context_summary", "appeal", "evidence", "policy", "context_snapshot"]
    return ["appeal", "context_summary", "policy", "evidence", "context_snapshot", "classification"]


def _append_voice_turn(resp: dict[str, Any]) -> None:
    st.session_state.voice_transcript.append(
        {
            "state": resp.get("current_state", ""),
            "payer": resp.get("user_text", ""),
            "provider": resp.get("assistant_text", ""),
            "fallback": resp.get("fallback_reason", ""),
        }
    )
    st.session_state.voice_last_response = resp


def _play_audio_if_present(resp: dict[str, Any]) -> None:
    audio_b64 = str(resp.get("audio_base64", "")).strip()
    if not audio_b64:
        return
    mime = str(resp.get("mime_type", "audio/mpeg")) or "audio/mpeg"
    try:
        st.audio(base64.b64decode(audio_b64), format=mime)
    except Exception:
        st.warning("Audio was generated but could not be rendered in the browser.")


def _render_appeals_tab(selected_role: str) -> None:
    input_mode = st.radio(
        "Input Mode",
        ["CSV only", "Denial letter only", "CSV + denial letter"],
        horizontal=True,
        key="appeal_input_mode",
    )

    claims_csv = None
    letter_file = None
    if input_mode in {"CSV only", "CSV + denial letter"}:
        claims_csv = st.file_uploader(
            "Upload physician claim CSV (TTS-style supported)",
            type=["csv"],
            key="claims_csv_uploader",
        )
    if input_mode in {"Denial letter only", "CSV + denial letter"}:
        letter_file = st.file_uploader(
            "Upload denial letter (PDF/Image/Text)",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            key="denial_letter_uploader",
        )

    run_clicked = st.button("Generate Appeal Draft", type="primary", key="generate_appeal_btn")
    if run_clicked:
        if input_mode == "CSV only" and claims_csv is None:
            st.error("Upload a claim CSV to continue.")
        elif input_mode == "Denial letter only" and letter_file is None:
            st.error("Upload a denial letter to continue.")
        elif input_mode == "CSV + denial letter" and (claims_csv is None or letter_file is None):
            st.error("Upload both claim CSV and denial letter to continue.")
        else:
            with st.spinner("Analyzing input and generating appeal..."):
                csv_context = ""
                csv_denial_source = ""
                csv_drg_code = ""
                csv_carc_code = ""
                csv_caveat_notes = ""
                csv_schema_valid = True
                csv_denial_reason = "Insufficient claim evidence - manual review needed"
                csv_denial_code = "D-REVIEW-000"
                csv_rationale = "Claim-level review generated from uploaded CSV."
                if claims_csv is not None:
                    st.session_state.score_response = _post_multipart(
                        "/score_claims_csv",
                        claims_csv.name,
                        claims_csv.read(),
                        claims_csv.type or "text/csv",
                    )
                    validation = st.session_state.score_response.get("schema_validation", {})
                    csv_schema_valid = bool(validation.get("is_valid", False))

                    if input_mode == "CSV only" and not csv_schema_valid:
                        st.session_state.appeal_response = {
                            "appeal_letter": "",
                            "provider_requested": LLM_PROVIDER,
                            "provider_used": "none",
                            "fallback_reason": "CSV schema validation failed; appeal generation blocked.",
                            "context_summary": "",
                            "context_snapshot": {"blocked_by": "csv_schema_validation"},
                        }
                        st.warning("Appeal generation blocked due to CSV schema validation errors.")
                        validation_errors = validation.get("blocking_errors", [])
                        if validation_errors:
                            st.error("; ".join(validation_errors))
                        st.stop()

                    scored = st.session_state.score_response.get("results", [])
                    if scored:
                        top = max(scored, key=lambda x: float(x.get("denial_probability") or 0.0))
                        csv_denial_reason = top.get("denial_reason", csv_denial_reason)
                        csv_denial_code = top.get("denial_code", csv_denial_code)
                        csv_rationale = top.get("rationale", csv_rationale)
                    claim_rows = st.session_state.score_response.get("claim_rows", [])
                    if claim_rows and csv_schema_valid:
                        csv_context = _build_context_from_claim_row(claim_rows[0])
                        csv_denial_source = str(claim_rows[0].get("denial_source", ""))
                        csv_drg_code = str(claim_rows[0].get("drg_code", ""))
                        csv_carc_code = str(claim_rows[0].get("carc_code", ""))
                        csv_caveat_notes = str(claim_rows[0].get("caveat_notes", ""))

                if letter_file is not None:
                    st.session_state.appeal_response = _post_multipart(
                        "/auto_appeal_from_letter",
                        letter_file.name,
                        letter_file.read(),
                        letter_file.type or "application/octet-stream",
                        data={
                            "patient_context": csv_context if csv_schema_valid else "",
                            "llm_provider": LLM_PROVIDER,
                            "denial_source": csv_denial_source,
                            "drg_code": csv_drg_code,
                            "carc_code": csv_carc_code,
                            "caveat_notes": csv_caveat_notes,
                        },
                    )
                    st.session_state.classification = st.session_state.appeal_response.get("classification", {})
                else:
                    payload = {
                        "denial_type": "D",
                        "denial_reason": csv_denial_reason,
                        "denial_code": csv_denial_code,
                        "rationale": csv_rationale,
                        "patient_context": csv_context,
                        "denial_source": csv_denial_source,
                        "drg_code": csv_drg_code,
                        "carc_code": csv_carc_code,
                        "caveat_notes": csv_caveat_notes,
                        "llm_provider": LLM_PROVIDER,
                    }
                    st.session_state.appeal_response = _post_json("/generate_appeal", payload)
                    st.session_state.classification = payload

    if st.session_state.score_response:
        st.subheader("CSV Triage Snapshot")
        score_response = st.session_state.score_response
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Rows Total", score_response.get("rows_total", 0))
        metric_col2.metric("Rows Scored", score_response.get("rows_scored", 0))
        metric_col3.metric("Rows Fallback", score_response.get("rows_fallback", 0))
        metric_col4.metric("Rows Rejected", score_response.get("rows_rejected", 0))
        validation = score_response.get("schema_validation", {})
        if validation:
            if validation.get("is_valid", False):
                st.success("Schema validation: passed")
            else:
                st.error("Schema validation: failed")
            with st.expander("Schema validation details"):
                st.json(validation)
        st.dataframe(score_response.get("results", []), use_container_width=True)

    if st.session_state.appeal_response:
        response = st.session_state.appeal_response
        st.subheader("Generated Appeal")

        provider_used = response.get("provider_used", response.get("generator", "unknown"))
        provider_requested = response.get("provider_requested", "unknown")
        st.caption(f"Provider requested: {provider_requested} | Provider used: {provider_used}")
        fallback_reason = response.get("fallback_reason", "")
        if fallback_reason:
            st.warning(f"Fallback: {fallback_reason}")

        context_snapshot = response.get("context_snapshot", {}) or {}
        caveat_notes = str(context_snapshot.get("caveat_notes", ""))
        caveat_warning_flag = str(context_snapshot.get("caveat_warning", "false")).lower() == "true"
        if caveat_warning_flag or _contains_false_positive(caveat_notes):
            st.warning(
                "Caveat detected: this case includes false-positive language and should receive manual clinical review before submission."
            )

        for panel in _panel_order(selected_role):
            if panel == "classification" and st.session_state.classification:
                with st.expander("Classification / denial summary"):
                    st.json(st.session_state.classification)

            if panel == "context_summary" and response.get("context_summary"):
                st.markdown("**Auto-generated Context Summary**")
                st.info(response.get("context_summary"))

            if panel == "appeal":
                st.text_area(
                    "Appeal Letter",
                    value=response.get("appeal_letter", ""),
                    height=280,
                    key="appeal_text_area",
                )

            if panel == "evidence":
                checklist = response.get("evidence_checklist", [])
                if checklist:
                    st.markdown("**Evidence Checklist**")
                    for item in checklist:
                        st.write(f"- {item}")

            if panel == "policy" and response.get("policy_excerpts"):
                with st.expander("Policy excerpts"):
                    st.write(response.get("policy_excerpts"))

            if panel == "context_snapshot" and context_snapshot:
                with st.expander("Context snapshot"):
                    st.json(context_snapshot)

        st.download_button(
            "Export Appeal Response (.json)",
            data=json.dumps(response, indent=2).encode("utf-8"),
            file_name="appeal_response.json",
            mime="application/json",
            key="appeal_download",
        )


def _build_voice_claim_context_from_session() -> dict[str, Any]:
    """Build claim context dict from the appeals workbench session state."""
    context: dict[str, Any] = {}

    score_resp = st.session_state.get("score_response")
    if score_resp:
        claim_rows = score_resp.get("claim_rows", [])
        if claim_rows:
            row = claim_rows[0]
            context["claim"] = {
                "claim_id": row.get("claim_id", ""),
                "drg_code": row.get("drg_code", ""),
                "drg_description": row.get("drg_description", ""),
                "primary_diagnosis_code": row.get("primary_diagnosis_code", ""),
                "primary_diagnosis_description": row.get("primary_diagnosis_description", ""),
                "secondary_diagnosis_code": row.get("secondary_diagnosis_code", ""),
                "carc_code": row.get("carc_code", ""),
                "carc_description": row.get("carc_description", ""),
                "rarc_code": row.get("rarc_code", ""),
                "rarc_description": row.get("rarc_description", ""),
                "length_of_stay_days": row.get("length_of_stay_days", ""),
            }
        results = score_resp.get("results", [])
        if results:
            top = max(results, key=lambda x: float(x.get("denial_probability") or 0.0))
            context["scoring"] = {
                "denial_probability": top.get("denial_probability"),
                "denial_reason": top.get("denial_reason", ""),
                "denial_code": top.get("denial_code", ""),
                "rationale": top.get("rationale", ""),
            }

    classification = st.session_state.get("classification")
    if classification:
        context["denial_summary"] = classification

    appeal_resp = st.session_state.get("appeal_response")
    if appeal_resp:
        letter = str(appeal_resp.get("appeal_letter", ""))
        policy = str(appeal_resp.get("policy_excerpts", ""))
        context["appeal"] = {
            "letter_preview": letter[:600],
            "policy_excerpts": policy[:500],
            "context_summary": str(appeal_resp.get("context_summary", "")),
        }

    return context


def _render_voice_tab() -> None:
    st.subheader("Insurance Voice Agent")
    st.caption(
        "DeepSeek (LLM, Grok backup) · ElevenLabs (TTS + STT) — speak naturally, the agent responds automatically when you stop recording."
    )

    scenario_label = st.selectbox("Scenario", list(VOICE_SCENARIO_OPTIONS.keys()), index=0)
    scenario = VOICE_SCENARIO_OPTIONS[scenario_label]

    # Show whether appeals workbench data is available for the call context
    has_appeals_data = bool(
        st.session_state.get("score_response") or st.session_state.get("appeal_response")
    )
    if has_appeals_data:
        st.success("✓ Appeals data loaded — voice agent will reference this claim context")
    else:
        st.info("Tip: Generate an appeal in the Appeals Workbench tab first to give the agent full claim context.")

    col_start, col_end = st.columns([1, 1])
    if col_start.button("▶ Start / Reset Call", type="primary", key="start_call_btn"):
        try:
            resp = _post_json(
                "/voice_call/start",
                {
                    "scenario": scenario,
                    "llm_provider": LLM_PROVIDER,
                    "tts_backend": TTS_BACKEND,
                    "claim_context": _build_voice_claim_context_from_session(),
                },
            )
            st.session_state.voice_call_id = resp.get("call_id", "")
            st.session_state.voice_transcript = []
            st.session_state.voice_summary = None
            st.session_state.voice_last_audio_hash = None
            st.session_state.voice_turn_key = 0
            st.session_state.voice_last_audio_b64 = resp.get("audio_base64", "")
            st.session_state.voice_last_audio_mime = resp.get("mime_type", "audio/mpeg")
            st.session_state.voice_play_audio = bool(resp.get("audio_base64", ""))
            st.session_state.auto_record_next = bool(resp.get("audio_base64", ""))
            _append_voice_turn(resp)
            st.success(f"Call started — {st.session_state.voice_call_id}")
        except Exception as exc:
            st.error(f"Unable to start call: {exc}")

    call_id = st.session_state.voice_call_id
    if col_end.button("⏹ End Call + Summary", key="end_call_btn", disabled=not call_id):
        try:
            summary = _get_json(f"/voice_call/{call_id}/summary?llm_provider={LLM_PROVIDER}")
            st.session_state.voice_summary = summary
            st.session_state.voice_call_id = ""
            st.session_state.voice_last_audio_hash = None
            st.rerun()
        except Exception as exc:
            st.error(f"Unable to summarize call: {exc}")

    call_id = st.session_state.voice_call_id

    # Last agent response — always visible above the mic
    if st.session_state.voice_last_response:
        agent_text = st.session_state.voice_last_response.get("assistant_text", "")
        if agent_text:
            st.info(f"**Agent:** {agent_text}")
        # Autoplay the response audio once per turn (flag cleared after rendering)
        if st.session_state.get("voice_play_audio") and st.session_state.get("voice_last_audio_b64"):
            try:
                st.audio(
                    base64.b64decode(st.session_state.voice_last_audio_b64),
                    format=st.session_state.get("voice_last_audio_mime", "audio/mpeg"),
                    autoplay=True,
                )
            except Exception:
                pass
            st.session_state.voice_play_audio = False

    # Always-listening mic — auto-processes when new audio is detected
    if call_id:
        # After each agent reply, auto-click the mic button so the user doesn't have to
        if st.session_state.get("auto_record_next"):
            components.html(
                """<script>
(function(){
  setTimeout(function(){
    var n=0,t=setInterval(function(){
      if(++n>8){clearInterval(t);return;}
      try{
        var b=window.parent.document.querySelector('[data-testid="stAudioInput"] button');
        if(b&&!b.disabled){b.click();clearInterval(t);}
      }catch(e){}
    },500);
  },2000);
})();
</script>""",
                height=0,
            )
            st.session_state.auto_record_next = False
        st.success("🎤 Listening — speak, then stop recording. The agent replies automatically.")
        audio_clip = st.audio_input(
            "Your turn",
            key=f"voice_audio_{st.session_state.voice_turn_key}",
        )

        if audio_clip is not None:
            audio_bytes = audio_clip.read()
            audio_hash = hashlib.md5(audio_bytes).hexdigest()
            if audio_hash != st.session_state.voice_last_audio_hash:
                st.session_state.voice_last_audio_hash = audio_hash
                with st.spinner("Agent is responding..."):
                    try:
                        resp = _post_multipart(
                            f"/voice_call/{call_id}/turn_audio",
                            getattr(audio_clip, "name", "voice_input.wav"),
                            audio_bytes,
                            getattr(audio_clip, "type", "audio/wav"),
                            data={"llm_provider": LLM_PROVIDER, "tts_backend": TTS_BACKEND},
                        )
                        _append_voice_turn(resp)
                        if resp.get("fallback_reason"):
                            st.warning(resp.get("fallback_reason"))
                        st.session_state.voice_last_audio_b64 = resp.get("audio_base64", "")
                        st.session_state.voice_last_audio_mime = resp.get("mime_type", "audio/mpeg")
                        st.session_state.voice_play_audio = bool(resp.get("audio_base64", ""))
                        st.session_state.auto_record_next = True
                        st.session_state.voice_turn_key += 1
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Turn failed: {exc}")

        with st.expander("Or type a message instead"):
            typed_text = st.text_input("Type your message", key=f"voice_text_{st.session_state.voice_turn_key}")
            if st.button("Send", key=f"send_text_{st.session_state.voice_turn_key}") and typed_text.strip():
                with st.spinner("Agent is responding..."):
                    try:
                        resp = _post_json(
                            f"/voice_call/{call_id}/turn",
                            {"user_text": typed_text, "llm_provider": LLM_PROVIDER, "tts_backend": TTS_BACKEND},
                        )
                        _append_voice_turn(resp)
                        if resp.get("fallback_reason"):
                            st.warning(resp.get("fallback_reason"))
                        st.session_state.voice_last_audio_b64 = resp.get("audio_base64", "")
                        st.session_state.voice_last_audio_mime = resp.get("mime_type", "audio/mpeg")
                        st.session_state.voice_play_audio = bool(resp.get("audio_base64", ""))
                        st.session_state.auto_record_next = True
                        st.session_state.voice_turn_key += 1
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Text turn failed: {exc}")
    else:
        st.info("Press ▶ Start / Reset Call to begin. The mic opens automatically.")

    # Transcript (collapsed by default to keep focus on the mic)
    if st.session_state.voice_transcript:
        with st.expander(f"Call transcript ({len(st.session_state.voice_transcript)} turns)", expanded=False):
            for i, turn in enumerate(st.session_state.voice_transcript, start=1):
                st.markdown(f"**Turn {i} — {turn.get('state', '')}**")
                st.markdown(f"- **You:** {turn.get('payer', '')}")
                st.markdown(f"- **Agent:** {turn.get('provider', '')}")
                if turn.get("fallback"):
                    st.caption(f"Fallback: {turn.get('fallback')}")
                st.divider()

    # Structured call notes (shown after End Call)
    if st.session_state.voice_summary:
        summary = st.session_state.voice_summary
        st.markdown("### Structured Call Notes")
        st.json(summary)
        st.download_button(
            "Download Call Notes (.json)",
            data=json.dumps(summary, indent=2).encode("utf-8"),
            file_name=f"voice_call_summary_{summary.get('call_id', 'unknown')}.json",
            mime="application/json",
            key="voice_summary_download",
        )


st.set_page_config(page_title="MedClaimly Appeals + Voice Workbench", layout="wide")
st.title("MedClaimly Appeals + Voice Workbench")
st.caption("Appeals generation and payer-call voice workflow for physician/RCM follow-up")

if "score_response" not in st.session_state:
    st.session_state.score_response = None
if "appeal_response" not in st.session_state:
    st.session_state.appeal_response = None
if "classification" not in st.session_state:
    st.session_state.classification = None
if "voice_call_id" not in st.session_state:
    st.session_state.voice_call_id = ""
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = []
if "voice_summary" not in st.session_state:
    st.session_state.voice_summary = None
if "voice_last_response" not in st.session_state:
    st.session_state.voice_last_response = None
if "voice_last_audio_hash" not in st.session_state:
    st.session_state.voice_last_audio_hash = None
if "voice_turn_key" not in st.session_state:
    st.session_state.voice_turn_key = 0
if "voice_last_audio_b64" not in st.session_state:
    st.session_state.voice_last_audio_b64 = ""
if "voice_last_audio_mime" not in st.session_state:
    st.session_state.voice_last_audio_mime = "audio/mpeg"
if "voice_play_audio" not in st.session_state:
    st.session_state.voice_play_audio = False
if "auto_record_next" not in st.session_state:
    st.session_state.auto_record_next = False

with st.sidebar:
    st.subheader("Settings")
    st.info("LLM: DeepSeek (Grok backup)  |  TTS/STT: ElevenLabs")
    selected_role = st.selectbox("Appeals viewer role", ROLE_OPTIONS, index=0)
    st.markdown("Provider Health")
    try:
        health = _get_json("/llm_provider_health")
        for entry in health.get("providers", []):
            label = entry.get("provider", "unknown")
            healthy = entry.get("healthy", False)
            configured = entry.get("configured", False)
            if healthy:
                st.success(f"{label}: healthy")
            elif configured:
                st.warning(f"{label}: configured but unavailable")
            else:
                st.error(f"{label}: missing config")
    except Exception:
        st.warning("Provider health endpoint unavailable")
    st.markdown("API endpoint")
    st.code(API_BASE_URL)

appeal_tab, voice_tab = st.tabs(["Appeals Workbench", "Insurance Voice Agent"])
with appeal_tab:
    _render_appeals_tab(selected_role=selected_role)
with voice_tab:
    _render_voice_tab()

with st.expander("Session payload"):
    st.code(
        json.dumps(
            {
                "has_scores": st.session_state.score_response is not None,
                "has_appeal": st.session_state.appeal_response is not None,
                "has_classification": st.session_state.classification is not None,
                "voice_call_id": st.session_state.voice_call_id,
                "voice_turns": len(st.session_state.voice_transcript),
                "has_voice_summary": st.session_state.voice_summary is not None,
            },
            indent=2,
        )
    )
