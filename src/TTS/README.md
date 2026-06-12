# Voice + LLM modules

Shared building blocks for the provider-to-payer denial call agent and the appeal-letter
generator. These modules are imported by the FastAPI backend ([src/demo_api](../demo_api));
there is no standalone CLI — drive everything through the API / Streamlit workbench
(see the root [README](../../README.md)).

## Files

| File | Purpose |
|------|---------|
| `llm_client.py` | OpenAI-compatible LLM clients: `DeepSeekClient` (primary), `GrokClient` (backup) + system-prompt builder |
| `state_machine.py` | Call phase tracking (IVR → auth → status → rep → docs → close) |
| `policy_rag.py` | TF-IDF retrieval over the Medicare LCD/NCD policy corpus |
| `sample_claim_context.json` | Default claim / IVR / appeal context for the voice scenarios |
| `synpuf_high_rejection_case.csv` | SynPUF-derived high-risk sample claim |
| `scenario_2_ivr_detailed_denial_info.md` | Reference script: IVR claim-status call |
| `scenario_3_authorization_appeal_confirmation.md` | Reference script: appeal-receipt confirmation |

## How it fits together

Each turn the backend: advances the **state machine**, retrieves relevant policy excerpts
via **policy RAG**, builds a system prompt with claim context + call phase + excerpts, and
calls the **LLM** (DeepSeek, failing over to Grok). Voice STT/TTS is handled by ElevenLabs
in [src/demo_api/services.py](../demo_api/services.py); edge-tts is a zero-config TTS fallback.

The LLM is explicitly instructed never to expose internal model/risk fields (e.g.
`denial_risk_score`, `classifier`) in spoken or written output — `build_system_prompt`
strips them from context and an output guardrail redacts them from drafts.

## Call state machine

**Scenario 2 (claim status):**
`IVR_GREETING → IVR_AUTH_NPI → IVR_CLAIM_LOOKUP → IVR_STATUS_CODES → TRANSFER_TO_REP → REP_DETAIL → REP_DOCUMENTATION → REP_REFERENCE → CALL_CLOSE`

**Scenario 3 (appeal confirmation):**
`IVR_GREETING → IVR_AUTH_NPI → TRANSFER_TO_REP → REP_CONFIRM_RECEIPT → REP_CLINICAL_REVIEW → REP_TRACKING → REP_TIMELINE → CALL_CLOSE`

## Policy RAG

TF-IDF (scikit-learn) over the LCD/NCD corpus at `rules/policy_corpus/docs/` — no embedding
API or model download. Build the corpus first (see root README), then quick-test retrieval:

```bash
python src/TTS/policy_rag.py "ventilator support respiratory failure COPD"
```

## Clinical caveat (sample case)

DRG 207 geometric-mean LOS is 9–12 days. The bundled 7-day stay is shorter than average but
clinically complex (ventilator ≥96 hrs, pulmonary candidiasis, atrial fibrillation). The
synthetic labeling rule flags it as short-stay upcoding — a likely **false positive** that
real-world review would uphold, which is why the model is treated as a risk proxy, not a verdict.
