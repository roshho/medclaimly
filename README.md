# MedClaimly — Denial-Appeals Automation

MedClaimly predicts which Medicare claims are likely to be **denied**, explains *why*, and
then drafts the appeal — by letter and by phone.

A two-stage model (deterministic rules → XGBoost) flags at-risk claims and surfaces the
anomalous parameters driving the risk. Those signals are fed to a general-purpose LLM that
generates an editable appeal letter (grounded in retrieved CMS policy), plus a voice agent
that can run the payer phone call end to end.

> Trained on CMS DE-SynPUF synthetic Medicare data. The classifier is a **pre-emptive
> denial-risk proxy**, not an adjudication verdict. See [notes.md](notes.md) for the full
> research log, metrics, and limitations.

## Architecture

```
                ┌──────────────────────── Offline pipeline ────────────────────────┐
 CMS / SynPUF → │ Stage 1 rules engine (hard-deny + pass-through)                   │
   raw claims   │      └→ Stage 2 XGBoost (denial probability + thresholded flag)   │
                │            └→ joined results parquet + performance report          │
                └──────────────────────────────────────────────────────────────────┘
                                              │  model.json + reference tables
                                              ▼
 Upload CSV / denial letter ─▶ FastAPI backend (src/demo_api)
                                  ├─ score_claims_csv ............ XGBoost risk scoring
                                  ├─ classify / auto_appeal ...... denial classification
                                  ├─ Policy RAG (rules/policy_corpus) ... grounding
                                  ├─ LLM: DeepSeek (primary) → Grok backup ... drafting
                                  └─ Voice: ElevenLabs STT/TTS + call state machine
                                              │
                                              ▼
                          Streamlit workbench (src/demo_app): appeals + voice agent
```

## Providers

| Role            | Primary                  | Backup / fallback        | Env var |
|-----------------|--------------------------|--------------------------|---------|
| Appeal/voice LLM| DeepSeek (`deepseek-chat`) | Grok / x.ai (`grok-2-latest`) | `DEEPSEEK_API_KEY`, `GROK_API_KEY` |
| Voice STT + TTS | ElevenLabs               | edge-tts (TTS, zero-config) | `ELEVENLABS_API_KEY` |

DeepSeek and Grok both speak the OpenAI-compatible chat API, so a single client wrapper
([src/TTS/llm_client.py](src/TTS/llm_client.py)) covers both. If DeepSeek is unavailable the
backend automatically fails over to Grok, then to a deterministic template.

## Quick start (demo)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in your keys
set -a && source .env && set +a

# Terminal 1 — API
uvicorn src.demo_api.app:app --reload --port 8000

# Terminal 2 — UI
streamlit run src/demo_app/streamlit_app.py
```

The demo runs against the committed Sample-1 model artifacts in `output/`
(`sample01_stage2_model.json`, metrics, reference tables) — no dataset download required.
Demo fixtures live in [demo_data/synthetic_denials.json](demo_data/synthetic_denials.json).

### Key endpoints
- `POST /score_claims_csv` — claim CSV → XGBoost denial probability + risk signals per row
- `POST /auto_appeal_from_letter` — denial letter → classification + drafted appeal + evidence checklist
- `POST /generate_appeal` — structured denial context → policy-grounded appeal letter
- `POST /voice_call/start` · `/voice_call/{id}/turn[_audio]` · `/voice_call/{id}/summary` — voice agent
- `GET  /llm_provider_health` — DeepSeek / Grok / ElevenLabs configuration status

## Rebuilding the model (optional)

The training pipeline needs the raw CMS DE-SynPUF samples under `dataset/CMS/` (gitignored).
End-to-end run for one sample:

```bash
python src/src_CMS/run_pipeline.py --sample-id 1
```

Stages: `load_reference_tables` → `stage1_rules_engine` → `stage2_prepare_training` →
`stage2_feature_engineering` → `stage2_train_xgboost` → `generate_performance_report`, then a
DuckDB join into `output/sample01_full_pipeline_results.parquet`.

Build the LCD/NCD policy corpus used by the RAG layer:

```bash
python src/src_CMS/build_policy_corpus.py            # full
python src/src_CMS/build_policy_corpus.py --max-records 20   # quick preview
```

## Layout

```
src/src_CMS/     Offline two-stage pipeline (rules + XGBoost) and policy-corpus builder
src/demo_api/    FastAPI backend: scoring, classification, appeal generation, voice agent
src/demo_app/    Streamlit appeals + voice workbench
src/TTS/         LLM clients, call state machine, policy RAG, voice scenarios
output/          Committed model artifacts + reference tables (large parquets gitignored)
tests/           Unit tests for service guardrails
notes.md         Research log, dataset survey, pipeline metrics & limitations
```

## Tests

```bash
python -m unittest discover -s tests
```

## License

See [LICENSE](LICENSE). Built on CMS DE-SynPUF synthetic data; not for clinical or billing use.
