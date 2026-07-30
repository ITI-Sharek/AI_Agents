# Share-k AI Agents

FastAPI service for evidence-backed Share-k AI workflows. The NestJS backend
owns authentication, business state, and final decisions. This service returns
structured recommendations only.

## Product Documentation

Shared product context, accepted decisions, and sprint plans live in the
canonical [Share-k Documentation](https://github.com/ITI-Sharek/Documentation)
repository. See `docs/SHARED-PRODUCT-DOCS.md` for the local workspace path and
reading order.

## Local Run

```bash
python3 -m pip install -r requirements.txt
export GROQ_API_KEY="your-rotated-groq-key"
export AI_SERVICE_AUTH_TOKEN="a-long-random-internal-token"
export LLM_PROVIDER="groq"
export LLM_MODEL="openai/gpt-oss-120b"
PYTHONPATH=src uvicorn sharek_agents.main:app --reload --port 8010
```

Configure the NestJS backend with the same `AI_SERVICE_AUTH_TOKEN` and with
`AI_SERVICE_URL=http://localhost:8010`.

## Skill Profiling Contract

`POST /skill-profiles/generate` requires an internal bearer token. It accepts
at most 10 backend-selected repository evidence capsules. Each capsule includes
contributor-specific authorship signals. Generated skills must cite exact
`evidenceId` values from the request; unmatched citations are discarded and the
NestJS adapter validates them again.

NestJS sends `role: "contributor"` for contributor profile generation. The
schema defaults omitted legacy roles to `contributor`; an unavailable GitHub
login remains valid evidence of missing attribution rather than causing HTTP
422.

Repository-wide languages, stars, or commits are not treated as contributor
authorship. Missing attributable activity produces weak evidence and a
`needs_more_evidence` recommendation.

Weak evidence is a valid structured response. Provider, timeout, or malformed
model-output failures surface as service errors so the backend BullMQ worker can
retry them and eventually record a safe failure state.

## Tests

```bash
PYTHONPATH=src python3 -m pytest
python3 -m compileall src
```
