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
export AI_SERVICE_AUTH_TOKEN="a-long-random-internal-token"
export GETAWAY_ITI_KEY="your-student-gateway-key"
export GETAWAY_BASE_URL="http://apiaccess.iti.net.eg"
export GETAWAY_MODEL="antropic.claude-sonnet-4.6"
export AI_ADVISORY_FIT_MAX_RETRIES="1"
PYTHONPATH=src uvicorn sharek_agents.main:app --reload --port 8010
```

Configure the NestJS backend with the same `AI_SERVICE_AUTH_TOKEN` and with
`AI_SERVICE_URL=http://localhost:8010`.

## Advisory Fit Contract

`POST /advisory-fit/assess` is an internal, bearer-authenticated endpoint. It
accepts the backend's immutable Requirement and authorized Evidence Snapshots:
`assessmentRequestId`, `requirements`, `evidence`, `allowedEvidenceIds`,
`requestedAt`, and `contractVersion: "advisory-fit-v1"`.

Completed responses contain exactly one finding per Requirement. Findings use
`SUPPORTED`, `PARTIALLY_SUPPORTED`, `NOT_EVIDENCED`, or `INCONCLUSIVE`, with
categorical confidence, allowed evidence citations, uncertainty, and a concise
explanation. Technical metadata is returned under `metadata`.

The AI service never returns a score, Fit Band, eligibility result, ranking,
recommendation, Application status, or owner decision. NestJS validates the
findings and derives the Fit Band. Empty `allowedEvidenceIds` returns
`NOT_STARTED_NO_ASSESSABLE_EVIDENCE` without a provider call; a configured
provider/system safeguard may return `NOT_STARTED_SYSTEM_LIMIT`.

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
