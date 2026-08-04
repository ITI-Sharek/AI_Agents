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

The skill-profile workflow uses two local processes:

- the AI orchestrator on port `8010`;
- the code-analysis service on port `8000`.

Both processes must receive the same internal bearer token. The orchestrator
reads it as `ANALYSIS_SERVICE_AUTH_TOKEN` when calling the analysis service;
the analysis service validates it as `AI_SERVICE_AUTH_TOKEN`. Each token must
be at least 32 characters long.

Create `ai/.env` from `.env.example` and set both token variables to the same
value. Model-provider configuration is selected by `AI_PROVIDER`:

- `openrouter` (default) requires `OPENROUTER_API_KEY` and optionally
  `OPENROUTER_MODEL`;
- `student-api-gateway` requires `GETAWAY_ITI_KEY`, `GETAWAY_BASE_URL`, and
  `GETAWAY_MODEL`.

The service fails with a clear configuration error when the selected
provider's credential is missing; it never constructs an empty bearer header.
The default model is OpenRouter's `openrouter/free` router so stale individual
free-model IDs do not silently strand background jobs. Provider SDK retries are
disabled; the skill-profile pipeline and BullMQ worker own the bounded retries.

Install the code-analysis service and AI dependencies:

```bash
python3 -m pip install -e ./code-analysis-service
python3 -m pip install -r requirements.txt
docker build -t code-analysis-runner:latest \
  -f code-analysis-service/src/code_analysis_service/Dockerfile.analysis \
  code-analysis-service/src/code_analysis_service
```

Start both local services with one command:

```bash
./run-services.sh
```

The launcher loads `.env` through `dotenv`, so the same command works from
Bash and Fish. It prefers executables from `ai/.venv`, prefixes output with
`[analysis]` or `[ai]`, and shuts down both processes when you press `Ctrl+C`
or when either service exits unexpectedly.

To run the processes separately for debugging, start the code-analysis
service in one terminal:

```bash
dotenv run -- code-analysis-api
```

The runner image is required whenever Docker is available. Without it, Docker's
image-pull error can look like repository authentication failure. Private
repositories remain analyzable from the bounded evidence snapshot supplied by
NestJS when clone-based enrichment is unavailable; GitHub App installation
tokens are not persisted or forwarded across the backend module boundary.

Then start the AI orchestrator in a second terminal:

```bash
dotenv run -- env PYTHONPATH=src uvicorn sharek_agents.main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8010
```

Configure the NestJS backend with the same `AI_SERVICE_AUTH_TOKEN` and with
`AI_SERVICE_URL=http://localhost:8010`.

`dotenv run` is shell-independent and is the recommended way to load `.env`;
it works in Fish as well as Bash. The AI orchestrator sends skill-profile
prompts through the configured provider and validates structured output against
the response schema before returning it to NestJS.

### 2026-08-03 skill-profile gateway fix

The skill-profile endpoint previously returned `502` after successful code
analysis because provider selection and structured-output handling were
inconsistent. A dormant Student API Gateway implementation was present, while
runtime environments could be configured for either OpenRouter or that gateway.

The fix made these implementation changes:

- `common/llm.py` now selects OpenRouter or the Student API Gateway from
  `AI_PROVIDER` and validates that provider's credential before making a call.
- The gateway client has an asynchronous structured-generation operation that
  includes the Pydantic JSON schema in the prompt, requires non-empty
  `output_text`, accepts plain JSON or one surrounding JSON code fence, and
  validates the result before returning it.
- `contract_service.py` calls that operation directly instead of the
  LangChain-only `with_structured_output()` interface.
- The no-evidence guardrail now recognizes authorized NestJS snapshot evidence
  independently from optional framework/static/graph enrichment, allowing
  private repositories to produce a profile without forwarding GitHub App
  installation credentials across module boundaries.
- Skill-profile audit metadata reports the selected provider and its actual
  configured model, including weak-evidence results.
- `main.py` records provider exception tracebacks server-side while preserving
  the contract-safe generic `502` response body.
- `tests/test_skill_profile_security.py` covers provider selection, gateway
  parsing, and the public `/skill-profiles/generate` path through a substituted
  external response. The test does not call GitHub or the analysis service.

Timeout, bounded retry, evidence-citation validation, maximum-skill validation,
and the document-understanding client were not changed.

## Advisory Fit Contract

`POST /advisory-fit/assess` is an internal, bearer-authenticated endpoint. It
accepts the backend's immutable Requirement and authorized Evidence Snapshots:
`assessmentRequestId`, `requirements`, `evidence`, `allowedEvidenceIds`,
`requestedAt`, and `contractVersion: "advisory-fit-v1"`.

The configured provider is called through the shared strict structured-output
adapter with `AdvisoryFitProviderOutput` as the JSON Schema. This prevents
plausible but contract-incompatible field names from being accepted, and the
response metadata records the provider and model selected by `AI_PROVIDER`.

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
