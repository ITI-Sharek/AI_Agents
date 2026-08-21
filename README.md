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
export AI_PROVIDER="alibaba"
export ALIBABA_API_KEY="your-model-studio-key"
export ALIBABA_BASE_URL="https://your-workspace-id.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
export ALIBABA_MODEL="qwen3.7-plus"
export AI_SERVICE_AUTH_TOKEN="a-long-random-internal-token"
PYTHONPATH=src uvicorn sharek_agents.main:app --reload --port 8010
```

For the temporary local material-analysis compatibility path, enable it before
starting FastAPI:

```bash
export MATERIAL_ANALYSIS_DEV_MODE=true
```

It accepts the NestJS `material-draft-v1` request with `contentBase64`, parses
Markdown, DOCX, text, and text-based PDF materials locally, and returns the
NestJS material-suggestion response shape. This adapter is disabled by default
and is not a production material-analysis contract.

Configure the NestJS backend with the same `AI_SERVICE_AUTH_TOKEN` and with
`AI_SERVICE_URL=http://localhost:8010`.

`AI_PROVIDER=alibaba` reuses the existing OpenAI-compatible LangChain client;
no Alibaba-specific SDK is required. The API key and base URL must belong to
the same Singapore workspace. Keep secrets only in environment configuration,
never in source control. Set `AI_PROVIDER=openrouter` with
`OPENROUTER_API_KEY` and `OPENROUTER_MODEL` to retain OpenRouter, or set
`AI_PROVIDER=groq` with `GROQ_API_KEY` and `GROQ_MODEL` for legacy Groq
environments.

The initial Alibaba rollout uses `qwen3.7-plus` for the shared Skill Profiling
and Advisory Fit chat-model factory. `ALIBABA_ENABLE_THINKING=false` is the
default because these workflows require synchronous structured output; thinking
adds latency and output tokens. Change `ALIBABA_MODEL` only after running the
same structured-output contract checks against another model. Document
Understanding keeps its separate chat and embedding configuration; this
rollout does not replace its embeddings or request-scoped vector store.

Missing credentials and incompatible Alibaba base URLs fail before a network
call. Provider authentication, invalid-model, quota, rate-limit, network, and
context-limit failures stay behind the existing sanitized provider error
boundary. The existing request timeout remains the separate timeout boundary.

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
