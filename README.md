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

Repository-wide languages, stars, or commits are not treated as contributor
authorship. Missing attributable activity produces weak evidence and a
`needs_more_evidence` recommendation.

Weak evidence is a valid structured response. Provider, timeout, or malformed
model-output failures surface as service errors so the backend BullMQ worker can
retry them and eventually record a safe failure state.

## Advisory Fit Contract

`POST /advisory-fit/assess` is an internal bearer-token endpoint consumed only
by the NestJS backend. It accepts fixed Requirement and authorized Evidence
snapshots and returns one bounded finding per Requirement. It never returns an
Application decision, eligibility verdict, score, rank, or workflow mutation.

Each evidence item is a strict bounded capsule with `evidenceId`, `type`,
`label`, and an optional bounded summary. The allowlist must exactly match the
unique capsule identifiers; opaque evidence objects and extra fields are
rejected before provider work.

No authorized evidence returns `NOT_STARTED_NO_ASSESSABLE_EVIDENCE` without a
provider call. Provider limits return `NOT_STARTED_SYSTEM_LIMIT`; timeouts and
invalid output fail with safe 504/502 responses. The backend remains the owner
of Fit Band derivation, persistence, and every owner decision.

## Requirement Inference Contract

`POST /requirements/infer` is an internal bearer-token endpoint consumed only by
the NestJS backend. It reads a Contribution Request — title, description,
ordered requirement texts, technology tags, difficulty — and names the technical
skills the work demands, each with the proficiency level it needs.

**It never sees contributor data.** There is no contributor identifier,
approved-skill list, or Application reference anywhere in the request or
response schema, and `extra="forbid"` means a caller that started sending one
gets a 422 rather than quietly handing it to a model. The agent is asked what
the *work* demands; who might do it cannot be made its business without changing
the schema.

It returns **findings, never a verdict** — the same split ADR 0001 set for
Advisory Fit. There is no `eligible`, `blocked`, `score`, or `rank` field and no
place to put one. NestJS derives the eligibility decision itself from these rows
and from its own frozen snapshot (DEC-078, ADR 0015).

Each entry is `{ skillName, requiredLevel, kind, confidence, rationale }`:

- `requiredLevel` is exactly `beginner | intermediate | advanced` — the three
  platform levels, because the backend compares an approved proficiency against
  it using a total order and a fourth level has no defined position in one.
- `confidence` is exactly `high | medium | low`. Never a number or a percentage:
  DEC-010 forbids presenting fit as a number, and a percentage invites the
  reader to treat an inferred level as a measurement.
- `skillName` is returned lowercase and space-collapsed, so the caller receives
  one spelling per skill.

The set is capped at 15 and duplicate names are collapsed keeping the first
occurrence. Both are handled by truncation rather than rejection: the result is
a draft the owner reviews and overrides before publication, and turning a
verbose or sloppy answer into a 502 would leave them with nothing to edit. An
empty set is a valid answer — a Request too vague to imply a skill must not be
made to invent one; requiring at least one `required` row before publication is
the backend's job.

**Untrusted input.** Every part of the Request is owner-written text and is
treated as data. It is JSON-encoded under a labelled heading, so a quote or
newline cannot terminate a field and begin what reads as a new instruction
section; the structured-output schema constrains the answer's shape; and **no
tool is bound to the model**, so an injected "fetch this URL" has no mechanism
to act through. A test asserts the run completes with the schema intact while
any outbound HTTP from that code path raises.

Missing token returns 401; an unconfigured server token returns 503 — an
unconfigured server is not the caller's mistake. A provider timeout returns 504
and invalid model output returns 502, and neither returns a partial set: half a
bar is worse than no bar, because the owner cannot tell it is half. Provider
detail never reaches the response body.

## Tests

```bash
PYTHONPATH=src python3 -m pytest
python3 -m compileall src
```

No test makes a paid provider call. Provider behaviour is exercised through an
injected fake or by monkeypatching `ChatGroq`.
