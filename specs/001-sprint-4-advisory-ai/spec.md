# Sprint 4 AI — Advisory Fit and Material-Assisted Drafting

## Problem Statement

The AI service currently supports skill-profile generation but does not expose
the accepted Sprint 4 analysis contracts. The superseded product model expected
AI to validate contributor eligibility during Application submission. The
current contract instead requires optional, owner-requested Advisory Fit
Assessments that remain evidence-based and decision-neutral.

The final Sprint 4 stretch outcome also requires document-assisted Draft
Suggestions. That capability must process only explicitly authorized Material
Versions, isolate retrieval by Project and purpose, and return private
suggestions without mutating business state.

## Solution

Add authenticated internal FastAPI capabilities for:

1. Advisory Fit analysis over backend-supplied fixed Requirement and Evidence
   Snapshots.
2. Final-stretch document extraction, bounded retrieval, and Draft Suggestion
   generation over an owner-authorized Analysis Set.

FastAPI returns strict structured analysis only. It does not receive authority
to change an Application, Project, Contribution Request, Owner Decision,
Assignment, Proposal, or Material access rule. NestJS validates the response,
derives the Fit Band, persists attempts and audits, and controls every business
mutation.

## User Stories

1. As a Project owner, I want an Advisory Fit Assessment only after I explicitly request one, so that AI is optional.
2. As a Project owner, I want one Requirement Finding per fixed Requirement, so that the assessment addresses the actual work contract.
3. As a Project owner, I want findings limited to Supported, Partially Supported, Not Evidenced, or Inconclusive, so that model language is bounded.
4. As a Project owner, I want every finding to include authorized evidence citations, so that claims are inspectable.
5. As a Project owner, I want categorical High, Medium, or Low confidence, so that false numerical precision is avoided.
6. As a Project owner, I want uncertainty stated explicitly, so that missing, stale, conflicting, or restricted evidence is visible.
7. As a Project owner, I want a concise explanation for each finding, so that citations have understandable context.
8. As a contributor, I want Not Evidenced to mean only that evidence was absent, so that it is not presented as incapability.
9. As a contributor, I want low confidence and inconclusive findings to remain decision-neutral, so that ambiguity cannot block my Application.
10. As a contributor, I want the model to return no eligibility, pass/fail, ranking, score, acceptance, decline, or Application transition, so that selection remains human.
11. As the NestJS backend, I want the model to cite only supplied evidence identifiers, so that invented sources are rejected.
12. As the NestJS backend, I want the response to cover each supplied Requirement exactly once, so that deterministic Fit Band derivation is possible.
13. As the NestJS backend, I want Preferred Requirements marked separately, so that they cannot affect the Fit Band.
14. As the NestJS backend, I want invalid output to fail closed with an audience-safe error, so that malformed analysis is never persisted as valid.
15. As an operator, I want bounded timeout and retry behavior, so that provider failures do not create unbounded cost or work.
16. As an operator, I want prompt, schema, provider, model, latency, and token metadata returned safely, so that NestJS can preserve an audit snapshot.
17. As a privacy reviewer, I want private evidence content excluded from logs and error messages, so that diagnostics do not leak contributor data.
18. As a Project owner, I want no-assessable-evidence to be detectable without a model call, so that absence of input is not turned into a negative assessment.
19. As a Project owner, I want to select exact Project Material Versions for analysis, so that later replacements do not alter an existing run.
20. As a Project owner, I want upload alone to cause no AI processing, so that consent is explicit.
21. As a Project owner, I want supported DOCX, Markdown, and text-based PDF content extracted without executing embedded instructions, so that documents are treated as untrusted data.
22. As a Project owner, I want images and scanned PDFs to remain assets only, so that unsupported OCR or image inference is not implied.
23. As a Project owner, I want extraction limits enforced before provider calls, so that cost and denial-of-service risk are bounded.
24. As a Project owner, I want document instructions, macros, links, and remote resources ignored, so that prompt injection and active content cannot control the analysis.
25. As a Project owner, I want chunks and embeddings isolated to the selected Project, Analysis Set, and purpose, so that retrieval cannot escape authorization.
26. As a privacy reviewer, I want Material content excluded from discovery, matching, model training, and Advisory Fit evidence, so that consent does not spread across features.
27. As a Project owner, I want Draft Suggestions for approved Project fields and draft Contribution Requests, so that documents can accelerate setup.
28. As a Project owner, I want suggestions to identify their source Material Versions, so that provenance is reviewable.
29. As a Project owner, I want no suggestion to become authoritative automatically, so that NestJS and the owner control adoption.
30. As a Project owner, I want no generated reward, date, assignment, publication state, or authoritative repository language, so that unsupported fields remain human-controlled.
31. As a Project owner, I want invalid model output to produce no partial suggestion set, so that retries are safe.
32. As an operator, I want deletion and re-indexing commands to target exact Material Versions, so that stale embeddings are removable.
33. As an operator, I want provider failures and unsafe input outcomes represented without raw content, so that backend retry and audit policy stays informed.
34. As a developer, I want deterministic contract fixtures and provider stubs, so that the AI boundary is testable without paid model calls.

## Implementation Decisions

- The AI service exposes internal authenticated HTTP contracts. They require the
  existing service bearer-token boundary and are never called directly by a
  browser.
- Advisory Fit input contains an Assessment Request identifier, immutable
  Requirement Snapshot, immutable authorized Evidence Snapshot, allowed
  evidence identifiers, requested timestamp, and contract versions. It does
  not include permission to transition an Application.
- Advisory Fit output contains one Requirement Finding per input Requirement.
  Allowed findings are `SUPPORTED`, `PARTIALLY_SUPPORTED`, `NOT_EVIDENCED`, and
  `INCONCLUSIVE`.
- Confidence is categorical `HIGH`, `MEDIUM`, or `LOW`. Percentage confidence,
  probability, eligibility score, aggregate model score, and rank are rejected
  by the output schema.
- Each finding includes evidence identifiers, explanation, and explicit
  uncertainty. Every cited identifier must occur in the request and every
  Requirement must appear exactly once.
- Preferred and Required classification is preserved in the response, but
  FastAPI does not derive the Fit Band. ADR 0001 assigns deterministic Fit Band
  derivation to NestJS.
- The model is instructed and schema-constrained never to emit an Application
  status, decision, recommendation, acceptance, decline, priority, or
  eligibility verdict.
- Empty assessment-eligible evidence is detected deterministically before a
  provider call and returned through an internal no-assessable-evidence result
  that lets NestJS record `NOT_STARTED_NO_ASSESSABLE_EVIDENCE`.
- Provider invocation uses bounded timeout and at most the approved retry
  count. Invalid citations, duplicate or missing Requirements, unsupported
  enums, and schema violations are invalid output, not partial success.
- Public error responses and logs use safe codes. Raw evidence, private
  repository details, Material content, provider credentials, and prompts with
  user content are not logged.
- Successful responses include safe audit metadata such as prompt version,
  schema version, provider and model identifiers, latency, and token counts
  when available. NestJS decides what to persist.
- Existing skill profiling remains a separate capability and evidence purpose.
  Advisory Fit does not reuse pending, rejected, disputed, unauthorized, or
  live-mutating profile data outside the backend-supplied Evidence Snapshot.
- Material-assisted drafting is a final stretch capability and remains blocked
  until the backend confirms the core Sprint 4 release gate and safe Material
  foundation.
- NestJS owns upload, object storage, malware-scan state, access control,
  entitlements, Material Versions, Analysis Sets, Analysis Runs, deletion
  commands, and final suggestion adoption.
- FastAPI processes only exact Material Versions authorized in one Analysis
  Set. Access uses narrowly scoped, short-lived internal references supplied by
  NestJS; the AI service does not browse storage or Projects generally.
- Supported analysis sources are DOCX, Markdown, and text-based PDF. Images and
  scanned PDFs remain non-analyzed assets. Macros, embedded code, remote links,
  and external resource loading are disabled.
- Document bytes and extracted text are untrusted data. Prompt construction
  clearly separates system instructions from document content and ignores
  instructions contained inside documents.
- Extraction and analysis enforce backend-provided and service-side maximum
  file count, total extracted characters, and supported media types before a
  model call.
- Chunk and embedding records carry exact Project, Analysis Set, Material
  Version, purpose, model, and schema scope. Retrieval requires all scope
  dimensions and cannot fall back to global similarity search.
- PostgreSQL with pgvector remains the accepted embedding direction. NestJS
  owns business persistence and deletion orchestration; FastAPI owns model and
  extraction implementation behind explicit contracts.
- Material analysis output is a complete private Draft Suggestion set or a
  failure. It may suggest Project title, description, technologies, category,
  difficulty, and zero or more draft Contribution Requests with descriptions,
  Requirements, technology tags, and difficulty.
- Output cannot set authoritative repository languages, rewards, dates,
  Proposed Delivery Duration, assignees, Applications, Assignments,
  publication state, or Owner Decisions.
- Every suggestion references the Material Versions that support it and is
  labeled as AI-generated. FastAPI cannot adopt or publish it.
- Re-indexing and deletion address exact Material Versions. Deleted or
  superseded versions cannot remain retrievable for later runs.
- The AI contract remains independent of payment checkout. NestJS supplies an
  already-authorized request after seeded, demo, or admin entitlement checks.

## Testing Decisions

- The highest AI seam is the authenticated FastAPI HTTP contract tested through
  `TestClient`. Tests replace model, embedding, extraction-storage access, and
  other external providers with deterministic fakes.
- Advisory Fit contract tests cover service authentication, complete
  Requirement coverage, exact citation allowlists, confidence enums,
  uncertainty, preferred classification, safe metadata, timeout, retry, and
  invalid-output failure.
- Negative contract tests prove responses cannot contain eligibility,
  Application status, pass/fail, score, rank, recommendation, acceptance, or
  decline fields.
- Evidence privacy tests use private markers and assert they never appear in
  logs, HTTP errors, or safe diagnostics.
- No-assessable-evidence tests prove no model call occurs and no finding or Fit
  Band is fabricated.
- Material extraction tests use small representative DOCX, Markdown, text PDF,
  scanned PDF, malformed, encrypted, macro-bearing, oversized, and
  prompt-injection fixtures.
- Material tests prove unsupported images and scanned PDFs are not analyzed,
  remote resources are never fetched, embedded instructions are ignored, and
  limits apply before provider invocation.
- Retrieval tests create colliding content across Projects, Analysis Sets, and
  Material Versions and prove that only the exact authorized scope is returned.
- Deletion and re-index tests prove removed or superseded Material Versions
  cannot be retrieved and that unrelated scopes remain intact.
- Draft Suggestion tests cover every allowed field, prohibited authoritative
  fields, source attribution, complete-or-fail behavior, and absence of
  mutation side effects.
- Prior art includes the existing FastAPI `TestClient` authentication test,
  deterministic skill-profile provider tests, strict Pydantic contracts,
  evidence citation validation, and retryable provider failure tests.
- Paid or live provider calls are not required for the repository quality gate.
  A separate deployment smoke exercise may verify configured providers after
  deterministic tests pass.

## Out of Scope

- Triggering assessment automatically when an Application is submitted.
- Eligibility, pass/fail, ranking, scoring, matching, acceptance, decline,
  Application visibility, admin eligibility review, or any business-state
  mutation.
- Live profile reads or repository crawling for an assessment. NestJS supplies
  the fixed authorized Evidence Snapshot.
- Project discovery, contributor matching, public Material search, model
  training, or reuse of Material embeddings as Advisory Fit evidence.
- Object-storage ownership, malware scanning, access grants, entitlement
  assignment, Analysis Run persistence, audit persistence, or Draft Suggestion
  adoption.
- OCR, image understanding, scanned-PDF analysis, audio, video, archives,
  executables, macros, or remote document resources.
- Payment checkout or billing-provider integration.
- Shipping the Material analysis endpoint before the complete core and safe
  Material release gates.

## Further Notes

- The authenticated FastAPI HTTP contract is the NestJS-to-AI testing seam.
- Advisory Fit and Material-assisted drafting use separate schemas, prompt
  versions, evidence purposes, and retrieval scopes.
- This specification implements the AI responsibilities in ADRs 0001, 0004,
  and 0005 while preserving the append-only backend audit boundary in ADR 0002.
- No production AI behavior is implemented by this specification issue.
