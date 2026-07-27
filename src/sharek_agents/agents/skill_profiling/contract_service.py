import asyncio
import json
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from sharek_agents.agents.skill_profiling.contract_schemas import (
    DependencyFileRef,
    FraudSignal,
    FrameworkDetectionEvidence,
    GraphRelationsEvidence,
    ModelSkillProfileAnalysis,
    RepositoryEvidenceCapsule,
    SkillProfileInput,
    SkillProfileResult,
    StaticAnalysisEvidence,
)
from sharek_agents.agents.skill_profiling.detection import (
    detect_frameworks as _detect_frameworks,
)
from sharek_agents.agents.skill_profiling.analysis_client import (
    run_step2_analysis as _run_step2_analysis,
)
from sharek_agents.common.llm import get_llm
from sharek_agents.config import settings
from sharek_agents.shared_tools.github_client import GithubClient

"""
NOTE: skill_profiling and project_creation currently share this
workflow and both output a `skills` list only. This is a deliberate,
temporary simplification for the current product phase. Splitting them
into fully separate agents/outputs is expected later once
project-side skill matching requirements are better defined. Do not
assume this merge is permanent when making future changes.
"""

logger = logging.getLogger(__name__)

PROMPT_VERSION = "skill-profile-v2"
SCHEMA_VERSION = "skill-profile-result-v1"
MAX_SKILLS = 20
MAX_EVIDENCE_ID_LOOKUP_RETRIES = 1

_DEPENDENCY_FILE_CANDIDATES = [
    "requirements.txt",
    "pyproject.toml",
    "Pipfile",
    "poetry.lock",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "composer.json",
    "composer.lock",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "go.mod",
    "Cargo.toml",
    "Gemfile",
    "Gemfile.lock",
    "pubspec.yaml",
    "Package.swift",
    "Podfile",
    "Cartfile",
    "packages.config",
    "Directory.Packages.props",
]


async def _run_step1_framework_detection(
    capsule: RepositoryEvidenceCapsule,
    github_pat: str | None = None,
) -> FrameworkDetectionEvidence:
    """Step 1 — Framework/Library/ORM detection via GitHub API only.

    Always runs first and independently of any analysis-service-based analysis.
    Never makes its execution conditional on Step 2's outcome.
    Uses only GitHub REST API to fetch dependency files; no clone,
    no subprocess, no local file reading.
    """
    parts = capsule.full_name.split("/")
    if len(parts) != 2:
        return FrameworkDetectionEvidence(
            frameworks_detected={},
            dependency_files_identified=[],
            frameworks_count=0,
            status="parse_error",
        )
    owner, repo_name = parts

    token = github_pat or settings.github_token
    client = GithubClient(token=token)

    dep_files: dict[str, str] = {}
    dep_refs: list[DependencyFileRef] = []

    contents = await asyncio.gather(
        *(client.get_content(owner, repo_name, path) for path in _DEPENDENCY_FILE_CANDIDATES),
        return_exceptions=True,
    )
    for path, content in zip(_DEPENDENCY_FILE_CANDIDATES, contents):
        if isinstance(content, str):
            dep_files[path] = content
            dep_refs.append(DependencyFileRef(filename=path))

    try:
        default_branch = capsule.default_branch
        tree = await client.get_tree(owner, repo_name, default_branch)
        if tree is not None:
            csproj_candidates = [
                item["path"]
                for item in tree.get("tree", [])
                if item.get("path", "").endswith(".csproj")
            ]
            if csproj_candidates:
                csproj_contents = await asyncio.gather(
                    *(client.get_content(owner, repo_name, f) for f in csproj_candidates[:5]),
                    return_exceptions=True,
                )
                for f, content in zip(csproj_candidates[:5], csproj_contents):
                    if isinstance(content, str):
                        dep_files[f] = content
                        dep_refs.append(DependencyFileRef(filename=f))
    except Exception:
        pass

    if not dep_files:
        return FrameworkDetectionEvidence(
            frameworks_detected={},
            dependency_files_identified=[],
            frameworks_count=0,
            status="no_dependency_files",
        )

    try:
        frameworks = _detect_frameworks(dep_files)
        return FrameworkDetectionEvidence(
            frameworks_detected=frameworks,
            dependency_files_identified=dep_refs,
            frameworks_count=len(frameworks),
            status="success",
        )
    except Exception:
        return FrameworkDetectionEvidence(
            frameworks_detected={},
            dependency_files_identified=dep_refs,
            frameworks_count=0,
            status="parse_error",
        )


def _build_evidence_bundle(request: SkillProfileInput) -> str:
    """Build the evidence bundle for the LLM.

    Pipeline order — explicit, enforced:
      Step 1 — framework detection results (stored in capsule.framework_detection)
               are ALWAYS present and unconditionally included. This service
               runs Step 1 itself (GitHub REST API, no clone/subprocess).

      Step 2 — static_analysis / graph_relations from the capsule follow;
               included ONLY for repos where the analysis service (called
               by NestJS before this request arrived) succeeded. This service
               never runs Step 2 itself — it consumes whatever NestJS provided.

    Enforcement:
      - Step 1's result is captured and stored on the capsule before Step 2
        is even checked. They are sequential, not parallel.
      - Step 2's success or failure NEVER affects whether Step 1's result
        is included in the final evidence sent to the LLM.
    """
    bundle: dict[str, Any] = {
        "role": request.role,
        "repositories": [],
    }

    for capsule in request.selected_repositories:
        repo_evidence: dict[str, Any] = {
            "evidence_id": capsule.evidence_id,
            "full_name": capsule.full_name,
            "html_url": capsule.html_url,
            "primary_language": capsule.primary_language,
            "description": capsule.description,
            "topics": capsule.topics,
            "technologies": capsule.technologies,
        }

        # Step 1 — always present, unconditionally included
        if capsule.framework_detection is not None:
            repo_evidence["framework_detection"] = capsule.framework_detection.model_dump(
                mode="json"
            )
        else:
            repo_evidence["framework_detection"] = {
                "frameworks_detected": {},
                "dependency_files_identified": [],
                "frameworks_count": 0,
                "status": "not_provided",
            }

        # Step 2 — may be absent or partial; never affects Step 1's inclusion.
        # Consumed for whichever repos the analysis service succeeded on.
        if capsule.static_analysis is not None:
            repo_evidence["static_analysis"] = capsule.static_analysis.model_dump(
                mode="json", exclude_none=True
            )
        else:
            repo_evidence["static_analysis"] = {"status": "not_provided"}

        if capsule.graph_relations is not None:
            repo_evidence["graph_relations"] = capsule.graph_relations.model_dump(
                mode="json", exclude_none=True
            )
        else:
            repo_evidence["graph_relations"] = {"status": "not_provided"}

        bundle["repositories"].append(repo_evidence)

    return json.dumps(bundle, indent=2)


def _build_system_prompt(role: str) -> str:
    role_instruction = (
        "The profile is being generated by a repository owner reviewing a "
        "contributor. Be conservative: flag unsupported claims, demand "
        "stronger evidence before assigning intermediate or advanced "
        "proficiency, and scrutinize ownership patterns.\n\n"
        "All repositories in this request belong to ONE project. Merge "
        "evidence from ALL repositories into a single unified assessment. "
        "Produce EXACTLY ONE skills list for the whole project — never "
        "one per repository, never fragmented per-repo output."
        if role == "owner"
        else (
            "The profile is a self-assessment by the contributor. Evaluate "
            "their actual authored changes, not repository-level activity. "
            "Be objective: do not inflate proficiency or confidence based on "
            "team or organizational activity."
        )
    )
    return f"""You analyze only the repository evidence capsules supplied by Share-k.
Return structured skill candidates and fraud signals. Never crawl repositories,
invent evidence, or cite an evidenceId absent from the input.

{role_instruction}

Evidence you receive per repository:

1.  **Framework/Library/ORM detection (Step 1)** — These results are
    ALWAYS present and reliable. They are produced by a deterministic
    dependency-file scanner that reads requirements.txt, pyproject.toml,
    package.json, pom.xml, *.csproj, and similar files via the GitHub
    REST API. Treat them as first-class, trustworthy evidence. Never
    second-guess or discard them.

2.  **Static analysis (Step 2)** — When `status` is "success", the
    numeric metrics are real and may be used. When `status` is anything
    other than "success" (e.g. "language_not_supported",
    "no_analyzable_content", "tool_unavailable", "not_provided"), do
    NOT estimate or guess any metric value. Instead, state plainly in
    the relevant skill's evidence text that static analysis was
    unavailable for that repo, including the specific status reason.
    Base judgment on whatever evidence remains, including Step 1's
    detection results which are unaffected by Step 2's outcome.

3.  **Graph relations (Step 2)** — Same rule as static analysis: when
    `status` is not "success", state it plainly rather than fabricating
    relationship counts. Step 1 evidence remains fully usable.

For each skill:
- Use only languages, technologies, framework detection, static
  analysis (when available), graph relations (when available),
  statistics, README excerpts, contribution activity, commit signals,
  and contributor authorship in the capsules.
- Cite at least one exact evidenceId from a capsule.
- Use proficiency beginner, intermediate, or advanced.
- Explain the concrete evidence and list material limitations.
- Do not treat repository-wide activity as contributor authorship.

Flag contradictions or suspicious authorship as fraud signals. Do not
include provider/model/version metadata; the service adds trusted audit
metadata.
"""


class SkillProfileProviderError(Exception):
    pass


class SkillProfileProviderTimeout(SkillProfileProviderError):
    pass


def assess_evidence_quality(
    repositories: list[RepositoryEvidenceCapsule],
    role: str = "contributor",
) -> str:
    authored = [
        repository
        for repository in repositories
        if repository.authorship.contribution_detected
        and (
            repository.authorship.total_commits > 0
            or repository.authorship.recent_commit_count > 0
            or bool(repository.authorship.matched_recent_commit_shas)
            or (
                repository.authorship.additions
                + repository.authorship.deletions
                > 0
            )
        )
    ]
    if not authored:
        return "weak"

    total_commits = sum(
        repository.authorship.total_commits for repository in authored
    )
    total_changes = sum(
        repository.authorship.additions + repository.authorship.deletions
        for repository in authored
    )
    evidenced_repositories = sum(
        bool(
            repository.languages
            or repository.technologies
            or repository.commit_signals
        )
        for repository in authored
    )

    if role == "owner":
        threshold_commits = 20
        threshold_changes = 1000
        threshold_repos = 3
    else:
        threshold_commits = 10
        threshold_changes = 500
        threshold_repos = 2

    if (
        len(authored) >= threshold_repos
        and total_commits >= threshold_commits
        and total_changes >= threshold_changes
        and evidenced_repositories >= threshold_repos
    ):
        return "strong"
    return "medium"


def _deterministic_fraud_signals(request: SkillProfileInput) -> list[FraudSignal]:
    signals: list[FraudSignal] = []
    for repository in request.selected_repositories:
        authorship = repository.authorship
        if authorship.github_login.casefold() != request.github_login.casefold():
            signals.append(
                FraudSignal(
                    code="github_login_mismatch",
                    severity="high",
                    message="Capsule authorship login does not match the requested contributor.",
                    repository_full_name=repository.full_name,
                )
            )
        if (
            authorship.contribution_detected
            and authorship.total_commits == 0
            and authorship.recent_commit_count == 0
            and not authorship.matched_recent_commit_shas
            and authorship.additions + authorship.deletions == 0
        ):
            signals.append(
                FraudSignal(
                    code="unsupported_contribution_claim",
                    severity="medium",
                    message=(
                        "Contribution was marked detected without authored commit "
                        "or change evidence."
                    ),
                    repository_full_name=repository.full_name,
                )
            )
    return signals


async def _invoke_model(
    request: SkillProfileInput,
    system_prompt: str,
    evidence: str,
) -> ModelSkillProfileAnalysis:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Analyze these evidence capsules:\n\n{evidence}"),
        ]
    )
    structured = get_llm().with_structured_output(ModelSkillProfileAnalysis)
    result = await (prompt | structured).ainvoke({"evidence": evidence})
    return ModelSkillProfileAnalysis.model_validate(result)


def _validate_model_citations(
    analysis: ModelSkillProfileAnalysis, request: SkillProfileInput
) -> None:
    # Handle case where model returns invalid type (e.g., dict instead of model)
    if not isinstance(analysis, ModelSkillProfileAnalysis):
        raise SkillProfileProviderError("model returned invalid output type")
    allowed_evidence_ids = {
        repository.evidence_id for repository in request.selected_repositories
    }
    allowed_repository_names = {
        repository.full_name for repository in request.selected_repositories
    }
    for skill in analysis.skills:
        if any(
            evidence_id not in allowed_evidence_ids
            for evidence_id in skill.evidence_ids
        ):
            raise SkillProfileProviderError("model cited an unknown evidence ID")
    for signal in analysis.fraud_signals:
        if (
            signal.repository_full_name is not None
            and signal.repository_full_name not in allowed_repository_names
        ):
            raise SkillProfileProviderError("model cited an unknown repository")


def _validate_skill_count(analysis: ModelSkillProfileAnalysis) -> None:
    if len(analysis.skills) > MAX_SKILLS:
        raise SkillProfileProviderError(
            f"model returned {len(analysis.skills)} skills, "
            f"exceeding maximum of {MAX_SKILLS}"
        )


def _merge_fraud_signals(
    deterministic: list[FraudSignal], model_signals: list[FraudSignal]
) -> list[FraudSignal]:
    merged: list[FraudSignal] = []
    seen: set[tuple[str, str | None]] = set()
    for signal in [*deterministic, *model_signals]:
        key = (signal.code, signal.repository_full_name)
        if key not in seen:
            seen.add(key)
            merged.append(signal)
    return merged


def _check_no_analyzable_evidence(request: SkillProfileInput) -> bool:
    """Guardrail: if Step 1 AND Step 2 produced entirely empty evidence
    for EVERY repository, return True to fail-fast with a clear error.

    Runs AFTER both Step 1 (unconditionally) and Step 2 (conditionally,
    per-repo) evidence has been captured on the capsule. Reads from the
    capsule directly — no separate parameter needed.

    This guardrail checks both evidence sources independently: Step 1's
    presence alone is sufficient to pass. Step 2's absence never causes
    a false positive."""
    for capsule in request.selected_repositories:
        has_frameworks = (
            capsule.framework_detection is not None
            and bool(capsule.framework_detection.frameworks_detected)
        )
        has_static = (
            capsule.static_analysis is not None
            and capsule.static_analysis.status == "success"
        )
        has_graph = (
            capsule.graph_relations is not None
            and capsule.graph_relations.status == "success"
        )
        if has_frameworks or has_static or has_graph:
            return False
    return True


async def generate_skill_profile(request: SkillProfileInput) -> SkillProfileResult:
    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE ORDER — EXPLICIT AND ENFORCED
    # ══════════════════════════════════════════════════════════════════════
    #
    # For every repository in this request, in this exact order:
    #
    #   Step 1 (always, first, unconditional):
    #     Framework/library/ORM detection via the existing detector
    #     — GitHub-API-sourced dependency data only. No clone, no
    #     subprocess, no analysis tooling. This service runs Step 1 itself.
    #
    #   Step 2 (immediately after Step 1, per-repo):
    #     static_analysis / graph_relations evidence, sourced via the
    #     analysis service's REST API. FastAPI calls
    #     ``POST /analyze/repo`` per-repo, and populates the capsule
    #     with the mapped results.
    #
    # ARCHITECTURE DECISION — FASTAPI CALLS THE ANALYSIS SERVICE OVER HTTP
    #
    #   FastAPI calls the code-analysis service via plain HTTP (httpx).
    #   It sends a POST to ``{ANALYSIS_SERVICE_URL}/analyze/repo`` with
    #   ``{repo_url, language, requested_tools, pat}`` in the JSON body
    #   for each repository independently. The ``github_pat`` is sent
    #   in the HTTP body for each per-repo call — never stored in a
    #   request-scoped variable that persists beyond the request, never
    #   logged, never cached across repos or across requests. The
    #   analysis service uses the PAT only for the single clone
    #   operation and discards it.
    #
    #   Rationale: HTTP transport removes the subprocess/stdio coupling
    #   and allows the analysis service to run independently. The
    #   evidence path becomes: FastAPI POSTs → analysis service clones
    #   + analyzes → returns JSON → FastAPI maps into capsule models →
    #   LLM assessment. No intermediate service, no pre-populated
    #   capsule handoff, no dual PAT paths.
    #
    # ENFORCEMENT
    #
    #   - Step 1's result is captured and stored on the capsule BEFORE
    #     Step 2 is even started. They are sequential, not parallel.
    #   - Step 2's success or failure NEVER affects whether Step 1's
    #     result is included in the final evidence sent to the LLM.
    #   - The guardrail (_check_no_analyzable_evidence) checks both
    #     sources independently: Step 1's presence alone is sufficient.
    #   - github_pat is scoped to a single per-repo HTTP call; the
    #     Python variable `request.github_pat` is used only to pass it
    #     in the JSON body and is never stored elsewhere.
    # ══════════════════════════════════════════════════════════════════════

    # ── STEP 1: Framework/Library/ORM Detection ──────────────────────────
    # Always runs first for every repository, unconditionally and
    # independently of Step 2 or evidence quality. Uses GitHub REST API
    #      only — no clone, no subprocess, no analysis-service tooling.
    # ──────────────────────────────────────────────────────────────────────
    framework_results = await asyncio.gather(
        *(
            _run_step1_framework_detection(capsule, request.github_pat)
            for capsule in request.selected_repositories
        ),
        return_exceptions=True,
    )
    resolved_frameworks: list[FrameworkDetectionEvidence] = []
    for result in framework_results:
        if isinstance(result, Exception):
            resolved_frameworks.append(
                FrameworkDetectionEvidence(
                    frameworks_detected={},
                    dependency_files_identified=[],
                    frameworks_count=0,
                    status="parse_error",
                )
            )
        else:
            resolved_frameworks.append(result)

    # Store detection results into each capsule before any Step 2 logic.
    # Downstream consumers (evidence bundle builder, guardrail) read from
    # capsule.framework_detection as the single source of truth.
    for capsule, fw_evidence in zip(
        request.selected_repositories, resolved_frameworks
    ):
        capsule.framework_detection = fw_evidence

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1 COMPLETE — results now stored on every capsule.
    #
    # Step 2 (analysis service) runs NOW, per-repo, independently.
    # The github_pat is sent in the HTTP body for each
    # per-repo call — never stored, never logged, never cached.
    #
    # If Step 2 fails for any repo (timeout, HTTP error, connection
    # refused), the capsule's static_analysis and graph_relations are
    # set to failure status values by the HTTP client. If the entire
    # analysis service is unreachable, we catch the exception and mark
    # ALL remaining repos as unavailable so the pipeline continues with
    # Step 1 evidence. Step 2's absence per repo NEVER affects Step 1's
    # inclusion.
    # ══════════════════════════════════════════════════════════════════════
    try:
        await _run_step2_analysis(
            repos=request.selected_repositories,
            github_pat=request.github_pat,
        )
    except Exception:
        logger.exception("Step 2 (analysis) crashed — marking all repos unavailable")
        for capsule in request.selected_repositories:
            if capsule.static_analysis is None:
                capsule.static_analysis = StaticAnalysisEvidence(status="tool_unavailable")
            if capsule.graph_relations is None:
                capsule.graph_relations = GraphRelationsEvidence(status="tool_unavailable")

    # ── ROLE-BASED BRANCH —────────────────────────────────────────────────
    # Two callers share this workflow — skill_profiling (contributor) and
    # project_creation (owner). Both consume the same Step 1 + Step 2
    # evidence sources, but differ in scope and aggregation semantics.
    #
    #   "contributor":
    #     Per-contributor assessment. Step 1 framework detection is always
    #     present. Step 2 (static_analysis/graph_relations) may be absent
    #     or partial per repo — the bundle marks missing fields as
    #     "not_provided". The LLM evaluates the contributor's actual
    #     authored changes, not repository-level activity.
    #
    #   "owner":
    #     Project-level assessment. Every repo in selected_repositories is
    #     treated as belonging to ONE project. Step 1 AND Step 2 evidence
    #     from ALL repos is merged into a SINGLE evidence bundle. The LLM
    #     is explicitly instructed to produce EXACTLY ONE unified skills
    #     list for the whole project — never one per repo, never fragmented
    #     per-repo output. Cross-repo deduplication (strongest evidence
    #     wins) is enforced via the system prompt.
    #
    # Both roles share the same evidence bundle structure, guardrail, and
    # single-LLM-call architecture. The only differences are:
    #   (a) the system prompt's role-specific instructions,
    #   (b) evidence quality thresholds (owner is stricter).
    # ──────────────────────────────────────────────────────────────────────

    evidence_quality = assess_evidence_quality(
        request.selected_repositories, request.role
    )
    deterministic_signals = _deterministic_fraud_signals(request)

    if evidence_quality == "weak":
        return SkillProfileResult(
            skills=[],
            fraud_signals=deterministic_signals,
            evidence_quality="weak",
            recommendation="needs_more_evidence",
            provider=settings.ai_provider,
            model=settings.default_model,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            service_version=settings.service_version,
        )

    # ── Merge Step 1 + Step 2 into evidence bundle ──────────────────────
    #   Step 1 (framework_detection) — unconditionally included for every repo.
    #   Step 2 (static_analysis, graph_relations) — included per repo where
    #             the analysis service succeeded; absent repos get "not_provided" sentinel.
    #   Step 2's availability per repo never affects Step 1's inclusion.
    #
    #   For "contributor": the bundle contains 1+ repos with per-contributor
    #     evidence; the LLM evaluates the contributor's authored changes.
    #   For "owner": ALL repos belong to ONE project; the bundle contains
    #     ALL repos' evidence merged; the LLM produces ONE unified skills list.
    # ──────────────────────────────────────────────────────────────────────
    evidence = _build_evidence_bundle(request)

    # ── Guardrail ────────────────────────────────────────────────────────
    # Fails fast only when BOTH Step 1 AND Step 2 produced zero evidence
    # for EVERY repository. Step 1's presence alone is sufficient to pass.
    if _check_no_analyzable_evidence(request):
        raise SkillProfileProviderError("no_analyzable_evidence")

    # ── LLM call ────────────────────────────────────────────────────────
    # Single call for ALL repos, regardless of role. The system prompt
    # (built from request.role) tells the LLM how to aggregate:
    #   - "contributor": per-contributor, authored-changes-scoped evaluation.
    #   - "owner": exactly one unified skills list, project-wide, strongest-
    #     evidence-wins cross-repo deduplication.
    system_prompt = _build_system_prompt(request.role)

    # Retry loop: LLM parsing errors AND structured-output validation
    # failures (citation validation, skill count) are retried once.
    last_error: Exception | None = None
    for attempt in range(1 + MAX_EVIDENCE_ID_LOOKUP_RETRIES):
        try:
            analysis = await asyncio.wait_for(
                _invoke_model(request, system_prompt, evidence),
                timeout=settings.ai_skill_profile_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise SkillProfileProviderTimeout(
                "skill-profile provider timed out"
            ) from exc
        except (ValidationError, ValueError, TypeError) as exc:
            last_error = exc
            if attempt >= MAX_EVIDENCE_ID_LOOKUP_RETRIES:
                raise SkillProfileProviderError(
                    f"skill-profile provider returned invalid output after retry: {exc}"
                ) from exc
            continue
        except Exception as exc:
            raise SkillProfileProviderError(
                "skill-profile provider failed"
            ) from exc

        # Validate model output — retry-able structured-output validation
        try:
            _validate_model_citations(analysis, request)
            _validate_skill_count(analysis)
        except SkillProfileProviderError as exc:
            last_error = exc
            if attempt >= MAX_EVIDENCE_ID_LOOKUP_RETRIES:
                raise SkillProfileProviderError(
                    f"skill-profile provider returned invalid output after retry: {exc}"
                ) from exc
            continue
        except (ValidationError, ValueError, TypeError) as exc:
            last_error = exc
            if attempt >= MAX_EVIDENCE_ID_LOOKUP_RETRIES:
                raise SkillProfileProviderError(
                    f"skill-profile provider returned invalid output after retry: {exc}"
                ) from exc
            continue

        return SkillProfileResult(
            skills=analysis.skills,
            fraud_signals=_merge_fraud_signals(
                deterministic_signals, analysis.fraud_signals
            ),
            evidence_quality=evidence_quality,
            recommendation="pending_review",
            provider=settings.ai_provider,
            model=settings.default_model,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            service_version=settings.service_version,
        )

    # Should not be reached — retries should have raised
    raise SkillProfileProviderError(
        f"skill-profile provider failed after {1 + MAX_EVIDENCE_ID_LOOKUP_RETRIES} attempts: {last_error}"
    )