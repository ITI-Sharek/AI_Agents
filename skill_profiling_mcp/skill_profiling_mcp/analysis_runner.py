"""In-container analysis runner for the Skill Profiling MCP sandbox.

Executed INSIDE the ``skill-profiling-mcp-analysis`` container image
(never on the host):

    python -m skill_profiling_mcp.analysis_runner static <scope_path> --language <language> --timeout N
    python -m skill_profiling_mcp.analysis_runner graph <repo_path> --timeout N
    python -m skill_profiling_mcp.analysis_runner graph_extract <repo_path> --timeout N
    python -m skill_profiling_mcp.analysis_runner graph_select <repo_path> --timeout N

The host only ever runs the ``docker`` CLI. This module runs the
project's existing static-analysis tools (radon, pylint, eslint,
checkstyle, gocyclo, golangci-lint, clippy-driver, rubocop, cppcheck,
clang-tidy, phpcs, detekt, swiftlint, dart, sqlfluff, dotnet)
plus the LOCAL offline Semgrep analyzer (python/javascript/typescript
only) and the external ``graphify`` binary, then emits ONE
deterministic JSON evidence object on stdout, preceded by the
``EVIDENCE_OK`` marker. The Graphify evidence carries the full graph
payload (``nodes``, ``edges``, and any other fields Graphify produced)
— it is never reduced to aggregate metrics. A tool missing from the
image is reported as deterministic ``tool_unavailable`` — never a host
fallback.

Semgrep runs exclusively inside the image against files already present
in the workspace, with a ruleset bundled into the image and the
``--disable-version-check --metrics=off`` flags: no registry, no
cloud, no API key, no network use, and ZERO git commands. It is an
ADDITIONAL analyzer merged into the same bounded findings list for
python/javascript/typescript — it never replaces radon, pylint, or
eslint. If the semgrep binary or its bundled rules are missing from the
image, the semgrep contribution is skipped (best effort) while the
primary analyzers keep working.

TypeScript is analyzed by the same ESLint adapter used for JavaScript,
but with the ``@typescript-eslint/parser`` and
``@typescript-eslint/eslint-plugin`` packages and a dedicated in-image
ESLint config, so ``.ts``/``.tsx`` files are genuinely parsed as
TypeScript instead of producing espree parse errors.

Vue single-file components are analyzed with the same ESLint
installation plus ``eslint-plugin-vue`` (and ``vue-eslint-parser``) and
a dedicated in-image ESLint config. The parser mapping routes
``<script lang="ts">`` blocks to ``@typescript-eslint/parser`` and
JavaScript blocks to ``espree``, so both JavaScript and TypeScript Vue
components are analyzed with the existing TypeScript ESLint setup.

Java is analyzed with the image-bundled Checkstyle 10.21.2 JAR
(``java -jar`` on the OpenJDK 17 JRE) using a deterministic in-image
``checkstyle.xml`` configuration — no Maven/Gradle and no runtime config
download. Checkstyle's XML report is normalized into the same finding
structure used by the other adapters.

Go is analyzed with gocyclo (average cyclomatic complexity metric) and
the pinned golangci-lint 2.1.6 binary (``--no-config`` standard linter
set), both running locally inside the image against the supplied
``.go`` files. golangci-lint's JSON report and the gocyclo average are
merged into the same bounded evidence structure used by the other
adapters.

Rust is analyzed with the ``clippy-driver`` of the image-bundled Rust
1.86.0 toolchain. Each ``.rs`` file is compiled individually as a ``lib``
crate and clippy's JSON diagnostics (stderr, ``--error-format=json``)
are normalized into the shared finding structure.

Ruby is analyzed with the image-bundled RuboCop 1.75.8 using a
deterministic in-image ``rubocop.yml`` (``--config``) so analysis never
depends on a repository's own RuboCop configuration; the JSON offense
report is normalized into the shared finding structure.

PHP is analyzed with the image-bundled PHP_CodeSniffer 3.11.3 phar run
through the in-image PHP 8.4 CLI and a deterministic in-image
``phpcs-standard.xml`` (``--standard=``) so analysis never depends on a
repository's own PHPCS configuration; the JSON report is normalized
into the shared finding structure. A non-zero phpcs exit code reports
violations, not failure.

 C is analyzed with the image-bundled Cppcheck 2.17.1 (built from the
 pinned source tarball). Cppcheck runs with ``--language=c`` (C mode for
 ``.c``/``.h`` files), the deterministic ``--enable=warning``
 check set, bounded configuration expansion (``--max-configs=1``), and
``--xml`` machine-readable output; its XML report is normalized into
the shared finding structure. Cppcheck's non-zero exit code reports
violations, not failure.

C++ is analyzed with the image-bundled clang-tidy 19 (Debian
``clang-tidy-19``) plus the same Cppcheck 2.17.1 as an additional
merged analyzer. clang-tidy runs per file with a deterministic
in-image config (``--config-file``) so a repository's own ``.clang-tidy``
 is never read, and its textual diagnostics (stderr) are normalized into
 the shared finding structure. Cppcheck runs with ``--language=c++``,
 ``--enable=warning``, ``--max-configs=1`` and ``--xml`` output
 over the same files. A non-zero exit code reports violations, not
 failure.

C# is analyzed with the .NET SDK 8.x ``dotnet build`` of the
repository's own project/solution (implicit restore for fresh clones,
``-warnaserror:false``)
with the Roslyn compiler and the built-in .NET analyzers enabled. A
machine-readable SARIF error log of ALL compiler (CS*) and analyzer
(CA*) diagnostics is produced by the Roslyn csc task
(``/p:ErrorLog=...``) and normalized into the shared finding structure;
the human-readable console output is not parsed. A missing .NET SDK,
an unparseable error log with a non-zero build exit, or a project that
cannot be built without restore are reported deterministically.

Kotlin is analyzed with the image-bundled Detekt 1.23.8 CLI
(``java -jar`` on the OpenJDK 21 JRE; the image also bundles the
Kotlin 2.1.21 compiler). Detekt runs syntactically (no type resolution,
no classpath) against the FULL workspace with its deterministic
built-in default configuration — the CLI never reads a repository's own
``detekt.yml`` — and writes a machine-readable JSON report
(``--report "json:..."``), which is normalized into the shared finding
structure. Detekt rules carry no per-finding severity, so findings map
to ``warning``. A non-zero exit code reports violations, not failure.
A missing JVM or Detekt JAR is reported as deterministic
``tool_unavailable`` — never a host fallback.

Swift is analyzed with the image-bundled SwiftLint 0.58.2 Linux binary
(the image also bundles the Swift 6.0.3 toolchain). SwiftLint runs
locally against the workspace ``.swift`` files with a deterministic
in-image config (``--config``) so a repository's own ``.swiftlint.yml``
is never read, and emits a machine-readable JSON report (``--reporter
json``) which is normalized into the shared finding structure. A
non-zero exit code reports violations, not failure. A missing
swiftlint binary or its bundled config is reported as deterministic
``tool_unavailable`` — never a host fallback.

Dart is analyzed with the official Dart Analyzer bundled in the Dart
SDK 3.7.3 (``dart analyze``; the same analyzer is used for Flutter
projects — there is no separate Flutter analyzer). ``dart analyze``
takes a single directory, not a file list, so the WHOLE scope is
analyzed with machine-readable JSON output (``--format=json``) and
diagnostics are mapped into the shared finding structure, filtering to
the discovered ``.dart`` files. Like the C# stack, the analyzer is
authoritative and deterministic per repository — the repository's own
project configuration (``pubspec.yaml`` / ``analysis_options.yaml``)
is respected with standard ``dart analyze`` semantics. A non-zero exit
code reports violations, not failure. A missing ``dart`` binary is
reported as deterministic ``tool_unavailable`` — never a host
fallback.

SQL is analyzed with the image-bundled SQLFluff 3.4.2 (a Python
linter installed via pip; parsing and linting are fully offline and
local — no database connection is ever required). SQLFluff runs
``lint`` against the supplied ``.sql`` files with the deterministic
in-image config (``--config``) so a repository's own ``.sqlfluff`` is
never read (dialect pinned to ANSI, raw templater — no template
rendering), and emits a machine-readable JSON report (``--format
json``) which is normalized into the shared finding structure.
SQLFluff violations carry no per-finding severity, so findings map to
``warning``. A non-zero exit code reports violations, not failure.
A missing SQLFluff binary or its bundled config is reported as
deterministic ``tool_unavailable`` — never a host fallback.

``static`` analyzes the FULL repository workspace (``/workspace/repo``)
— there is no contributor file filtering before static analysis.
``graph`` analyzes the FULL repository (``/workspace/repo``) with
Graphify and selects the contributor-related nodes/relations for
evidence from the scope manifest written by ``filter_contributor_code``
(``/workspace/scope.manifest``). The orchestrated flow splits Graphify
into ``graph_extract`` (full-repository extraction only, no scope
consulted) followed by ``graph_select`` (contributor graph filtering of
the extracted artifact), so the MCP orchestrator can run Graphify
concurrently with the contributor filter while guaranteeing that graph
filtering always happens AFTER Graphify and after the scope manifest
exists.

This module is stdlib-only so it runs in the image without installing the
MCP package dependencies. It shares the language/analyzer registry with
``skill_profiling_mcp.analysis`` (whose import chain is stdlib-only).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from skill_profiling_mcp.analysis import ANALYZER_NAMES, LANGUAGE_ALIASES

DEFAULT_TIMEOUT = 60
MAX_FINDINGS = 200
MAX_RELATIONS = 50
MAX_MESSAGE_CHARS = 200
MAX_ERROR_CHARS = 500
MAX_RULE_ID_CHARS = 120

# Bundled in the analysis image (see ``skill_profiling_mcp/docker/``).
SEMGREP_CONFIG = "/opt/semgrep-rules/rules.yml"
TS_ESLINT_CONFIG = "/opt/ts-eslint/eslintrc.json"
VUE_ESLINT_CONFIG = "/opt/vue-eslint/vue-eslintrc.json"
CHECKSTYLE_JAR = "/opt/checkstyle/checkstyle.jar"
CHECKSTYLE_CONFIG = "/opt/checkstyle/checkstyle.xml"
RUBOCOP_CONFIG = "/opt/rubocop/rubocop.yml"
PHPCS_PHAR = "/opt/phpcs/phpcs.phar"
PHPCS_CONFIG = "/opt/phpcs/phpcs-standard.xml"
CLANG_TIDY_CONFIG = "/opt/clang-tidy/clang-tidy.yml"
DETEKT_JAR = "/opt/detekt/detekt-cli-all.jar"
SWIFTLINT_CONFIG = "/opt/swiftlint/swiftlint.yml"
SQLFLUFF_CONFIG = "/opt/sqlfluff/.sqlfluff"

EXTENSIONS: dict[str, frozenset[str]] = {
    "python": frozenset({".py"}),
    "javascript": frozenset({".js", ".jsx", ".mjs"}),
    "typescript": frozenset({".ts", ".tsx"}),
    "vue": frozenset({".vue"}),
    "java": frozenset({".java"}),
    "go": frozenset({".go"}),
    "rust": frozenset({".rs"}),
    "ruby": frozenset({".rb"}),
    "php": frozenset({".php"}),
    "c": frozenset({".c", ".h"}),
    "cpp": frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}),
    "kotlin": frozenset({".kt", ".kts"}),
    "swift": frozenset({".swift"}),
    "csharp": frozenset({".cs"}),
    "dart": frozenset({".dart"}),
    "sql": frozenset({".sql"}),
}


class ToolUnavailable(RuntimeError):
    """Raised when a required analysis tool is not installed in the image."""


class ToolTimeout(RuntimeError):
    """Raised when an analysis tool exceeds its time limit."""


def _invoke(cmd: list[str], timeout: int) -> str:
    """Run an analysis tool inside the container; return stdout."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ToolUnavailable from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeout from exc
    return proc.stdout


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_files(scope_path: str, extensions: frozenset[str]) -> list[str]:
    root = Path(scope_path)
    if not root.is_dir():
        return []
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.suffix in extensions
    )


def _abs(scope_path: str, rel: str) -> str:
    return str(Path(scope_path) / rel)


def _relative_path(raw: str, scope_path: str) -> str:
    """Map a tool-reported path to a scope-relative path.

    Falls back to the bare file name — never leaks a host path.
    """
    try:
        root = Path(scope_path).resolve()
        rel = Path(raw).resolve().relative_to(root)
        return str(rel)
    except (ValueError, OSError):
        return Path(raw).name


# ---------------------------------------------------------------------------
# Findings normalization (deterministic + bounded)
# ---------------------------------------------------------------------------


def _finding(
    *,
    file_path: str,
    line: int,
    column: int,
    severity: str,
    category: str,
    rule_id: str,
    message: str,
) -> dict[str, object]:
    return {
        "file_path": file_path,
        "line": int(line),
        "column": int(column),
        "severity": severity,
        "category": category,
        "rule_id": rule_id,
        "message": message[:MAX_MESSAGE_CHARS],
    }


def _normalize_severity(raw: str) -> str:
    value = str(raw).lower()
    if value in ("error", "fatal", "critical", "major", "high"):
        return "error"
    if value in ("warning", "warn", "minor", "medium", "convention", "refactor"):
        return "warning"
    return "info"


def _bound_findings(findings: list[dict[str, object]]) -> tuple[list[dict[str, object]], bool]:
    findings.sort(
        key=lambda f: (f["file_path"], f["line"], f["column"], f["rule_id"], f["message"])
    )
    truncated = len(findings) > MAX_FINDINGS
    return findings[:MAX_FINDINGS], truncated


# ---------------------------------------------------------------------------
# Semgrep (additional LOCAL offline analyzer — python/javascript/typescript)
# ---------------------------------------------------------------------------


def _load_semgrep_json(output: str) -> dict[str, object] | None:
    """Parse the ``--json`` output of ``semgrep scan`` (defensively).

    ``--json`` writes the results object to stdout; a leading-log fallback
    extracts the first balanced JSON object when the version prints extra
    lines. Returns None when no JSON could be parsed.
    """
    try:
        data = json.loads(output)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _semgrep_rule_id(check_id: str) -> str:
    """Namespaced, bounded, deterministic rule id for a semgrep finding."""
    rule_id = f"semgrep.{check_id}" if check_id else "semgrep"
    if len(rule_id) > MAX_RULE_ID_CHARS:
        return rule_id[:MAX_RULE_ID_CHARS]
    return rule_id


def _semgrep_findings(
    scope_path: str, files: list[str], timeout: int
) -> list[dict[str, object]]:
    """Run the LOCAL offline semgrep analyzer; return normalized findings.

    Best effort by design: semgrep is an ADDITIONAL analyzer. When the
    binary or the bundled ruleset is missing from the image, or the scan
    fails/times out, ``[]`` is returned and the primary analyzers (radon,
    pylint, eslint) remain authoritative — analysis never fails because
    of semgrep.

    Semgrep executes ONLY against the files already present in the
    workspace, with the image-bundled ruleset and ``--disable-version-check``
    / ``--metrics=off``: no registry, no cloud, no
    API key, no network use, and no git commands.
    """
    if not files or not Path(SEMGREP_CONFIG).is_file():
        return []
    abs_files = [_abs(scope_path, f) for f in files]
    try:
        output = _invoke(
            [
                "semgrep",
                "scan",
                "--disable-version-check",
                "--metrics=off",
                "--config",
                SEMGREP_CONFIG,
                "--json",
                *abs_files,
            ],
            timeout,
        )
    except (ToolUnavailable, ToolTimeout):
        return []

    data = _load_semgrep_json(output)
    if data is None:
        return []
    results = data.get("results")
    if not isinstance(results, list):
        return []

    findings: list[dict[str, object]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        extra = result.get("extra")
        extra = extra if isinstance(extra, dict) else {}
        start = result.get("start")
        start = start if isinstance(start, dict) else {}
        severity = extra.get("severity") or result.get("severity") or "warning"
        findings.append(
            _finding(
                file_path=_relative_path(str(result.get("path", "")), scope_path),
                line=int(start.get("line", 0) or 0),
                column=int(start.get("col", 0) or 0),
                severity=_normalize_severity(str(severity)),
                category="semgrep",
                rule_id=_semgrep_rule_id(str(result.get("check_id", ""))),
                message=str(extra.get("message", "")),
            )
        )
    return findings


def _static_evidence(
    canonical: str,
    analyzer: str,
    files: list[str],
    complexity: float | None,
    maintainability_index: float | None,
    findings: list[dict[str, object]],
    findings_truncated: bool,
) -> dict[str, object]:
    severities = dict(
        sorted(Counter(f["severity"] for f in findings).items())
    )
    return {
        "status": "success",
        "language": canonical,
        "analyzer": analyzer,
        "files_analyzed": len(files),
        "complexity": complexity,
        "maintainability_index": maintainability_index,
        "findings": findings,
        "finding_count": len(findings),
        "finding_truncated": findings_truncated,
        "severity_counts": severities,
        "error_message": None,
    }


def _failure_evidence(
    canonical: str, status: str, analyzer: str, message: str | None = None
) -> dict[str, object]:
    return {
        "status": status,
        "language": canonical,
        "analyzer": analyzer,
        "files_analyzed": 0,
        "complexity": None,
        "maintainability_index": None,
        "findings": [],
        "finding_count": 0,
        "finding_truncated": False,
        "severity_counts": {},
        "error_message": message[:MAX_ERROR_CHARS] if message else None,
    }


def _no_analyzable(canonical: str, analyzer: str) -> dict[str, object]:
    return _failure_evidence(canonical, "no_analyzable_content", analyzer)


def _unsupported(language: str) -> dict[str, object]:
    return {
        "status": "unsupported_language",
        "language": language,
        "analyzer": None,
        "files_analyzed": 0,
        "complexity": None,
        "maintainability_index": None,
        "findings": [],
        "finding_count": 0,
        "finding_truncated": False,
        "severity_counts": {},
        "error_message": None,
    }


# ---------------------------------------------------------------------------
# Language adapters (tools identical to the project's analysis capabilities)
# ---------------------------------------------------------------------------
#
# Semgrep is an ADDITIONAL local analyzer merged into the findings of the
# python/javascript/typescript adapters (see ``_semgrep_findings``); it
# never replaces radon, pylint, or eslint.


def _analyze_python(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    analyzer = ANALYZER_NAMES["python"]
    abs_files = [_abs(scope_path, f) for f in files]
    try:
        cc_output = _invoke(["radon", "cc", "-j", *abs_files], timeout)
        mi_output = _invoke(["radon", "mi", "-j", *abs_files], timeout)
        lint_output = _invoke(
            ["pylint", "--output-format=json", "--rcfile=/dev/null", *abs_files],
            timeout,
        )
    except ToolUnavailable:
        return _failure_evidence("python", "tool_unavailable", analyzer)
    except ToolTimeout:
        return _failure_evidence("python", "timeout", analyzer)

    findings: list[dict[str, object]] = []
    try:
        for item in json.loads(lint_output or "[]"):
            findings.append(
                _finding(
                    file_path=_relative_path(item.get("path", ""), scope_path),
                    line=item.get("line", 0) or 0,
                    column=item.get("column", 0) or 0,
                    severity=_normalize_severity(item.get("type", "info")),
                    category=item.get("symbol", "") or "lint",
                    rule_id=item.get("symbol", "") or "lint",
                    message=str(item.get("message", "")),
                )
            )
    except (json.JSONDecodeError, TypeError, AttributeError):
        findings = []
    findings, truncated = _bound_findings(
        findings + _semgrep_findings(scope_path, files, timeout)
    )

    return _static_evidence(
        "python",
        analyzer,
        files,
        _radon_avg_cc(cc_output),
        _radon_avg_mi(mi_output),
        findings,
        truncated,
    )


def _radon_avg_cc(output: str) -> float | None:
    if not output.strip():
        return None
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None
    values = [
        float(func["complexity"])
        for funcs in data.values()
        if isinstance(funcs, list)
        for func in funcs
        if isinstance(func, dict) and "complexity" in func
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _radon_avg_mi(output: str) -> float | None:
    if not output.strip():
        return None
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None
    values = []
    for file_data in data.values():
        if isinstance(file_data, dict) and file_data.get("mi") is not None:
            values.append(float(file_data["mi"]))
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _analyze_js(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    analyzer = ANALYZER_NAMES["javascript"]
    abs_files = [_abs(scope_path, f) for f in files]
    try:
        output = _invoke(["eslint", "--no-eslintrc", "--format", "json", *abs_files], timeout)
    except ToolUnavailable:
        return _failure_evidence("javascript", "tool_unavailable", analyzer)
    except ToolTimeout:
        return _failure_evidence("javascript", "timeout", analyzer)

    findings: list[dict[str, object]] = []
    try:
        for entry in json.loads(output or "[]"):
            fp = entry.get("filePath", "")
            for msg in entry.get("messages", []):
                findings.append(
                    _finding(
                        file_path=_relative_path(fp, scope_path),
                        line=msg.get("line", 0) or 0,
                        column=msg.get("column", 0) or 0,
                        severity={1: "warning", 2: "error"}.get(
                            msg.get("severity", 0), "info"
                        ),
                        category=msg.get("ruleId", "") or "eslint",
                        rule_id=msg.get("ruleId", "") or "eslint",
                        message=str(msg.get("message", "")),
                    )
                )
    except (json.JSONDecodeError, TypeError, AttributeError):
        findings = []
    findings, truncated = _bound_findings(
        findings + _semgrep_findings(scope_path, files, timeout)
    )
    return _static_evidence(
        "javascript", analyzer, files, None, None, findings, truncated
    )


def _analyze_typescript(
    scope_path: str, files: list[str], timeout: int
) -> dict[str, object]:
    """Analyze TypeScript with ESLint using the in-image TS parser stack.

    ESLint is invoked with the dedicated image-bundled config
    (``@typescript-eslint/parser`` + ``@typescript-eslint/eslint-plugin``)
    so ``.ts``/``.tsx`` files are genuinely parsed as TypeScript. The
    config file is part of the analysis image; when it is missing the
    result is deterministic ``tool_unavailable`` — never a host fallback.
    """
    analyzer = ANALYZER_NAMES["typescript"]
    if not Path(TS_ESLINT_CONFIG).is_file():
        return _failure_evidence(
            "typescript",
            "tool_unavailable",
            analyzer,
            "typescript eslint config is not installed in the image",
        )
    abs_files = [_abs(scope_path, f) for f in files]
    try:
        output = _invoke(
            [
                "eslint",
                "--no-eslintrc",
                "--config",
                TS_ESLINT_CONFIG,
                "--format",
                "json",
                *abs_files,
            ],
            timeout,
        )
    except ToolUnavailable:
        return _failure_evidence("typescript", "tool_unavailable", analyzer)
    except ToolTimeout:
        return _failure_evidence("typescript", "timeout", analyzer)

    findings: list[dict[str, object]] = []
    try:
        for entry in json.loads(output or "[]"):
            fp = entry.get("filePath", "")
            for msg in entry.get("messages", []):
                findings.append(
                    _finding(
                        file_path=_relative_path(fp, scope_path),
                        line=msg.get("line", 0) or 0,
                        column=msg.get("column", 0) or 0,
                        severity={1: "warning", 2: "error"}.get(
                            msg.get("severity", 0), "info"
                        ),
                        category=msg.get("ruleId", "") or "eslint",
                        rule_id=msg.get("ruleId", "") or "eslint",
                        message=str(msg.get("message", "")),
                    )
                )
    except (json.JSONDecodeError, TypeError, AttributeError):
        findings = []
    findings, truncated = _bound_findings(
        findings + _semgrep_findings(scope_path, files, timeout)
    )
    return _static_evidence(
        "typescript", analyzer, files, None, None, findings, truncated
    )


def _analyze_vue(
    scope_path: str, files: list[str], timeout: int
) -> dict[str, object]:
    """Analyze Vue SFCs with ESLint + eslint-plugin-vue.

    ESLint is invoked with the dedicated image-bundled config that
    routes ``.vue`` single-file components through ``vue-eslint-parser``
    and reuses the in-image TypeScript stack: the parser mapping routes
    ``<script lang="ts">`` blocks to ``@typescript-eslint/parser`` and
    JavaScript blocks to ``espree``, so both JavaScript and TypeScript
    Vue components are analyzed (``plugin:vue/vue3-recommended`` +
    ``plugin:@typescript-eslint/recommended``). The config file is part
    of the analysis image; when it is missing the result is
    deterministic ``tool_unavailable`` — never a host fallback.
    """
    analyzer = ANALYZER_NAMES["vue"]
    if not Path(VUE_ESLINT_CONFIG).is_file():
        return _failure_evidence(
            "vue",
            "tool_unavailable",
            analyzer,
            "vue eslint config is not installed in the image",
        )
    abs_files = [_abs(scope_path, f) for f in files]
    try:
        output = _invoke(
            [
                "eslint",
                "--no-eslintrc",
                "--config",
                VUE_ESLINT_CONFIG,
                "--format",
                "json",
                *abs_files,
            ],
            timeout,
        )
    except ToolUnavailable:
        return _failure_evidence("vue", "tool_unavailable", analyzer)
    except ToolTimeout:
        return _failure_evidence("vue", "timeout", analyzer)

    findings: list[dict[str, object]] = []
    try:
        for entry in json.loads(output or "[]"):
            fp = entry.get("filePath", "")
            for msg in entry.get("messages", []):
                findings.append(
                    _finding(
                        file_path=_relative_path(fp, scope_path),
                        line=msg.get("line", 0) or 0,
                        column=msg.get("column", 0) or 0,
                        severity={1: "warning", 2: "error"}.get(
                            msg.get("severity", 0), "info"
                        ),
                        category=msg.get("ruleId", "") or "eslint",
                        rule_id=msg.get("ruleId", "") or "eslint",
                        message=str(msg.get("message", "")),
                    )
                )
    except (json.JSONDecodeError, TypeError, AttributeError):
        findings = []
    findings, truncated = _bound_findings(findings)
    return _static_evidence(
        "vue", analyzer, files, None, None, findings, truncated
    )


def _analyze_java(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    """Analyze Java with the image-bundled Checkstyle 10.21.2 JAR.

    Checkstyle runs as ``java -jar`` (OpenJDK 17 JRE) against the
    supplied ``.java`` files with the deterministic in-image
    ``checkstyle.xml`` configuration (no Maven/Gradle, no runtime config
    download). Its XML report is written to stdout and normalized into
    the shared finding structure.

    Checkstyle's CLI exit code equals the number of reported errors, so
    a non-zero code alone does NOT mean failure: analysis succeeded iff
    stdout contains a parseable ``<checkstyle>`` XML document. When no
    XML is produced (invalid configuration, unreadable inputs), the
    result is deterministic ``error`` evidence with the bounded stderr
    detail. A missing JVM or bundled Checkstyle JAR is reported as
    deterministic ``tool_unavailable`` — never a host fallback.
    """
    analyzer = ANALYZER_NAMES["java"]
    if shutil.which("java") is None or not Path(CHECKSTYLE_JAR).is_file():
        return _failure_evidence(
            "java",
            "tool_unavailable",
            analyzer,
            "checkstyle (java -jar) is not installed in the image",
        )
    if not Path(CHECKSTYLE_CONFIG).is_file():
        return _failure_evidence(
            "java",
            "tool_unavailable",
            analyzer,
            "checkstyle configuration is not installed in the image",
        )
    abs_files = [_abs(scope_path, f) for f in files]
    try:
        proc = subprocess.run(
            [
                "java",
                "-jar",
                CHECKSTYLE_JAR,
                "-c",
                CHECKSTYLE_CONFIG,
                "-f",
                "xml",
                *abs_files,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return _failure_evidence("java", "tool_unavailable", analyzer)
    except subprocess.TimeoutExpired:
        return _failure_evidence("java", "timeout", analyzer)

    root = _load_checkstyle_xml(proc.stdout)
    if root is None:
        if proc.returncode == 0:
            return _static_evidence(
                "java", analyzer, files, None, None, [], False
            )
        detail = (proc.stderr or proc.stdout or "").strip()
        return _failure_evidence(
            "java",
            "error",
            analyzer,
            detail or f"checkstyle failed with exit code {proc.returncode}",
        )

    findings, truncated = _bound_findings(_checkstyle_findings(root, scope_path))
    return _static_evidence(
        "java", analyzer, files, None, None, findings, truncated
    )


def _load_checkstyle_xml(output: str) -> ET.Element | None:
    """Parse Checkstyle's XML report (defensively).

    Checkstyle 10.21.2 writes ONLY the ``<checkstyle>`` document to
    stdout; a leading-log fallback extracts the first ``<checkstyle>``
    block when other lines are present. Returns None when no XML could
    be parsed (used to distinguish hard failures from violations).
    """
    try:
        return ET.fromstring(output)
    except ET.ParseError:
        pass
    match = re.search(r"<checkstyle\b.*?</checkstyle>", output, re.DOTALL)
    if match is None:
        return None
    try:
        return ET.fromstring(match.group(0))
    except ET.ParseError:
        return None


def _checkstyle_findings(root: ET.Element, scope_path: str) -> list[dict[str, object]]:
    """Normalize Checkstyle violations into the shared finding structure."""
    findings: list[dict[str, object]] = []
    for file_elem in root.findall("file"):
        fp = file_elem.get("name", "")
        for err in file_elem.findall("error"):
            source = err.get("source", "")
            findings.append(
                _finding(
                    file_path=_relative_path(fp, scope_path),
                    line=int(err.get("line", 0) or 0),
                    column=int(err.get("column", 0) or 0),
                    severity=_normalize_severity(err.get("severity", "error")),
                    category=source.split(".")[-1] if source else "checkstyle",
                    rule_id=source.split(".")[-1] if source else "checkstyle",
                    message=str(err.get("message", "")),
                )
            )
    return findings


def _load_golangci_json(output: str) -> dict[str, object] | None:
    """Parse the ``--output.json.path=stdout`` report of golangci-lint 2.x.

    The JSON report is written as the first line of stdout; golangci-lint
    appends a human summary line ("N issues.") after it. A leading-log
    fallback extracts the first balanced JSON object. Returns None when
    no JSON could be parsed.
    """
    try:
        data = json.loads(output)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _analyze_go(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    """Analyze Go with gocyclo (complexity) + golangci-lint 2.1.6 (lint).

    ``gocyclo -avg`` provides the average cyclomatic complexity metric;
    ``golangci-lint`` (pinned 2.1.6, ``--no-config`` standard linter
    set) provides the lint findings. Both run locally inside the image
    against the supplied ``.go`` files, and their contributions are
    merged into the shared bounded evidence structure — mirroring how
    the python/javascript/typescript adapters merge their extra
    analyzer contribution. A missing Go toolchain, gocyclo, or
    golangci-lint is reported as deterministic ``tool_unavailable`` —
    never a host fallback.
    """
    analyzer = ANALYZER_NAMES["go"]
    if shutil.which("go") is None:
        return _failure_evidence(
            "go",
            "tool_unavailable",
            analyzer,
            "go toolchain is not installed in the image",
        )
    abs_files = [_abs(scope_path, f) for f in files]
    try:
        gocyclo_output = _invoke(["gocyclo", "-avg", *abs_files], timeout)
        lint_output = _invoke(
            [
                "golangci-lint",
                "run",
                "--no-config",
                "--output.json.path=stdout",
                *abs_files,
            ],
            timeout,
        )
    except ToolUnavailable:
        return _failure_evidence("go", "tool_unavailable", analyzer)
    except ToolTimeout:
        return _failure_evidence("go", "timeout", analyzer)

    complexity: float | None = None
    match = re.search(r"(?:Average|Avg):\s*([\d.eE+-]+)", gocyclo_output)
    if match:
        complexity = round(float(match.group(1)), 4)

    findings: list[dict[str, object]] = []
    data = _load_golangci_json(lint_output)
    if data is not None:
        for issue in data.get("Issues", []):
            if not isinstance(issue, dict):
                continue
            pos = issue.get("Pos")
            pos = pos if isinstance(pos, dict) else {}
            findings.append(
                _finding(
                    file_path=_relative_path(str(pos.get("Filename", "")), scope_path),
                    line=int(pos.get("Line", 0) or 0),
                    column=int(pos.get("Column", 0) or 0),
                    severity=_normalize_severity(
                        str(issue.get("Severity") or "warning")
                    ),
                    category=str(issue.get("FromLinter", "") or "golangci-lint"),
                    rule_id=str(issue.get("FromLinter", "") or "golangci-lint"),
                    message=str(issue.get("Text", "")),
                )
            )
    findings, truncated = _bound_findings(findings)
    return _static_evidence(
        "go", analyzer, files, complexity, None, findings, truncated
    )


def _analyze_rust(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    """Analyze Rust with the image-bundled ``clippy-driver``.

    clippy-driver (the clippy component of the Rust 1.86.0 toolchain) is
    a rustc driver and accepts exactly ONE input file per invocation, so
    each ``.rs`` file is analyzed individually as a ``lib`` crate (this
    avoids E0601 "main function not found" noise on library files while
    clippy lints still fire). JSON diagnostics are emitted on stderr
    (``--error-format=json``; clippy-driver does not accept
    ``--message-format``) and normalized into the shared finding
    structure. A missing clippy-driver is reported as deterministic
    ``tool_unavailable``.
    """
    analyzer = ANALYZER_NAMES["rust"]
    abs_files = [_abs(scope_path, f) for f in files]
    findings: list[dict[str, object]] = []
    for abs_file in abs_files:
        try:
            proc = subprocess.run(
                [
                    "clippy-driver",
                    "--edition",
                    "2021",
                    "--crate-type",
                    "lib",
                    "--error-format=json",
                    abs_file,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return _failure_evidence("rust", "tool_unavailable", analyzer)
        except subprocess.TimeoutExpired:
            return _failure_evidence("rust", "timeout", analyzer)
        findings.extend(_clippy_findings(proc.stderr, scope_path))
    findings, truncated = _bound_findings(findings)
    return _static_evidence(
        "rust", analyzer, files, None, None, findings, truncated
    )


def _clippy_findings(output: str, scope_path: str) -> list[dict[str, object]]:
    """Normalize clippy-driver's stderr JSON diagnostics into findings.

    Parses ``{"$message_type":"diagnostic", ...}`` lines. Only
    warning/error diagnostics with a concrete code (clippy lint name or
    rustc error code) and a span become findings — summary lines
    ("N warnings emitted", "aborting due to...") have a null code and
    are skipped.
    """
    findings: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or data.get("$message_type") != "diagnostic":
            continue
        level = str(data.get("level", ""))
        if level not in ("warning", "error"):
            continue
        code = data.get("code")
        if not isinstance(code, dict) or not code.get("code"):
            continue
        spans = data.get("spans")
        if not isinstance(spans, list) or not spans:
            continue
        span = spans[0]
        findings.append(
            _finding(
                file_path=_relative_path(str(span.get("file_name", "")), scope_path),
                line=int(span.get("line_start", 0) or 0),
                column=int(span.get("column_start", 0) or 0),
                severity=_normalize_severity(level),
                category=str(code.get("code")),
                rule_id=str(code.get("code")),
                message=str(data.get("message", "")),
            )
        )
    return findings


def _analyze_ruby(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    """Analyze Ruby with the image-bundled RuboCop 1.75.8.

    RuboCop runs with the deterministic in-image ``rubocop.yml``
    (``--config``) so analysis never depends on a repository's own
    RuboCop configuration, emits JSON on stdout, and its offenses are
    normalized into the shared finding structure. ``--no-server`` keeps
    the run self-contained (no daemon left behind). A missing RuboCop
    or its bundled config is reported as deterministic
    ``tool_unavailable`` — never a host fallback.
    """
    analyzer = ANALYZER_NAMES["ruby"]
    if not Path(RUBOCOP_CONFIG).is_file():
        return _failure_evidence(
            "ruby",
            "tool_unavailable",
            analyzer,
            "rubocop configuration is not installed in the image",
        )
    abs_files = [_abs(scope_path, f) for f in files]
    try:
        output = _invoke(
            [
                "rubocop",
                "--no-server",
                "--config",
                RUBOCOP_CONFIG,
                "--format",
                "json",
                *abs_files,
            ],
            timeout,
        )
    except ToolUnavailable:
        return _failure_evidence("ruby", "tool_unavailable", analyzer)
    except ToolTimeout:
        return _failure_evidence("ruby", "timeout", analyzer)

    findings: list[dict[str, object]] = []
    try:
        for entry in json.loads(output or "{}").get("files", []):
            fp = entry.get("path", "")
            for off in entry.get("offenses", []):
                loc = off.get("location", {})
                findings.append(
                    _finding(
                        file_path=_relative_path(fp, scope_path),
                        line=loc.get("line", 0) or 0,
                        column=loc.get("column", 0) or 0,
                        severity=_normalize_severity(off.get("severity", "convention")),
                        category=off.get("cop_name", "") or "rubocop",
                        rule_id=off.get("cop_name", "") or "rubocop",
                        message=str(off.get("message", "")),
                    )
                )
    except (json.JSONDecodeError, TypeError, AttributeError):
        findings = []
    findings, truncated = _bound_findings(findings)
    return _static_evidence(
        "ruby", analyzer, files, None, None, findings, truncated
    )


def _analyze_php(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    """Analyze PHP with the image-bundled PHP_CodeSniffer 3.11.3 phar.

    PHPCS runs as ``php /opt/phpcs/phpcs.phar`` (the in-image PHP 8.4
    CLI) with the deterministic in-image ``phpcs-standard.xml`` passed
    via ``--standard=``, so analysis never depends on a repository's own
    PHPCS configuration. Its JSON report is normalized into the shared
    finding structure. PHPCS reports violations with a non-zero exit
    code, so only stdout is authoritative. A missing PHP CLI, phar, or
    bundled standard is reported as deterministic ``tool_unavailable``
    — never a host fallback.
    """
    analyzer = ANALYZER_NAMES["php"]
    if shutil.which("php") is None or not Path(PHPCS_PHAR).is_file():
        return _failure_evidence(
            "php",
            "tool_unavailable",
            analyzer,
            "phpcs (php /opt/phpcs/phpcs.phar) is not installed in the image",
        )
    if not Path(PHPCS_CONFIG).is_file():
        return _failure_evidence(
            "php",
            "tool_unavailable",
            analyzer,
            "phpcs standard configuration is not installed in the image",
        )
    abs_files = [_abs(scope_path, f) for f in files]
    try:
        output = _invoke(
            [
                "php",
                PHPCS_PHAR,
                f"--standard={PHPCS_CONFIG}",
                "--report=json",
                *abs_files,
            ],
            timeout,
        )
    except ToolUnavailable:
        return _failure_evidence("php", "tool_unavailable", analyzer)
    except ToolTimeout:
        return _failure_evidence("php", "timeout", analyzer)

    findings: list[dict[str, object]] = []
    try:
        for fp, file_data in json.loads(output or "{}").get("files", {}).items():
            for msg in file_data.get("messages", []):
                findings.append(
                    _finding(
                        file_path=_relative_path(fp, scope_path),
                        line=msg.get("line", 0) or 0,
                        column=msg.get("column", 0) or 0,
                        severity=_normalize_severity(msg.get("type", "error")),
                        category=msg.get("source", "") or "phpcs",
                        rule_id=msg.get("source", "") or "phpcs",
                        message=str(msg.get("message", "")),
                    )
                )
    except (json.JSONDecodeError, TypeError, AttributeError):
        findings = []
    findings, truncated = _bound_findings(findings)
    return _static_evidence(
        "php", analyzer, files, None, None, findings, truncated
    )


def _analyze_c(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    """Analyze C with the image-bundled Cppcheck 2.17.1.

    Cppcheck runs locally inside the image against the supplied
    ``.c``/``.h`` files with ``--language=c`` (pinned C mode — no
    language detection), the deterministic warning check set
    (``--enable=warning``; ``error`` is always enabled and not a valid
    ``--enable`` name), bounded configuration expansion
    (``--max-configs=1``), and ``--xml`` machine-readable output on
    stdout, which is normalized into the shared finding structure.
    Cppcheck's non-zero exit code reports violations, not failure, so
    analysis succeeded iff stdout contains a parseable ``<results>``
    XML document. A missing cppcheck binary is reported as deterministic
    ``tool_unavailable`` — never a host fallback.
    """
    analyzer = ANALYZER_NAMES["c"]
    if shutil.which("cppcheck") is None:
        return _failure_evidence(
            "c",
            "tool_unavailable",
            analyzer,
            "cppcheck is not installed in the image",
        )
    abs_files = [_abs(scope_path, f) for f in files]
    try:
        proc = subprocess.run(
            [
                "cppcheck",
                "--xml",
                "--quiet",
                "--language=c",
                "--enable=warning",
                "--max-configs=1",
                *abs_files,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return _failure_evidence("c", "tool_unavailable", analyzer)
    except subprocess.TimeoutExpired:
        return _failure_evidence("c", "timeout", analyzer)

    root = _load_cppcheck_xml((proc.stdout or "") + "\n" + (proc.stderr or ""))
    if root is None:
        if proc.returncode == 0:
            return _static_evidence("c", analyzer, files, None, None, [], False)
        detail = (proc.stderr or proc.stdout or "").strip()
        return _failure_evidence(
            "c",
            "error",
            analyzer,
            detail or f"cppcheck failed with exit code {proc.returncode}",
        )

    findings, truncated = _bound_findings(_cppcheck_findings(root, scope_path))
    return _static_evidence("c", analyzer, files, None, None, findings, truncated)


def _load_cppcheck_xml(output: str) -> ET.Element | None:
    """Parse Cppcheck's ``--xml`` report (defensively).

    Cppcheck's 2.x XML report is a single ``<results version="2">``
    document on stdout; a leading-log fallback extracts the first
    ``<results>`` block when other lines are present (``--quiet`` keeps
    the "Checking ..." progress lines off stdout). Returns None when no
    XML could be parsed (used to distinguish hard failures from
    violations).
    """
    try:
        return ET.fromstring(output)
    except ET.ParseError:
        pass
    match = re.search(r"<results\b.*?</results>", output, re.DOTALL)
    if match is None:
        return None
    try:
        return ET.fromstring(match.group(0))
    except ET.ParseError:
        return None


def _cppcheck_findings(root: ET.Element, scope_path: str) -> list[dict[str, object]]:
    """Normalize Cppcheck violations into the shared finding structure.

    Each ``<error id=... severity=... msg=...>`` with a ``<location
    file=... line=... column=...>`` becomes one finding; errors without
    any location cannot be mapped to a line and are skipped.
    """
    findings: list[dict[str, object]] = []
    errors = root.find("errors")
    if errors is None:
        return findings
    for err in errors.findall("error"):
        loc = err.find("location")
        if loc is None:
            continue
        raw_fp = loc.get("file", "")
        findings.append(
            _finding(
                file_path=_relative_path(raw_fp, scope_path),
                line=int(loc.get("line", 0) or 0),
                column=int(loc.get("column", 0) or 0),
                severity=_normalize_severity(err.get("severity", "error")),
                category=str(err.get("id", "") or "cppcheck"),
                rule_id=str(err.get("id", "") or "cppcheck"),
                message=str(err.get("msg", "")),
            )
        )
    return findings


_CLANG_TIDY_RE = re.compile(
    r"^(.+?):(\d+):(\d+):\s*(warning|error):\s*(.+?)(?:\s*\[([^\]]+)\])?$"
)


def _analyze_cpp(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    """Analyze C++ with clang-tidy + Cppcheck.

    clang-tidy (Debian ``clang-tidy-19``) is the primary analyzer: each
    ``.cpp``/``.cc``/``.cxx``/``.hpp``/``.hh``/``.hxx`` file is compiled
    as a standalone C++17 translation unit (``-- -x c++ -std=c++17``)
    with ``-Wall`` so the ``clang-diagnostic-*`` checks in the bundled
    config actually see compiler warnings, using the deterministic
    in-image config (``--config-file``) so a repository's own
    ``.clang-tidy`` is never read; its textual
    diagnostics on stderr are normalized into the shared finding
    structure (only diagnostics whose resolved path belongs to the
    discovered files are kept). Cppcheck 2.17.1 is an ADDITIONAL merged
    analyzer (``--language=c++`` ``--enable=warning``
    ``--max-configs=1`` ``--xml``) whose XML report is normalized the
    same way — best effort, like the semgrep merge: an unparseable
    cppcheck report silently contributes no findings while clang-tidy
    stays authoritative. Missing tools or the bundled config are
    reported as deterministic ``tool_unavailable``; clang-tidy timeouts
    are ``timeout`` — never a host fallback. Non-zero exit codes report
    violations, not failure.
    """
    analyzer = ANALYZER_NAMES["cpp"]
    for tool, detail in (
        ("clang-tidy-19", "clang-tidy-19 is not installed in the image"),
        ("cppcheck", "cppcheck is not installed in the image"),
    ):
        if shutil.which(tool) is None:
            return _failure_evidence("cpp", "tool_unavailable", analyzer, detail)
    if not Path(CLANG_TIDY_CONFIG).is_file():
        return _failure_evidence(
            "cpp",
            "tool_unavailable",
            analyzer,
            "clang-tidy configuration is not installed in the image",
        )
    abs_files = [_abs(scope_path, f) for f in files]

    findings: list[dict[str, object]] = []
    for abs_file in abs_files:
        try:
            proc = subprocess.run(
                [
                    "clang-tidy-19",
                    "--quiet",
                    f"--config-file={CLANG_TIDY_CONFIG}",
                    abs_file,
                "--",
                "-x",
                "c++",
                "-std=c++17",
                "-Wall",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return _failure_evidence("cpp", "tool_unavailable", analyzer)
        except subprocess.TimeoutExpired:
            return _failure_evidence("cpp", "timeout", analyzer)
        findings.extend(
            _clang_tidy_findings(
                proc.stderr + "\n" + proc.stdout, scope_path, abs_files
            )
        )

    try:
        cppcheck_proc = subprocess.run(
            [
                "cppcheck",
                "--xml",
                "--quiet",
                "--language=c++",
                "--enable=warning",
                "--max-configs=1",
                *abs_files,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return _failure_evidence("cpp", "tool_unavailable", analyzer)
    except subprocess.TimeoutExpired:
        return _failure_evidence("cpp", "timeout", analyzer)
    root = _load_cppcheck_xml(
        (cppcheck_proc.stdout or "") + "\n" + (cppcheck_proc.stderr or "")
    )
    if root is not None:
        findings.extend(_cppcheck_findings(root, scope_path))

    findings, truncated = _bound_findings(findings)
    return _static_evidence("cpp", analyzer, files, None, None, findings, truncated)


def _clang_tidy_findings(
    output: str, scope_path: str, abs_files: list[str]
) -> list[dict[str, object]]:
    """Normalize clang-tidy's textual diagnostics into findings.

    Matches ``path:line:column: warning/error: message [check-name]``
    lines. Diagnostics whose resolved path is not one of the discovered
    files (e.g. from system headers) are skipped. Lines without a
    bracketed check name use ``clang-tidy`` as the rule id.
    """
    abs_set = {str(Path(p).resolve()) for p in abs_files}
    findings: list[dict[str, object]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _CLANG_TIDY_RE.match(line)
        if not match:
            continue
        try:
            resolved = str(Path(match.group(1)).resolve())
        except OSError:
            continue
        if resolved not in abs_set:
            continue
        check = match.group(6) or "clang-tidy"
        findings.append(
            _finding(
                file_path=_relative_path(match.group(1), scope_path),
                line=int(match.group(2)),
                column=int(match.group(3)),
                severity=_normalize_severity(match.group(4)),
                category=check,
                rule_id=check,
                message=match.group(5),
            )
        )
    return findings


def _analyze_csharp(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    """Analyze C# with the .NET SDK 8.x Roslyn compiler + analyzers.

    ``dotnet build`` of the repository's own project/solution
    (implicit restore, ``-t:Rebuild`` so the SARIF error log is written
    even for previously compiled workspaces, ``-warnaserror:false``)
    runs the Roslyn compiler and
    the built-in .NET analyzers locally; the Roslyn csc task emits a
    machine-readable SARIF error log (``/p:ErrorLog=...``,
    ``/p:GenerateFullPaths=true``) capturing ALL compiler (CS*) and
    analyzer (CA*) diagnostics, which are normalized into the shared
    finding structure. The human-readable console text is never parsed.
    A project/solution is required (deterministic ``error`` when none
    exists — the same empty-content contract as before); a missing .NET
    SDK is ``tool_unavailable`` and a build timeout is ``timeout``.
    A non-zero build exit reports violations, not failure: analysis
    succeeded iff the error log is present and parseable.
    """
    analyzer = ANALYZER_NAMES["csharp"]
    abs_files = [_abs(scope_path, f) for f in files]
    root = Path(scope_path)
    csproj = sorted(root.rglob("*.csproj"))
    project_file = str(csproj[0]) if csproj else None
    if project_file is None:
        sln = sorted(root.rglob("*.sln"))
        project_file = str(sln[0]) if sln else None
    if project_file is None:
        return {
            **_failure_evidence("csharp", "error", analyzer),
            "error_message": "no .csproj or .sln file found in the contributor scope",
        }

    error_log = str(Path(tempfile.gettempdir()) / "csharp-sarif.xml")
    try:
        proc = subprocess.run(
            [
                "dotnet",
                "build",
                project_file,
                "-t:Rebuild",
                "-warnaserror:false",
                f"/p:ErrorLog={error_log}",
                "/p:GenerateFullPaths=true",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return _failure_evidence("csharp", "tool_unavailable", analyzer)
    except subprocess.TimeoutExpired:
        return _failure_evidence("csharp", "timeout", analyzer)

    findings = _load_csharp_error_log(error_log, abs_files, scope_path)
    if findings is None:
        if proc.returncode == 0:
            return _static_evidence("csharp", analyzer, files, None, None, [], False)
        detail = (proc.stderr or proc.stdout or "").strip()
        return _failure_evidence(
            "csharp",
            "error",
            analyzer,
            detail or f"dotnet build failed with exit code {proc.returncode}",
        )
    findings, truncated = _bound_findings(findings)
    return _static_evidence(
        "csharp", analyzer, files, None, None, findings, truncated
    )


def _load_csharp_error_log(
    path: str, abs_files: list[str], scope_path: str
) -> list[dict[str, object]] | None:
    """Parse the Roslyn SARIF error log (defensively).

    Returns normalized findings, or None when the error log is missing
    or unparseable (used to distinguish hard build failures from
    violations). Both the SARIF XML form (``<CheckResults>`` /
    ``<Check>`` with ``<Location file= line= column=>``, emitted by
    ``csc -errorlog:``) and the SARIF v2.1.0 JSON form (``runs`` /
    ``results``) are accepted. Diagnostics whose resolved path is not
    one of the discovered ``.cs`` files (e.g. generated assets) are
    skipped.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    abs_set = {str(Path(p).resolve()) for p in abs_files}
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return _sarif_json_findings(text, abs_set, scope_path)
    return _sarif_xml_findings(text, abs_set, scope_path)


def _normalize_sarif_path(raw: str) -> str:
    """Normalize a SARIF artifact uri/file path to a filesystem path."""
    path = raw.replace("\\", "/")
    if path.startswith("file://"):
        path = path[len("file://"):]
    return path


def _sarif_xml_findings(
    text: str, abs_set: set[str], scope_path: str
) -> list[dict[str, object]] | None:
    """Normalize a SARIF XML (``<CheckResults>``) error log into findings."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    findings: list[dict[str, object]] = []
    for check in root.findall("Check"):
        loc = check.find("Location")
        if loc is None:
            continue
        raw_fp = loc.get("file", "")
        try:
            if str(Path(_normalize_sarif_path(raw_fp)).resolve()) not in abs_set:
                continue
        except OSError:
            continue
        rule_id = str(check.get("id", "") or "roslyn")
        findings.append(
            _finding(
                file_path=_relative_path(raw_fp, scope_path),
                line=int(loc.get("line", 0) or 0),
                column=int(loc.get("column", 0) or 0),
                severity=_normalize_severity(check.get("severity", "error")),
                category=rule_id,
                rule_id=rule_id,
                message=str(check.get("message", "")),
            )
        )
    return findings


def _sarif_json_findings(
    text: str, abs_set: set[str], scope_path: str
) -> list[dict[str, object]] | None:
    """Normalize a SARIF JSON error log into findings.

    Accepts both the SARIF v2.1.0 form (``locations`` with
    ``physicalLocation``) and the SARIF v1.0.0 form emitted by the
    Roslyn csc task (``locations`` with ``resultFile``)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    findings: list[dict[str, object]] = []
    runs = data.get("runs")
    if not isinstance(runs, list):
        return None
    for run in runs:
        if not isinstance(run, dict):
            continue
        results = run.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            rule_id = str(result.get("ruleId", "") or "roslyn")
            message = result.get("message")
            if isinstance(message, dict):
                message = message.get("text", "")
            elif not isinstance(message, str):
                message = ""
            locations = result.get("locations")
            if not isinstance(locations, list) or not locations:
                continue
            loc = locations[0]
            if not isinstance(loc, dict):
                continue
            physical = loc.get("physicalLocation")
            if isinstance(physical, dict):
                artifact = physical.get("artifactLocation")
                region = physical.get("region")
            else:
                result_file = loc.get("resultFile")
                artifact = result_file if isinstance(result_file, dict) else None
                region = result_file.get("region") if isinstance(result_file, dict) else None
            artifact = artifact if isinstance(artifact, dict) else {}
            region = region if isinstance(region, dict) else {}
            raw_fp = str(artifact.get("uri", "") or "")
            if not raw_fp:
                continue
            try:
                if str(Path(_normalize_sarif_path(raw_fp)).resolve()) not in abs_set:
                    continue
            except OSError:
                continue
            findings.append(
                _finding(
                    file_path=_relative_path(raw_fp, scope_path),
                    line=int(region.get("startLine", 0) or 0),
                    column=int(region.get("startColumn", 0) or 0),
                    severity=_normalize_severity(
                        str(result.get("level", "warning"))
                    ),
                    category=rule_id,
                    rule_id=rule_id,
                    message=str(message),
                )
            )
    return findings


def _analyze_kotlin(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    """Analyze Kotlin with the image-bundled Detekt 1.23.8 CLI.

    Detekt runs as ``java -jar`` (the OpenJDK 21 JRE) against the FULL
    workspace ``.kt``/``.kts`` files with its deterministic built-in
    default configuration (the CLI never reads a repository's own
    ``detekt.yml``) and writes a machine-readable SARIF v2.1.0 report
    (``--report "sarif:<path>"``) which is normalized into the shared
    finding structure. Detekt is deliberately run without type
    resolution (no classpath), so analysis needs no restored
    dependencies. A non-zero exit code reports violations, not failure:
    analysis succeeded iff the report is present and parseable. Missing
    JVM or Detekt JAR is reported as deterministic ``tool_unavailable``
    — never a host fallback.
    """
    analyzer = ANALYZER_NAMES["kotlin"]
    if shutil.which("java") is None or not Path(DETEKT_JAR).is_file():
        return _failure_evidence(
            "kotlin",
            "tool_unavailable",
            analyzer,
            "detekt (java -jar) is not installed in the image",
        )
    report_path = str(Path(tempfile.gettempdir()) / "detekt-report.sarif")
    try:
        proc = subprocess.run(
            [
                "java",
                "-jar",
                DETEKT_JAR,
                "--input",
                scope_path,
                "--report",
                f"sarif:{report_path}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return _failure_evidence("kotlin", "tool_unavailable", analyzer)
    except subprocess.TimeoutExpired:
        return _failure_evidence("kotlin", "timeout", analyzer)

    abs_set = {str(Path(p).resolve()) for p in [_abs(scope_path, f) for f in files]}
    try:
        report_text = Path(report_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        if proc.returncode == 0:
            return _static_evidence("kotlin", analyzer, files, None, None, [], False)
        detail = (proc.stderr or proc.stdout or "").strip()
        return _failure_evidence(
            "kotlin",
            "error",
            analyzer,
            detail or f"detekt failed with exit code {proc.returncode}",
        )

    findings = _sarif_json_findings(report_text, abs_set, scope_path)
    if findings is None:
        detail = (proc.stderr or proc.stdout or "").strip()
        return _failure_evidence(
            "kotlin",
            "error",
            analyzer,
            detail or "detekt report could not be parsed",
        )
    findings, truncated = _bound_findings(findings)
    return _static_evidence(
        "kotlin", analyzer, files, None, None, findings, truncated
    )


def _analyze_swift(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    """Analyze Swift with the image-bundled SwiftLint 0.58.2.

    SwiftLint runs locally against the workspace ``.swift`` files with a
    deterministic in-image config (``--config``) so a repository's own
    ``.swiftlint.yml`` is never read, and emits a machine-readable JSON
    report (``--reporter json``) on stdout which is normalized into the
    shared finding structure. A non-zero exit code reports violations,
    not failure: analysis succeeded iff the JSON report is parseable. A
    missing swiftlint binary or its bundled config is reported as
    deterministic ``tool_unavailable`` — never a host fallback.
    """
    analyzer = ANALYZER_NAMES["swift"]
    if shutil.which("swiftlint") is None:
        return _failure_evidence(
            "swift",
            "tool_unavailable",
            analyzer,
            "swiftlint is not installed in the image",
        )
    if not Path(SWIFTLINT_CONFIG).is_file():
        return _failure_evidence(
            "swift",
            "tool_unavailable",
            analyzer,
            "swiftlint configuration is not installed in the image",
        )
    abs_files = [_abs(scope_path, f) for f in files]
    try:
        proc = subprocess.run(
            [
                "swiftlint",
                "lint",
                "--quiet",
                "--reporter",
                "json",
                "--config",
                SWIFTLINT_CONFIG,
                *abs_files,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return _failure_evidence("swift", "tool_unavailable", analyzer)
    except subprocess.TimeoutExpired:
        return _failure_evidence("swift", "timeout", analyzer)

    entries: object = []
    try:
        entries = json.loads(proc.stdout or "[]")
    except (json.JSONDecodeError, TypeError):
        entries = None
    if not isinstance(entries, list):
        if proc.returncode == 0:
            return _static_evidence("swift", analyzer, files, None, None, [], False)
        detail = (proc.stderr or proc.stdout or "").strip()
        return _failure_evidence(
            "swift",
            "error",
            analyzer,
            detail or f"swiftlint failed with exit code {proc.returncode}",
        )

    findings: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rule_id = str(entry.get("rule_id", "") or "swiftlint")
        findings.append(
            _finding(
                file_path=_relative_path(str(entry.get("file", "")), scope_path),
                line=int(entry.get("line", 0) or 0),
                column=int(entry.get("character", 0) or 0),
                severity=_normalize_severity(str(entry.get("severity", "warning"))),
                category=rule_id,
                rule_id=rule_id,
                message=str(entry.get("reason", "")),
            )
        )
    findings, truncated = _bound_findings(findings)
    return _static_evidence(
        "swift", analyzer, files, None, None, findings, truncated
    )


def _analyze_dart(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    """Analyze Dart with the official Dart Analyzer (Dart SDK 3.7.3).

    ``dart analyze`` takes a single directory, not a file list, so the
    whole scope is analyzed with machine-readable JSON output
    (``--format=json``); diagnostics are mapped into the shared finding
    structure, filtered to the discovered ``.dart`` files. Flutter
    projects use the same analyzer — there is no separate Flutter
    analyzer. Like the C# stack, the analyzer is authoritative and
    deterministic per repository: the repository's own project
    configuration (``pubspec.yaml`` / ``analysis_options.yaml``) is
    respected with standard ``dart analyze`` semantics. A non-zero exit
    code reports violations, not failure: analysis succeeded iff the
    JSON report is parseable. A missing ``dart`` binary is reported as
    deterministic ``tool_unavailable`` — never a host fallback.
    """
    analyzer = ANALYZER_NAMES["dart"]
    if shutil.which("dart") is None:
        return _failure_evidence(
            "dart",
            "tool_unavailable",
            analyzer,
            "dart is not installed in the image",
        )
    abs_files = [_abs(scope_path, f) for f in files]
    abs_set = set(abs_files)
    try:
        proc = subprocess.run(
            ["dart", "analyze", "--format=json", scope_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return _failure_evidence("dart", "tool_unavailable", analyzer)
    except subprocess.TimeoutExpired:
        return _failure_evidence("dart", "timeout", analyzer)

    result: object = {}
    try:
        result = json.loads(proc.stdout or "{}")
    except (json.JSONDecodeError, TypeError):
        result = None
    if not isinstance(result, dict):
        if proc.returncode == 0:
            return _static_evidence("dart", analyzer, files, None, None, [], False)
        detail = (proc.stderr or proc.stdout or "").strip()
        return _failure_evidence(
            "dart",
            "error",
            analyzer,
            detail or f"dart analyze failed with exit code {proc.returncode}",
        )

    findings: list[dict[str, object]] = []
    diagnostics = result.get("diagnostics") if isinstance(result, dict) else []
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            location = diagnostic.get("location")
            if not isinstance(location, dict):
                continue
            raw_fp = str(location.get("file", "") or "")
            if raw_fp not in abs_set:
                continue
            start = location.get("range")
            if isinstance(start, dict):
                start = start.get("start")
            start = start if isinstance(start, dict) else {}
            rule_id = str(diagnostic.get("code", "") or "dart")
            message = str(diagnostic.get("problemMessage", "") or "")
            correction = str(diagnostic.get("correctionMessage", "") or "")
            if message and correction:
                message = f"{message} {correction}"
            findings.append(
                _finding(
                    file_path=_relative_path(raw_fp, scope_path),
                    line=int(start.get("line", 0) or 0),
                    column=int(start.get("column", 0) or 0),
                    severity=_normalize_severity(
                        str(diagnostic.get("severity", "WARNING"))
                    ),
                    category=rule_id,
                    rule_id=rule_id,
                    message=message,
                )
            )
    findings, truncated = _bound_findings(findings)
    return _static_evidence(
        "dart", analyzer, files, None, None, findings, truncated
    )


def _load_sqlfluff_json(output: str) -> list[dict[str, object]] | None:
    """Parse SQLFluff's JSON report (defensively).

    The report is a JSON array of per-file objects with a
    ``violations`` list. Strict parsing is attempted first; if the
    output carries stray non-JSON lines, the first ``[`` to the last
    ``]`` is parsed instead. Returns None when no array could be
    parsed.
    """
    try:
        data = json.loads(output or "[]")
    except (json.JSONDecodeError, TypeError):
        try:
            start = output.find("[")
            end = output.rfind("]")
            if start != -1 and end > start:
                data = json.loads(output[start : end + 1])
            else:
                return None
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, list) else None


def _analyze_sql(scope_path: str, files: list[str], timeout: int) -> dict[str, object]:
    """Analyze SQL with the image-bundled SQLFluff 3.4.2.

    SQLFluff runs ``lint`` against the supplied ``.sql`` files with the
    deterministic in-image config (``--config``) so a repository's own
    ``.sqlfluff`` is never read (dialect pinned to ANSI, raw templater
    — no template rendering). Parsing and linting are fully offline and
    local; no database connection is ever required. The machine-readable
    JSON report (``--format json``) is normalized into the shared
    finding structure (rule id and severity ``warning``, since SQLFluff
    violations carry no per-finding severity). A non-zero exit code
    reports violations, not failure: analysis succeeded iff the JSON
    report is parseable. A missing sqlfluff binary or its bundled
    config is reported as deterministic ``tool_unavailable`` — never a
    host fallback.
    """
    analyzer = ANALYZER_NAMES["sql"]
    if shutil.which("sqlfluff") is None:
        return _failure_evidence(
            "sql",
            "tool_unavailable",
            analyzer,
            "sqlfluff is not installed in the image",
        )
    if not Path(SQLFLUFF_CONFIG).is_file():
        return _failure_evidence(
            "sql",
            "tool_unavailable",
            analyzer,
            "sqlfluff configuration is not installed in the image",
        )
    abs_files = [_abs(scope_path, f) for f in files]
    try:
        proc = subprocess.run(
            [
                "sqlfluff",
                "lint",
                "--format",
                "json",
                "--config",
                SQLFLUFF_CONFIG,
                *abs_files,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return _failure_evidence("sql", "tool_unavailable", analyzer)
    except subprocess.TimeoutExpired:
        return _failure_evidence("sql", "timeout", analyzer)

    entries: object = _load_sqlfluff_json(proc.stdout)
    if entries is None:
        if proc.returncode == 0:
            return _static_evidence("sql", analyzer, files, None, None, [], False)
        detail = (proc.stderr or proc.stdout or "").strip()
        return _failure_evidence(
            "sql",
            "error",
            analyzer,
            detail or f"sqlfluff failed with exit code {proc.returncode}",
        )

    findings: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_fp = str(entry.get("filepath", "") or "")
        violations = entry.get("violations")
        if not isinstance(violations, list):
            continue
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            rule_id = str(violation.get("code", "") or "sqlfluff")
            findings.append(
                _finding(
                    file_path=_relative_path(raw_fp, scope_path),
                    line=int(violation.get("start_line_no", 0) or 0),
                    column=int(violation.get("start_line_pos", 0) or 0),
                    severity=_normalize_severity("warning"),
                    category=rule_id,
                    rule_id=rule_id,
                    message=str(
                        violation.get("description", "") or ""
                    ),
                )
            )
    findings, truncated = _bound_findings(findings)
    return _static_evidence(
        "sql", analyzer, files, None, None, findings, truncated
    )


_ADAPTERS = {
    "python": _analyze_python,
    "javascript": _analyze_js,
    "typescript": _analyze_typescript,
    "vue": _analyze_vue,
    "java": _analyze_java,
    "go": _analyze_go,
    "rust": _analyze_rust,
    "ruby": _analyze_ruby,
    "php": _analyze_php,
    "c": _analyze_c,
    "cpp": _analyze_cpp,
    "kotlin": _analyze_kotlin,
    "swift": _analyze_swift,
    "dart": _analyze_dart,
    "sql": _analyze_sql,
    "csharp": _analyze_csharp,
}


# ---------------------------------------------------------------------------
# Graphify
# ---------------------------------------------------------------------------


def _graph_evidence(
    status: str, message: str | None = None
) -> dict[str, object]:
    return {
        "status": status,
        "analyzer": "graphify",
        "graph_available": status == "success",
        "node_count": 0,
        "edge_count": 0,
        "relations": {},
        "relation_count": 0,
        "relation_truncated": False,
        "inheritance_depth": None,
        "coupling": None,
        "circular_import_count": 0,
        "error_message": message[:MAX_ERROR_CHARS] if message else None,
    }


def _graph_success(
    node_count: int,
    edge_count: int,
    relations: dict[str, int],
    relation_truncated: bool,
    inheritance_depth: int | None,
    coupling: float | None,
    circular_import_count: int,
    nodes: list[dict],
    edges: list[dict],
) -> dict[str, object]:
    return {
        "status": "success",
        "analyzer": "graphify",
        "graph_available": True,
        "node_count": node_count,
        "edge_count": edge_count,
        "relations": relations,
        "relation_count": len(relations),
        "relation_truncated": relation_truncated,
        "inheritance_depth": inheritance_depth,
        "coupling": coupling,
        "circular_import_count": circular_import_count,
        "nodes": nodes,
        "edges": edges,
        "error_message": None,
    }


def _inheritance_depth(edges: list[dict]) -> int | None:
    parents: dict[str, list[str]] = {}
    children: set[str] = set()
    for e in edges:
        relation = str(e.get("relation", ""))
        if relation in ("inherits", "extends", "implements"):
            child = str(e.get("source", ""))
            parent = str(e.get("target", ""))
            parents.setdefault(child, []).append(parent)
            children.add(child)
    if not parents:
        return None
    max_depth = 0
    memo: dict[str, int] = {}

    def depth(node: str, visited: set[str]) -> int:
        if node in memo:
            return memo[node]
        if node in visited:
            return 0
        visited.add(node)
        deepest = 0
        for parent in sorted(parents.get(node, [])):
            deepest = max(deepest, 1 + depth(parent, visited))
        visited.discard(node)
        memo[node] = deepest
        return deepest

    for child in sorted(children):
        max_depth = max(max_depth, depth(child, set()))
    return max_depth if max_depth > 0 else None


def _circular_import_count(edges: list[dict]) -> int:
    graph: dict[str, list[str]] = {}
    for e in edges:
        if str(e.get("relation", "")) in ("imports", "import"):
            graph.setdefault(str(e.get("source", "")), []).append(
                str(e.get("target", ""))
            )
    visited: set[str] = set()
    rec_stack: list[str] = []
    cycles = 0

    def dfs(node: str) -> None:
        nonlocal cycles
        visited.add(node)
        rec_stack.append(node)
        for neighbor in sorted(graph.get(node, [])):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                cycles += 1
        rec_stack.pop()

    for node in sorted(graph):
        if node not in visited:
            dfs(node)
    return cycles


def run_graph_extract(repo_path: str, timeout: int) -> dict[str, object]:
    """Run the external ``graphify`` binary against the FULL repository.

    Extraction only: the full-repository graph artifact is produced and
    verified, and NO contributor scope is consulted in this step. The
    contributor graph filtering happens strictly afterwards in
    ``run_graph_select``, which reads the scope manifest written by
    ``filter_contributor_code``. Splitting extraction from selection
    lets the MCP orchestration run Graphify on the full repository
    concurrently with the contributor filter while guaranteeing that
    graph filtering always follows Graphify. The complete graph payload
    from ``graph.json`` (``nodes``, ``edges``, and any other fields
    Graphify produced) is preserved in the returned evidence under
    ``graph`` — the graph structure is never reduced to counts.
    """
    graphify_bin = shutil.which("graphify")
    if graphify_bin is None:
        return _graph_evidence(
            "tool_unavailable", "graphify binary not found in the sandbox image"
        )
    try:
        proc = subprocess.run(
            [graphify_bin, "extract", repo_path, "--code-only", "--no-cluster"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return _graph_evidence(
            "tool_unavailable", "graphify binary not found in the sandbox image"
        )
    except subprocess.TimeoutExpired:
        return _graph_evidence("timeout", "graphify timed out")
    if proc.returncode != 0:
        return _graph_evidence("error", (proc.stderr or proc.stdout or "")[:MAX_ERROR_CHARS])

    graph_file = Path(repo_path) / "graphify-out" / "graph.json"
    if not graph_file.is_file():
        return _graph_evidence("error", "graphify did not produce graph.json")

    try:
        data = json.loads(graph_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return _graph_evidence("error", f"cannot read graph: {exc}")

    raw_nodes = data.get("nodes", []) if isinstance(data, dict) else []
    node_count = sum(1 for n in raw_nodes if isinstance(n, dict))
    return {
        "status": "extracted",
        "analyzer": "graphify",
        "graph_available": True,
        "node_count": node_count,
        "error_message": None,
        "graph": data if isinstance(data, dict) else {},
    }


def run_graph_select(repo_path: str, timeout: int) -> dict[str, object]:
    """Select the contributor-related evidence from the FULL repository graph.

    Reads the graph artifact produced by ``run_graph_extract``
    (``<repo>/graphify-out/graph.json``) and the contributor scope
    manifest (``<repo>/../scope.manifest``, written by
    ``filter_contributor_code`` inside the sandbox), then selects the
    contributor-related nodes and relations from the full graph: nodes
    whose ``source_file`` is in the scope manifest plus the relations
    touching those nodes — so the contributor's architectural
    relationships with the rest of the repository are preserved. The
    selected subgraph (its nodes and edges) is preserved in the returned
    evidence alongside the derived counts and relation summaries.
    """
    graph_file = Path(repo_path) / "graphify-out" / "graph.json"
    if not graph_file.is_file():
        return _graph_evidence("error", "graphify did not produce graph.json")

    try:
        data = json.loads(graph_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return _graph_evidence("error", f"cannot read graph: {exc}")

    raw_nodes = data.get("nodes", []) if isinstance(data, dict) else []
    raw_edges = data.get("edges", []) if isinstance(data, dict) else []
    if not isinstance(raw_edges, list):
        raw_edges = data.get("links", []) if isinstance(data, dict) else []
    edges = [e for e in raw_edges if isinstance(e, dict)]

    manifest = _read_scope_manifest(repo_path)
    nodes, edges = _select_contributor_subgraph(raw_nodes, edges, manifest)

    relations: dict[str, int] = {}
    for e in edges:
        rel = str(e.get("relation", ""))
        relations[rel] = relations.get(rel, 0) + 1
    sorted_relations = dict(sorted(relations.items()))
    relation_truncated = len(sorted_relations) > MAX_RELATIONS
    bounded_relations = dict(list(sorted_relations.items())[:MAX_RELATIONS])

    coupling = round(len(edges) / len(nodes), 4) if nodes else None

    return _graph_success(
        node_count=len(nodes),
        edge_count=len(edges),
        relations=bounded_relations,
        relation_truncated=relation_truncated,
        inheritance_depth=_inheritance_depth(edges),
        coupling=coupling,
        circular_import_count=_circular_import_count(edges),
        nodes=nodes,
        edges=edges,
    )


def run_graph(repo_path: str, timeout: int) -> dict[str, object]:
    """Run Graphify against the FULL repository and select contributor evidence.

    Single-call mode (extraction + contributor selection in one step),
    kept for the legacy ``graph`` command. The orchestrated flow splits
    this into ``run_graph_extract`` followed by ``run_graph_select`` so
    Graphify can run concurrently with the contributor filter while
    graph filtering always follows Graphify.
    """
    extracted = run_graph_extract(repo_path, timeout)
    if extracted.get("status") != "extracted":
        return extracted
    return run_graph_select(repo_path, timeout)


def _read_scope_manifest(repo_path: str) -> frozenset[str]:
    """Read the contributor scope manifest written by the filter.

    The manifest lives at ``<repo_path>/../scope.manifest`` inside the
    sandbox (``/workspace/scope.manifest``). It is written by
    ``filter_contributor_code`` in the same container, so it is never
    missing when a scope is registered. Missing/empty input yields an
    empty manifest (deterministic empty evidence).
    """
    manifest_file = Path(repo_path).resolve().parent / "scope.manifest"
    try:
        return frozenset(
            line.strip()
            for line in manifest_file.read_text().splitlines()
            if line.strip()
        )
    except OSError:
        return frozenset()


def _select_contributor_subgraph(
    raw_nodes: list, edges: list[dict], manifest: frozenset[str]
) -> tuple[list[dict], list[dict]]:
    """Select the contributor-related nodes and relations from the FULL graph.

    A node is contributor-related when its ``source_file`` (repo-relative)
    is listed in the registered scope manifest. A relation is
    contributor-related when at least one of its endpoints belongs to a
    contributor-related node — relations between the contributor's code
    and the rest of the repository (e.g. imports of external modules) are
    therefore preserved. Deterministic: inputs are sorted and normalized.
    """
    node_by_ref: dict[str, dict] = {}
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        if node.get("id") is not None:
            node_by_ref[str(node["id"])] = node
        if node.get("label") is not None:
            node_by_ref.setdefault(str(node["label"]), node)

    def node_in_scope(node: dict) -> bool:
        raw_file = str(node.get("source_file") or node.get("file") or "")
        return _normalize_scope_path(raw_file) in manifest

    scope_nodes = [n for n in raw_nodes if isinstance(n, dict) and node_in_scope(n)]
    scope_refs = {str(n.get("id")) for n in scope_nodes if n.get("id") is not None}
    scope_refs |= {str(n.get("label")) for n in scope_nodes if n.get("label") is not None}

    scope_edges = [
        e
        for e in edges
        if str(e.get("source")) in scope_refs or str(e.get("target")) in scope_refs
    ]
    scope_edges.sort(
        key=lambda e: (str(e.get("source")), str(e.get("target")), str(e.get("relation")))
    )
    return scope_nodes, scope_edges


def _normalize_scope_path(path: str) -> str:
    """Normalize a graph node file path for manifest matching."""
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_static(language: str, scope_path: str, timeout: int) -> dict[str, object]:
    """Discover analyzable files in the repository workspace and analyze them."""
    canonical = LANGUAGE_ALIASES.get(language.strip().lower())
    if canonical is None:
        return _unsupported(language)
    analyzer = ANALYZER_NAMES[canonical]
    files = _discover_files(scope_path, EXTENSIONS[canonical])
    if not files:
        return _no_analyzable(canonical, analyzer)
    return _ADAPTERS[canonical](scope_path, files, timeout)


def emit(evidence: dict[str, object]) -> None:
    """Print the evidence marker and one JSON object (deterministic keys)."""
    print("EVIDENCE_OK")
    print(json.dumps(evidence))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analysis_runner")
    parser.add_argument(
        "kind",
        choices=("static", "graph", "graph_extract", "graph_select"),
    )
    parser.add_argument("path")
    parser.add_argument("--language", default="")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)

    try:
        if args.kind == "static":
            evidence = run_static(args.language, args.path, args.timeout)
        elif args.kind == "graph_extract":
            evidence = run_graph_extract(args.path, args.timeout)
        elif args.kind == "graph_select":
            evidence = run_graph_select(args.path, args.timeout)
        else:
            evidence = run_graph(args.path, args.timeout)
        emit(evidence)
        return 0
    except Exception as exc:  # noqa: BLE001 - fail closed: bounded error evidence, exit non-zero
        emit(
            {
                "status": "error",
                "analyzer": "analysis-runner",
                "error_message": str(exc)[:MAX_ERROR_CHARS],
            }
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
