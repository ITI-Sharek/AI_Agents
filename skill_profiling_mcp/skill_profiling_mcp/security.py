"""Security-sensitive helpers for the Skill Profiling MCP server.

Phase 4/5: repository URL validation, GitHub PAT validation, identifier
validation, and secret redaction. Phase 6: workspace path and language
validation. Phase 22: ``REPO_WORKSPACE_PATH`` (the full repository
workspace) is an analyzable workspace alongside the contributor scope —
static analysis runs against the full repository, Graphify selects
contributor-related evidence from the full graph. Repository input is
untrusted and must be validated before it is passed to the sandbox.
"""

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

MAX_PAT_LENGTH = 200
MAX_BRANCH_LENGTH = 100
MAX_LANGUAGE_LENGTH = 32

# The full repository workspace cloned by ``acquire_repository`` inside
# the sandbox (``/workspace/repo``). Phase 22: ``analyze_static`` analyzes
# this workspace — there is no contributor file filtering before static
# analysis.
REPO_WORKSPACE_PATH = "/workspace/repo"

# The contributor-scoped workspace built by ``filter_contributor_code``.
# ``analyze_graph`` validates its ``workspace_path`` against this scope.
SCOPE_WORKSPACE_PATH = "/workspace/scope"

_SEGMENT_RE = re.compile(r"[A-Za-z0-9._-]+")
_BRANCH_RE = re.compile(r"[A-Za-z0-9._/-]+")
_REPOSITORY_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)+")
_CONTRIBUTOR_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9]([A-Za-z0-9-]{0,38})")
_SANDBOX_IDENTIFIER_RE = re.compile(r"skill-profiling-mcp-[0-9a-f]{12}")


class InvalidRepositoryUrlError(ValueError):
    """Raised when a repository URL is malformed, unsupported, or unsafe."""


@dataclass(frozen=True)
class RepositoryReference:
    """A validated repository URL plus its normalized identifier."""

    url: str
    identifier: str


def parse_repository_url(repo_url: str) -> RepositoryReference:
    """Validate an http(s) repository URL and return a safe reference.

    Rejects non-http(s) schemes, credentials embedded in the URL, query
    strings, fragments, empty paths, and unsafe path segments.
    """
    if not isinstance(repo_url, str) or not repo_url.strip():
        raise InvalidRepositoryUrlError("repository URL is required")

    url = repo_url.strip()
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise InvalidRepositoryUrlError("repository URL is malformed") from exc

    if parts.scheme not in ("http", "https"):
        raise InvalidRepositoryUrlError(
            "only http(s) repository URLs are supported"
        )
    if not parts.hostname:
        raise InvalidRepositoryUrlError("repository URL has no host")
    if parts.username is not None or parts.password is not None:
        raise InvalidRepositoryUrlError(
            "credentials embedded in the URL are not allowed"
        )
    if parts.query or parts.fragment:
        raise InvalidRepositoryUrlError(
            "query strings and fragments are not allowed"
        )

    path = parts.path.rstrip("/")
    if not path.startswith("/") or len(path) < 2:
        raise InvalidRepositoryUrlError("repository URL has no repository path")

    identifier = path[1:]
    identifier = identifier.removesuffix(".git")
    segments = identifier.split("/")
    if len(segments) < 2:
        raise InvalidRepositoryUrlError(
            "repository identifier must include an owner/name"
        )
    for segment in segments:
        if segment in ("", ".", "..") or not _SEGMENT_RE.fullmatch(segment):
            raise InvalidRepositoryUrlError(
                "repository identifier contains an unsafe segment"
            )

    return RepositoryReference(url=url, identifier=identifier)


def validate_github_pat(github_pat: str | None) -> str | None:
    """Validate a request-scoped GitHub PAT. Returns None when absent."""
    if github_pat is None or github_pat == "":
        return None
    if not isinstance(github_pat, str):
        raise TypeError("github_pat must be a string")
    if any(ch.isspace() for ch in github_pat):
        raise ValueError("github_pat must not contain whitespace")
    if len(github_pat) > MAX_PAT_LENGTH:
        raise ValueError("github_pat is too long")
    return github_pat


def safe_branch_name(branch: str | None) -> str | None:
    """Return a branch name only if it looks like a safe ref name."""
    if not branch or not isinstance(branch, str):
        return None
    if branch in (".", ".."):
        return None
    if len(branch) > MAX_BRANCH_LENGTH:
        return None
    if not _BRANCH_RE.fullmatch(branch):
        return None
    return branch


def validate_repository_identifier(identifier: str) -> str:
    """Validate an ``owner/name``-style repository identifier."""
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("repository_identifier is required")
    identifier = identifier.strip()
    if not _REPOSITORY_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError("repository_identifier is invalid")
    return identifier


def validate_contributor_identifier(contributor: str) -> str:
    """Validate a GitHub login-style contributor identifier."""
    if not isinstance(contributor, str) or not contributor.strip():
        raise ValueError("contributor_identifier is required")
    contributor = contributor.strip()
    if not _CONTRIBUTOR_IDENTIFIER_RE.fullmatch(contributor):
        raise ValueError("contributor_identifier is invalid")
    return contributor


def validate_sandbox_identifier(sandbox_identifier: str) -> str:
    """Validate a sandbox identifier returned by ``acquire_repository``."""
    if not isinstance(sandbox_identifier, str) or not sandbox_identifier.strip():
        raise ValueError("sandbox_identifier is required")
    sandbox_identifier = sandbox_identifier.strip()
    if not _SANDBOX_IDENTIFIER_RE.fullmatch(sandbox_identifier):
        raise ValueError("sandbox_identifier is invalid")
    return sandbox_identifier


def validate_workspace_path(workspace_path: str) -> str:
    """Validate an analyzable sandbox workspace path.

    The only analyzable workspaces are the full repository workspace
    (``/workspace/repo``) and the contributor scope produced by
    ``filter_contributor_code`` (``/workspace/scope``). Anything else is
    rejected. The tool-level invariant decides which workspace each
    analyzer accepts: ``analyze_static`` requires the full repository
    workspace, ``analyze_graph`` requires the registered contributor
    scope workspace.
    """
    if not isinstance(workspace_path, str) or not workspace_path.strip():
        raise ValueError("workspace_path is required")
    workspace_path = workspace_path.strip()
    if workspace_path not in (REPO_WORKSPACE_PATH, SCOPE_WORKSPACE_PATH):
        raise ValueError(
            "workspace_path is invalid: only the full repository workspace "
            "or the contributor scope workspace are analyzable"
        )
    return workspace_path


def validate_language(language: str) -> str:
    """Validate the language argument; returns it lowercased.

    Support itself is decided by the analysis layer (deterministic
    ``unsupported_language`` evidence), this only rejects malformed input.
    """
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language is required")
    language = language.strip()
    if any(ch.isspace() for ch in language):
        raise ValueError("language must not contain whitespace")
    if len(language) > MAX_LANGUAGE_LENGTH:
        raise ValueError("language is too long")
    return language.lower()


def redact(secret: str | None, text: str) -> str:
    """Replace every occurrence of a secret in text with a placeholder.

    Used as defense in depth so a PAT never leaks through error messages
    or logs.
    """
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")


__all__ = [
    "REPO_WORKSPACE_PATH",
    "SCOPE_WORKSPACE_PATH",
    "InvalidRepositoryUrlError",
    "RepositoryReference",
    "parse_repository_url",
    "redact",
    "safe_branch_name",
    "validate_contributor_identifier",
    "validate_github_pat",
    "validate_language",
    "validate_repository_identifier",
    "validate_sandbox_identifier",
    "validate_workspace_path",
]
