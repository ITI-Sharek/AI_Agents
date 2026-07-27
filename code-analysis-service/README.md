# Code Analysis Service

A standalone HTTP analysis service that exposes code quality
and structure analysis tools via a REST API.

**Scope — quality & structure only.** This service reports cyclomatic
complexity, maintainability index, lint violations, inheritance
relationships, coupling hints, and circular import chains. It NEVER
detects, reports, or infers framework, library, or ORM names.

## Transport

The service runs as a FastAPI application over HTTP via uvicorn.

## Dependencies

- Python ≥ 3.11
- `fastapi`
- `pydantic`
- `uvicorn`

Each language adapter requires its underlying analysis tool to be
installed on the system `PATH`:

| Language   | Tool(s) required            |
|------------|-----------------------------|
| Python     | `radon`, `pylint`           |
| JavaScript | `eslint`                    |
| TypeScript | `eslint`                    |
| Java       | `checkstyle`                |
| Go         | `gocyclo`, `golangci-lint`  |
| Rust       | `clippy-driver`             |
| Ruby       | `rubocop`                   |
| PHP        | `phpcs`                     |
| C#         | `dotnet` SDK                |

## Usage

```bash
# Install
pip install .

# Run the API server
code-analysis-api
```

The server exposes three endpoints:

### `GET /health`

Returns `{"status": "ok"}`.

### `POST /analyze/repo`

**Inputs:**
- `repo_url` (str) — URL of the repository
- `language` (str) — one of: python, javascript, typescript, java, go,
  rust, ruby, php, csharp
- `requested_tools` (list[str]) — `["static_analysis"]`, `["graph_relations"]`, or both
- `pat` (str | None) — optional GitHub PAT for private repos

**Returns:** JSON-serialised `AnalysisResult` with `static_analysis`
and `graph_relations` fields.

### `POST /analyze/static`

**Inputs:**
- `repo_path` (str) — path to the local repository
- `language` (str) — one of: python, javascript, typescript, java, go,
  rust, ruby, php, csharp
- `file_paths` (list[str]) — files to analyse (relative to repo_path or
  absolute)
- `timeout` (int, default 60) — per-tool subprocess timeout in seconds

**Returns:** JSON-serialised `StaticAnalysisEvidence` with fields:
`status`, `language`, `files_analyzed`, `complexity`,
`maintainability_index`, `issues[]`, `structure`.

### `POST /analyze/graph`

**Inputs:**
- `repo_path` (str) — path to the cloned repository

**Returns:** JSON-serialised `GraphRelationsEvidence`.

## Local config suppression

| Tool           | Flag used                          | Notes                                    |
|----------------|------------------------------------|------------------------------------------|
| radon          | —                                  | No config file support                   |
| pylint         | `--rcfile=/dev/null`               | Unix-only; loads no local rcfile         |
| eslint         | `--no-eslintrc`                    |                                          |
| checkstyle     | `-c /sun_checks.xml`               | Uses built-in Sun checks                 |
| gocyclo        | —                                  | No config file support                   |
| golangci-lint  | `--no-config`                      |                                          |
| clippy-driver  | —                                  | **Known limitation** — clippy reads      |
|                |                                    | `.clippy.toml` if present; no official   |
|                |                                    | `--no-config` flag exists yet.           |
| rubocop        | `--force-default-config`           |                                          |
| phpcs          | `--standard=Generic`               | Overrides local phpcs.xml/ruleset        |
| dotnet/Roslyn  | —                                  | **Known limitation** — Roslyn reads      |
|                |                                    | `.editorconfig` and project settings; no |
|                |                                    | built-in flag disables them entirely.    |

## Tool choice for Java

**checkstyle** was chosen over PMD because:
- checkstyle is the de-facto standard for Java code style analysis
- Its XML output is easier to parse reliably
- It ships with Sun/Google checks that work out of the box

## Development

```bash
pip install -e ".[dev]"
pytest
```
