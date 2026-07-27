# Skill Profiling Agent

Profiles GitHub repositories by analyzing code quality, frameworks, architecture, and metadata — producing a structured skill profile per repo.

## Backend Evidence-Capsule Contract

The NestJS worker uses `POST /skill-profiles/generate`. This endpoint accepts
backend-selected repository evidence capsules, analyzes only that supplied
evidence, and returns camel-cased skill candidates with audit/version metadata.
It does not crawl the submitted repository URLs.

Generation requires a shared bearer token of at least 32 characters:

```bash
export AI_SERVICE_AUTH_TOKEN='replace-with-the-same-long-random-token-as-NestJS'
uvicorn sharek_agents.main:app --reload --host 0.0.0.0 --port 8000
```

`GET /health` remains unauthenticated. Binding to `0.0.0.0` makes the service
reachable from the local NestJS Docker container through
`http://host.docker.internal:8000`; protect the listener with the local
machine/firewall.

Optional audit/runtime settings are `AI_PROVIDER`, `DEFAULT_MODEL`,
`AI_SKILL_PROFILE_TIMEOUT_SECONDS`, and `AI_SERVICE_VERSION`.

---

## Architecture Overview

```
repo URLs
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  router.py  ───  profile_repos(urls)                 │
│                                                      │
│  Calls tools.py to gather evidence, then sends       │
│  each repo's data through the LLM graph,             │
│  post-processes results into RepoProfile objects.    │
└──────┬───────────────────────────────────────────────┘
       │ evidence dict
       ▼
┌──────────────────────────────────────────────────────┐
│  tools.py  ───  gather_all_evidence(urls)            │
│                                                      │
│  For each URL: parse → fetch metadata → get file     │
│  tree → detect frameworks → clone → static analysis  │
│  → graphify → return structured dict.                │
└──────┬───────────────────────────────────────────────┘
       │ evidence dict
       ▼
┌──────────────────────────────────────────────────────┐
│  graph.py  ───  run(evidence)                        │
│                                                      │
│  Sends evidence to LLM via ChatOpenRouter,            │
│  parses structured output (SkillProfilingResult).    │
│  Retries once on schema validation failure.          │
└──────┬───────────────────────────────────────────────┘
       │ AgentResponse(status, data: SkillProfilingResult)
       ▼
┌──────────────────────────────────────────────────────┐
│  router.py  (post-processing)                         │
│                                                      │
│  _build_skills → _compute_confidence → RepoProfile   │
└──────┬───────────────────────────────────────────────┘
       │ list[RepoProfile]
       ▼
```

---

## Input

A list of GitHub repository URLs.

```
profile_repos([
    "https://github.com/owner/repo-a",
    "https://github.com/owner/repo-b",
])
```

Supported URL formats:
- `https://github.com/owner/repo`
- `https://github.com/owner/repo.git`
- `git@github.com:owner/repo.git`

---

## Output

```python
class RepoProfile(BaseModel):
    repo: RepoInfo          # name, owner, language, topics, stars, forks, etc.
    status: "success" | "needs_review"
    confidence: float        # 0.0 – 1.0
    skills: list[Skill]      # identified skills with per-skill confidence + sources

class RepoInfo(BaseModel):
    name: str
    owner: str
    description: str
    language: str            # primary language
    topics: list[str]
    stars: int
    forks: int
    clone_url: str
    default_branch: str

class Skill(BaseModel):
    name: str
    confidence: float
    sources: list[Source]    # evidence backing this skill

class Source(BaseModel):
    type: "repo_metadata" | "framework_detection" | "static_analysis" | "graphify_relations"
    detail: str              # human-readable with concrete numbers
```

---

## Evidence Flow (per repo)

### 1. Repo Metadata
Fetched via GitHub API `GET /repos/{owner}/{repo}`:
- name, owner, description, language, topics, stars, forks, default_branch, clone_url

### 2. Framework Detection
Scans dependency files from GitHub API:
- `requirements.txt`, `pyproject.toml` → Python
- `package.json` → JavaScript
- `pom.xml` → Java
- `*.csproj` → C#

Tokenizes file contents and matches against known framework keywords.

| Language     | Frameworks Detected                                                 |
|-------------|---------------------------------------------------------------------|
| Python      | fastapi, django, flask                                               |
| JavaScript  | react, vue, @angular/core, express, @nestjs/core, next               |
| Java        | spring, spring-boot, jakarta, javax                                  |
| C#          | aspnetcore, entity-framework, dapper, serilog                        |

### 3. Static Analysis
Clones repo with `git clone --depth=1`, runs language-specific tooling:

| Language       | Tools                          | Metrics                                           |
|---------------|--------------------------------|---------------------------------------------------|
| Python        | `radon cc`, `radon mi`, `pylint` | avg_complexity, maintainability_index, pylint_score |
| JavaScript/TS | `eslint`                         | error_count, warning_count                         |

Returns early with `skipped: true` if no source files or language is unsupported.

### 4. Graphify Relations
Clones repo and runs `graphify update` to extract inheritance and call graphs.
Prunes edges to only the repo's source files, caps output at ~2000 tokens.

---

## LLM Integration

Uses `ChatOpenRouter` from `langchain-openrouter` (OpenAI-compatible endpoint).

| Env Variable         | Default                | Description              |
|---------------------|------------------------|--------------------------|
| `OPENROUTER_API_KEY` | —                      | OpenRouter API key       |
| `OPENROUTER_MODEL`   | `openai/gpt-4o-mini` | Model identifier on OpenRouter |

The LLM receives a JSON evidence dict and returns `SkillProfilingResult` (structured output via Pydantic).

---

## Confidence Scoring

Per-skill confidence depends on evidence type:
- `repo_metadata`: `min(0.5 + 0.05 × N, 0.6)`
- `framework_detection`: `min(0.55 + 0.05 × N, 0.7)`
- `static_analysis`: `min(0.65 + 0.05 × N, 1.0)`
- `graphify_relations`: `min(0.6 + 0.05 × N, 1.0)`

Where `N` = number of repos profiled.

Overall confidence = average of per-skill scores, discounted if `N < 2`.

---

## File Map

```
src/sharek_agents/agents/skill_profiling/
├── __init__.py       # empty
├── schemas.py        # Pydantic models (input, output, intermediate)
├── tools.py          # evidence gathering (GitHub API, subprocess tools)
├── router.py         # orchestration + post-processing
├── graph.py          # LLM invocation via ChatOpenRouter
├── prompts.py        # system prompt for the LLM
└── README.md         # this file

src/sharek_agents/shared_tools/
└── github_client.py  # HTTP client wrapping GitHub REST API

src/sharek_agents/
└── config.py         # settings from environment variables
```

---

## Dependencies

| Package               | Purpose                        |
|-----------------------|--------------------------------|
| `httpx`               | GitHub API HTTP client          |
| `langchain`           | LLM invocation framework        |
| `langchain-openrouter`| OpenRouter LLM integration      |
| `pydantic`            | Schema validation               |
| `python-dotenv`       | Env file loading                |
| `radon`               | Python static analysis          |
| `pylint`              | Python linting                  |
| `eslint`              | JavaScript/TypeScript linting   |
| `graphifyy`           | Code graph extraction           |
| `git`                 | Repo cloning                    |
