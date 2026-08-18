"""Dependency manifest parsers for the Detection Agent Tool.

Each parser takes raw file content (``str``) and returns a set of
lowercased package names. Parsers are pure, deterministic, and isolated:
a failure inside one parser never affects other files.

Only well-understood manifest formats are supported. The lock-file /
Swift formats that were misassigned in the legacy implementation
(``poetry.lock``, ``yarn.lock``, ``pnpm-lock.yaml``, ``Package.swift``)
are intentionally not supported here.
"""

from __future__ import annotations

import json
import re
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from collections.abc import Callable

_SIMPLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_requirements_txt(content: str) -> set[str]:
    packages: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        code = line.split("#", 1)[0].strip()
        name = re.split(r"[=<>!~;@]", code, maxsplit=1)[0].strip()
        name = re.sub(r"\[.*?\]", "", name).strip()
        if name:
            packages.add(name.lower())
    return packages


def parse_pyproject_toml(content: str) -> set[str]:
    packages: set[str] = set()
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        data = _toml_fallback(content)
    if not isinstance(data, dict):
        return packages

    project = data.get("project")
    if isinstance(project, dict):
        _collect_dependency_list(project.get("dependencies"), packages)
        optional = project.get("optional-dependencies")
        if isinstance(optional, dict):
            for dependencies in optional.values():
                _collect_dependency_list(dependencies, packages)

    tool = data.get("tool")
    if isinstance(tool, dict):
        poetry = tool.get("poetry")
        if isinstance(poetry, dict):
            poetry_deps = poetry.get("dependencies")
            if isinstance(poetry_deps, dict):
                packages.update(_package_name(pkg) for pkg in poetry_deps)
        uv = tool.get("uv")
        if isinstance(uv, dict):
            uv_deps = uv.get("dependencies")
            if isinstance(uv_deps, dict):
                packages.update(_package_name(pkg) for pkg in uv_deps)
    return {pkg for pkg in packages if pkg}


def _collect_dependency_list(value: object, packages: set[str]) -> None:
    if not isinstance(value, list):
        return
    for dependency in value:
        name = _package_name(dependency)
        if name:
            packages.add(name)


def _package_name(dependency: object) -> str | None:
    """Extract the package name from a PEP 508 / TOML dependency spec."""
    if not isinstance(dependency, str):
        return None
    name = re.split(r"[=<>!~;@\[\]]", dependency.strip(), maxsplit=1)[0].strip()
    return name.lower() if name else None


def _toml_fallback(content: str) -> dict:
    """Minimal section-based fallback for environments without tomllib.

    Only understands ``[section]`` headers and ``key = "value"`` lines,
    which is enough to recover ``[project]`` / ``[tool.poetry]``
    dependency tables used by detection.
    """
    data: dict = {}
    current: dict = data
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        header = re.match(r"^\[([A-Za-z0-9._-]+)\]$", line)
        if header:
            current = data.setdefault(header.group(1), {})
            continue
        pair = re.match(r'^([A-Za-z0-9_.-]+)\s*=\s*"([^"]*)"', line)
        if pair:
            current[pair.group(1)] = pair.group(2)
    return data


def parse_package_json(content: str) -> set[str]:
    packages: set[str] = set()
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return packages
    if not isinstance(data, dict):
        return packages
    for section in (
        "dependencies",
        "devDependencies",
        "peerDependencies",
        "optionalDependencies",
    ):
        dependencies = data.get(section)
        if isinstance(dependencies, dict):
            packages.update(name.lower() for name in dependencies)
    return packages


def parse_composer_json(content: str) -> set[str]:
    packages: set[str] = set()
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return packages
    if not isinstance(data, dict):
        return packages
    for section in ("require", "require-dev"):
        dependencies = data.get(section)
        if isinstance(dependencies, dict):
            packages.update(name.lower() for name in dependencies)
    return packages


def parse_pom_xml(content: str) -> set[str]:
    artifacts: set[str] = set()
    for dependency_block in re.findall(r"<dependency>.*?</dependency>", content, re.DOTALL):
        match = re.search(r"<artifactId>([^<]+)</artifactId>", dependency_block)
        if match:
            artifact = match.group(1).strip()
            if artifact:
                artifacts.add(artifact.lower())
    parent = re.search(r"<parent>.*?</parent>", content, re.DOTALL)
    if parent:
        match = re.search(r"<artifactId>([^<]+)</artifactId>", parent.group(0))
        if match:
            artifact = match.group(1).strip()
            if artifact:
                artifacts.add(artifact.lower())
    return artifacts


_GRADLE_CONFIGURATIONS = (
    "implementation",
    "api",
    "compile",
    "compileOnly",
    "runtimeOnly",
    "testImplementation",
    "testApi",
    "androidTestImplementation",
    "debugImplementation",
    "kapt",
    "ksp",
    "annotationProcessor",
)


def parse_gradle(content: str) -> set[str]:
    packages: set[str] = set()
    configuration_pattern = re.compile(
        r"^\s*(" + "|".join(_GRADLE_CONFIGURATIONS) + r")\s*[(']\s*['\"]?([^'\")]+)['\"]?\s*\)?\s*$"
    )
    for line in content.splitlines():
        match = configuration_pattern.match(line)
        if not match:
            continue
        coordinate = match.group(2).strip()
        parts = coordinate.split(":")
        if len(parts) >= 2:
            packages.add(f"{parts[0]}:{parts[1]}".lower())
        else:
            packages.add(coordinate.lower())
    return packages


def parse_go_mod(content: str) -> set[str]:
    packages: set[str] = set()
    in_require_block = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("require ("):
            in_require_block = True
            continue
        if in_require_block:
            if stripped == ")":
                in_require_block = False
                continue
            if stripped.startswith(("//", "replace", "exclude")):
                continue
            module_path = stripped.split()[0] if stripped else ""
            if module_path:
                packages.add(module_path.lower())
            continue
        match = re.match(r"^require\s+(\S+)", stripped)
        if match:
            packages.add(match.group(1).lower())
    return packages


def parse_cargo_toml(content: str) -> set[str]:
    packages: set[str] = set()
    dependency_sections = {"[dependencies]", "[dev-dependencies]", "[build-dependencies]"}
    in_section = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped in dependency_sections:
            in_section = True
            continue
        if in_section:
            if stripped.startswith("["):
                in_section = False
                continue
            match = re.match(r"^([A-Za-z0-9_-]+)\s*=", stripped)
            if match:
                packages.add(match.group(1).lower())
    return packages


def parse_gemfile(content: str) -> set[str]:
    packages: set[str] = set()
    for line in content.splitlines():
        match = re.match(r"^\s*gem\s+['\"]([^'\"]+)['\"]", line)
        if match:
            packages.add(match.group(1).lower())
    return packages


def parse_pubspec_yaml(content: str) -> set[str]:
    packages: set[str] = set()
    in_section = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped in ("dependencies:", "dev_dependencies:"):
            in_section = True
            continue
        if not in_section:
            continue
        if stripped and not line.startswith((" ", "\t")):
            in_section = False
            continue
        if stripped.startswith(("#", "-")) or ":" not in stripped:
            continue
        match = re.match(r"^([A-Za-z0-9_-]+)\s*:", stripped)
        if match:
            packages.add(match.group(1).lower())
    return packages


def parse_pipfile(content: str) -> set[str]:
    packages: set[str] = set()
    in_packages_section = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped in ("[packages]", "[dev-packages]"):
            in_packages_section = True
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_packages_section = False
            continue
        if not in_packages_section:
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*=\s*['\"]", stripped)
        if match:
            packages.add(match.group(1).lower())
    return packages


def parse_podfile(content: str) -> set[str]:
    packages: set[str] = set()
    for line in content.splitlines():
        match = re.match(r"^\s*pod\s+['\"]([^'\"]+)['\"]", line)
        if match:
            packages.add(match.group(1).lower())
    return packages


def parse_cartfile(content: str) -> set[str]:
    packages: set[str] = set()
    for line in content.splitlines():
        match = re.match(r'^\s*(?:github|git|binary)\s+["\']([^"\']+)["\']', line)
        if not match:
            continue
        location = match.group(1)
        name = _cartfile_package_name(location)
        if name:
            packages.add(name)
    return packages


def _cartfile_package_name(location: str) -> str | None:
    path = location.rstrip("/")
    segment = path.rsplit("/", 1)[-1].strip()
    if not segment:
        return None
    if path.endswith(".zip") and segment.endswith(".zip"):
        segment = segment[: -len(".zip")]
    return segment.lower().replace(".", "-")


def parse_csproj(content: str) -> set[str]:
    packages: set[str] = set()
    for match in re.finditer(r'<PackageReference\s+Include\s*=\s*"([^"]+)"', content):
        packages.add(match.group(1).lower())
    return packages


def parse_directory_packages_props(content: str) -> set[str]:
    packages: set[str] = set()
    for attribute in ("PackageReference", "PackageVersion"):
        for match in re.finditer(
            rf'<{attribute}\s+Include\s*=\s*"([^"]+)"', content
        ):
            packages.add(match.group(1).lower())
    return packages


def parse_packages_config(content: str) -> set[str]:
    packages: set[str] = set()
    for match in re.finditer(r'<package\s+id\s*=\s*"([^"]+)"', content):
        packages.add(match.group(1).lower())
    return packages


# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------

_FILE_NAME_PARSERS: dict[str, Callable[[str], set[str]]] = {
    "requirements.txt": parse_requirements_txt,
    "pyproject.toml": parse_pyproject_toml,
    "Pipfile": parse_pipfile,
    "package.json": parse_package_json,
    "composer.json": parse_composer_json,
    "pom.xml": parse_pom_xml,
    "build.gradle": parse_gradle,
    "build.gradle.kts": parse_gradle,
    "go.mod": parse_go_mod,
    "Cargo.toml": parse_cargo_toml,
    "Gemfile": parse_gemfile,
    "pubspec.yaml": parse_pubspec_yaml,
    "Podfile": parse_podfile,
    "Cartfile": parse_cartfile,
    "Directory.Packages.props": parse_directory_packages_props,
    "packages.config": parse_packages_config,
}

_EXTENSION_PARSERS: tuple[tuple[str, Callable[[str], set[str]]], ...] = (
    (".csproj", parse_csproj),
)


def parser_for_path(path: str) -> Callable[[str], set[str]] | None:
    """Resolve the parser for a repository file path, if any."""
    basename = path.rsplit("/", 1)[-1]
    parser = _FILE_NAME_PARSERS.get(basename)
    if parser is not None:
        return parser
    for extension, extension_parser in _EXTENSION_PARSERS:
        if basename.endswith(extension):
            return extension_parser
    return None
