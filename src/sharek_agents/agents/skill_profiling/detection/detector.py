from __future__ import annotations

import os

from sharek_agents.agents.skill_profiling.detection.parsers import (
    DEPENDENCY_PARSERS,
    parse_cargo_toml,
    parse_cartfile,
    parse_composer_json,
    parse_csproj,
    parse_gemfile,
    parse_go_mod,
    parse_gradle,
    parse_package_json,
    parse_package_lock,
    parse_pipfile,
    parse_podfile,
    parse_pom_xml,
    parse_pubspec_yaml,
    parse_pyproject_toml,
    parse_requirements_txt,
)
from sharek_agents.agents.skill_profiling.detection.registry import (
    match_by_dependency_package,
)

_PARSER_FN_MAP = {
    "requirements": parse_requirements_txt,
    "pyproject": parse_pyproject_toml,
    "pipfile": parse_pipfile,
    "poetry": parse_requirements_txt,
    "package_json": parse_package_json,
    "package_lock": parse_package_lock,
    "composer": parse_composer_json,
    "pom": parse_pom_xml,
    "gradle": parse_gradle,
    "gomod": parse_go_mod,
    "cargo": parse_cargo_toml,
    "gemfile": parse_gemfile,
    "pubspec": parse_pubspec_yaml,
    "swift": parse_requirements_txt,
    "podfile": parse_podfile,
    "cartfile": parse_cartfile,
    "csproj": parse_csproj,
}


def detect_frameworks(dependency_files: dict[str, str]) -> dict[str, list[str]]:
    all_packages: set[str] = set()

    for filename, content in dependency_files.items():
        parser_key = _resolve_dep_parser(filename)
        if parser_key is None:
            continue
        parser_fn = _PARSER_FN_MAP.get(parser_key)
        if parser_fn is None:
            continue
        try:
            packages = parser_fn(content)
            all_packages.update(packages)
        except Exception:
            continue

    return _match_frameworks(all_packages)


def _resolve_dep_parser(filename: str) -> str | None:
    basename = os.path.basename(filename)
    if basename in DEPENDENCY_PARSERS:
        return DEPENDENCY_PARSERS[basename]
    if filename.endswith(".csproj"):
        return "csproj"
    if filename == "Directory.Packages.props":
        return "csproj"
    if filename == "packages.config":
        return "csproj"
    return None


def _match_frameworks(packages: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for pkg in packages:
        entries = match_by_dependency_package(pkg)
        for entry in entries:
            _add_to_result(result, entry.name)
    return result


def _add_to_result(result: dict[str, list[str]], name: str) -> None:
    if name not in result:
        result[name] = [name]
