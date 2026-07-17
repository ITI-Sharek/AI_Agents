from __future__ import annotations

import json
import re


# =========================================================================
# Dependency file parsers
# Each takes raw file content (str) and returns a set of package names.
# =========================================================================

def parse_requirements_txt(content: str) -> set[str]:
    packages: set[str] = set()
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        pkg = re.split(r"[=<>!~@]", line, maxsplit=1)[0].strip()
        pkg = re.sub(r"\[.*?\]", "", pkg).strip()
        pkg = re.sub(r'\s*#.*$', '', pkg).strip()
        if pkg and not pkg.startswith("-"):
            packages.add(pkg.lower())
    return packages


def parse_pyproject_toml(content: str) -> set[str]:
    packages: set[str] = set()
    try:
        import tomllib as _toml
        data = _toml.loads(content)
    except (ImportError, Exception):
        data = _parse_toml_fallback(content)
    if isinstance(data, dict):
        dependencies = data.get("project", {}).get("dependencies", []) if isinstance(data.get("project"), dict) else []
        for dep in dependencies:
            if isinstance(dep, str):
                m = re.match(r"^([A-Za-z0-9_.-]+)", dep)
                if m:
                    packages.add(m.group(1).lower())
        poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {}) if isinstance(data.get("tool"), dict) else {}
        for pkg_name in poetry_deps:
            packages.add(pkg_name.lower())
    return packages


def _parse_toml_fallback(content: str) -> dict:
    result: dict = {}
    current_section = result
    section_path: list[str] = []
    current_key: str | None = None
    current_value: str | None = None
    in_array = False
    array_items: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        section_match = re.match(r"^\[([^\]]+)\]$", stripped)
        if section_match:
            keys = section_match.group(1).strip().split(".")
            section_path = keys
            current_section = result
            for k in keys:
                k = k.strip()
                if k not in current_section:
                    current_section[k] = {}
                current_section = current_section[k]
            continue
        if stripped == "[[array]]" or stripped.startswith("[["):
            continue
        if in_array:
            m = re.match(r'^\s*"([^"]+)"', stripped)
            if m:
                array_items.append(m.group(1))
            if stripped == "]":
                if len(section_path) >= 2:
                    parent = result
                    for k in section_path[:-1]:
                        parent = parent[k]
                    parent[section_path[-1]] = array_items
                in_array = False
                array_items = []
            continue
        kv_match = re.match(r'^"([^"]+)"\s*=\s*"([^"]*)"', stripped)
        if kv_match:
            if section_path:
                current_section[kv_match.group(1)] = kv_match.group(2)
            continue
        kv_match2 = re.match(r'^(\w+)\s*=\s*"([^"]*)"', stripped)
        if kv_match2:
            if section_path:
                current_section[kv_match2.group(1)] = kv_match2.group(2)
            continue
        kv_match3 = re.match(r'^(\w+)\s*=\s*(\S+)', stripped)
        if kv_match3:
            if section_path:
                current_section[kv_match3.group(1)] = kv_match3.group(2)
            continue
        if stripped == "[":
            in_array = True
            array_items = []
            continue
    return result


def parse_package_json(content: str) -> set[str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return set()
    packages: set[str] = set()
    for section in ("dependencies", "devDependencies", "peerDependencies"):
        deps = data.get(section, {})
        if isinstance(deps, dict):
            for pkg in deps:
                packages.add(pkg.lower())
    return packages


def parse_package_lock(content: str) -> set[str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return set()
    packages: set[str] = set()
    deps = data.get("dependencies", {}) if isinstance(data, dict) else {}
    for pkg in deps:
        packages.add(pkg.lower())
    return packages


def parse_composer_json(content: str) -> set[str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return set()
    packages: set[str] = set()
    for section in ("require", "require-dev"):
        deps = data.get(section, {})
        if isinstance(deps, dict):
            for pkg in deps:
                packages.add(pkg.lower())
    return packages


def parse_pom_xml(content: str) -> set[str]:
    artifacts = re.findall(r"<artifactId>([^<]+)</artifactId>", content)
    return {a.strip().lower() for a in artifacts if a.strip()}


def parse_gradle(content: str) -> set[str]:
    packages: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        m = re.match(r"(?:implementation|api|compile|testImplementation|androidTestImplementation|kapt|annotationProcessor)\s+['\"]([^'\"]+)['\"]", stripped)
        if m:
            pkg = m.group(1)
            parts = pkg.split(":")
            if len(parts) >= 2:
                packages.add(f"{parts[0]}:{parts[1]}".lower())
            else:
                packages.add(pkg.lower())
    return packages


def parse_csproj(content: str) -> set[str]:
    packages: set[str] = set()
    for match in re.finditer(r'<PackageReference\s+Include\s*=\s*"([^"]+)"', content):
        packages.add(match.group(1).lower())
    return packages


def parse_go_mod(content: str) -> set[str]:
    packages: set[str] = set()
    in_require = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("require ("):
            in_require = True
            continue
        if in_require:
            if stripped == ")":
                in_require = False
                continue
            m = re.match(r'^(\S+)', stripped)
            if m:
                packages.add(m.group(1).lower())
            continue
        m = re.match(r'^require\s+(\S+)', stripped)
        if m:
            packages.add(m.group(1).lower())
    return packages


def parse_cargo_toml(content: str) -> set[str]:
    packages: set[str] = set()
    in_deps = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "[dependencies]":
            in_deps = True
            continue
        if in_deps:
            if stripped.startswith("["):
                in_deps = False
                continue
            m = re.match(r'^(\w[\w-]*)', stripped)
            if m:
                packages.add(m.group(1).lower())
    return packages


def parse_gemfile(content: str) -> set[str]:
    packages: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        m = re.match(r"^\s*gem\s+['\"]([^'\"]+)['\"]", stripped)
        if m:
            packages.add(m.group(1).lower())
    return packages


def parse_pubspec_yaml(content: str) -> set[str]:
    packages: set[str] = set()
    in_deps = False
    for line in content.splitlines():
        if line.strip() == "dependencies:":
            in_deps = True
            continue
        if in_deps:
            if line.strip().startswith("dev_dependencies:") or line.strip().startswith("dependency_overrides:"):
                break
            m = re.match(r'^\s+(\w[\w_-]*)\s*:', line)
            if m:
                packages.add(m.group(1).lower())
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                break
    return packages


def parse_pipfile(content: str) -> set[str]:
    packages: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        m = re.match(r'^(\w[\w.-]*)\s*=\s*"', stripped)
        if m:
            packages.add(m.group(1).lower())
    return packages


def parse_podfile(content: str) -> set[str]:
    packages: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        m = re.match(r"^\s*pod\s+['\"]([^'\"]+)['\"]", stripped)
        if m:
            packages.add(m.group(1).lower())
    return packages


def parse_cartfile(content: str) -> set[str]:
    packages: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        m = re.match(r'^(?:github|git|binary)\s+"([^"]+)"', stripped)
        if m:
            repo = m.group(1)
            parts = repo.split("/")
            if len(parts) >= 2:
                packages.add(parts[-1].lower().replace(".", "-"))
    return packages


DEPENDENCY_PARSERS: dict[str, str] = {
    "requirements.txt": "requirements",
    "Pipfile": "pipfile",
    "poetry.lock": "poetry",
    "pyproject.toml": "pyproject",
    "package.json": "package_json",
    "package-lock.json": "package_lock",
    "pnpm-lock.yaml": "package_lock",
    "yarn.lock": "package_lock",
    "composer.json": "composer",
    "composer.lock": "composer",
    "pom.xml": "pom",
    "build.gradle": "gradle",
    "build.gradle.kts": "gradle",
    "go.mod": "gomod",
    "Cargo.toml": "cargo",
    "Gemfile": "gemfile",
    "Gemfile.lock": "gemfile",
    "pubspec.yaml": "pubspec",
    "Package.swift": "swift",
    "Podfile": "podfile",
    "Cartfile": "cartfile",
    "packages.config": "csproj",
    "Directory.Packages.props": "csproj",
}
