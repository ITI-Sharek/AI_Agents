"""Independent technology registry for the Detection Agent Tool.

Contains the static detection knowledge (display name, category,
dependency package names) and deterministic, case-insensitive matching.

Matching semantics preserved for a dependency package ``pkg`` against a
registry package ``ref`` (both compared lowercased):

* exact match: ``pkg == ref``
* namespace prefix match: ``pkg`` starts with ``ref`` followed by a
  namespace separator (``.``, ``/`` or ``:``)
* separator-substring match: ``ref`` appears inside ``pkg`` immediately
  after one of those separators (e.g. ``flask`` inside ``dj-flask-ext``
  or ``io.quarkus`` inside ``org.acme.io.quarkus``)

This module is fully independent of the legacy detection package; it is
never imported from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sharek_agents.agents.skill_profiling_agent.detection.registry_data import (
    TECHNOLOGY_ENTRIES,
)

TECHNOLOGY_CATEGORIES = (
    "framework",
    "orm",
    "template_engine",
    "testing",
    "library",
)

_SEPARATORS = (".", "/", ":")


@dataclass(frozen=True)
class TechnologyEntry:
    """One known technology with the package names that signal it."""

    name: str
    category: str
    dependency_packages: tuple[str, ...] = field(default_factory=tuple)


class TechnologyRegistry:
    """Immutable, deterministic registry of known technologies.

    The package index is built once at construction time. Match results
    are returned in entry-insertion order, so identical input always
    yields identical output.
    """

    def __init__(
        self,
        entries: tuple[TechnologyEntry, ...] | list[TechnologyEntry] | None = None,
    ) -> None:
        if entries is None:
            entries = [
                TechnologyEntry(name=name, category=category, dependency_packages=packages)
                for name, category, packages in TECHNOLOGY_ENTRIES
            ]
        self._entries: list[TechnologyEntry] = list(entries)
        self._package_index: dict[str, list[TechnologyEntry]] = {}
        for entry in self._entries:
            for package in entry.dependency_packages:
                key = package.lower()
                if key:
                    self._package_index.setdefault(key, []).append(entry)

    @property
    def size(self) -> int:
        return len(self._entries)

    def all_entries(self) -> list[TechnologyEntry]:
        return list(self._entries)

    def match(self, package_name: str) -> list[TechnologyEntry]:
        """Return deduplicated entries matching ``package_name``.

        A technology is reported at most once per lookup, keeping the
        first (insertion-ordered) occurrence on duplicates.
        """
        package = package_name.strip().lower()
        if not package:
            return []
        matched: list[TechnologyEntry] = []
        seen_names: set[str] = set()
        for reference, entries in self._package_index.items():
            if not _package_matches(package, reference):
                continue
            for entry in entries:
                if entry.name in seen_names:
                    continue
                seen_names.add(entry.name)
                matched.append(entry)
        return matched


def _package_matches(package: str, reference: str) -> bool:
    if package == reference:
        return True
    for separator in _SEPARATORS:
        if package.startswith(reference + separator):
            return True
    for separator in _SEPARATORS:
        if separator + reference in package:
            return True
    return False
