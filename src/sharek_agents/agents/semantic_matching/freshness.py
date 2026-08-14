"""Semantic Matching freshness/version contract (Phase 2) + decision (Phase 4).

The matching index must remember which SOURCE VERSION a stored embedding
belongs to. This is NOT the date the matching record was updated.

Contract:

- The authoritative source data (main DB) carries a version/freshness
  marker: ``ProjectSourceData.source_version`` /
  ``ProjectSourceData.source_updated_at`` (and the Contributor equivalents).
- The matching-index record (``ProjectMatchRecord`` /
  ``ContributorMatchRecord``) stores the SAME marker at the moment the
  vector was generated: ``source_version`` / ``source_updated_at``.

The purpose:

    Main source version        vs        Version represented by local vector

    equal        ->  local vector is current
    differ       ->  local vector is stale
                     (the service fetches fresh data, regenerates the
                     embedding, and updates the index)

Phase 2 defined the concept and documented the contract only. Phase 4 adds
the actual decision logic (:func:`is_current`) used by the service's lazy
indexing (``ensure_project_indexed`` / ``ensure_contributor_indexed``). The
exact authoritative field mapping remains controlled by the
SourceDataProvider; this module only compares the markers both sides carry.

TODO/unresolved: the real field name(s) for the authoritative version live
on the backend/NestJS main database, which is not available in this
repository. The generic ``source_version`` / ``source_updated_at`` names are
placeholders to be mapped to the actual fields when the main DB connection
is wired.

Note: ``SkillProfileInput.generation_id`` and ``SkillProfileInput.requested_at``
must NOT be used as the authoritative business version: they are
request-scoped identifiers for a single skill-profile generation and do not
represent an approved/current profile.
"""

from __future__ import annotations

from datetime import datetime

#: Authoritative version marker carried by both source data and index records.
#: ``None`` means "no version known yet" (TODO: map to the real backend field).
SourceVersion = str | None

#: Authoritative update marker carried by both source data and index records.
#: ``None`` means "no update timestamp known yet" (TODO: map to the real
#: backend field).
SourceUpdatedAt = datetime | None


# ── Error hierarchy ───────────────────────────────────────────────────────────


class InvalidFreshnessMetadataError(Exception):
    """Freshness cannot be verified from the available markers.

    Raised when the authoritative source data carries NO freshness marker at
    all (``source_version`` and ``source_updated_at`` are both ``None``), so
    the local record cannot be confirmed current. Silently reusing the local
    vector could serve stale data, so the caller must surface the error
    instead.
    """


# ── Decision logic ────────────────────────────────────────────────────────────


def is_current(
    source_version: SourceVersion,
    source_updated_at: SourceUpdatedAt,
    record_version: SourceVersion,
    record_updated_at: SourceUpdatedAt,
) -> bool:
    """Decide whether a local matching record is current vs the source.

    The source is authoritative; only markers the source exposes are used:

    - ``source_version`` present -> version decides: equal = current,
      different = stale. A missing local version is treated as different.
    - only ``source_updated_at`` present -> timestamp decides: equal =
      current, different = stale. A missing local timestamp is treated as
      different.
    - neither marker present on the source side -> freshness cannot be
      verified: raises :class:`InvalidFreshnessMetadataError`.

    Args:
        source_version: Authoritative source version marker.
        source_updated_at: Authoritative source update marker.
        record_version: Version stored on the local matching record.
        record_updated_at: Update marker stored on the local matching record.

    Returns:
        ``True`` when the local record represents the same authoritative
        source state (reuse the stored vector); ``False`` when it is stale
        (regenerate).

    Raises:
        InvalidFreshnessMetadataError: The source exposes no freshness
            marker, so currency cannot be determined.
    """
    if source_version is None and source_updated_at is None:
        raise InvalidFreshnessMetadataError(
            "The authoritative source data carries no freshness marker "
            "(source_version and source_updated_at are both None), so the "
            "local matching record cannot be verified as current."
        )
    if source_version is not None:
        return source_version == record_version
    return source_updated_at == record_updated_at
