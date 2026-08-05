from __future__ import annotations

import json

from .schemas import AdvisoryFitInput


SYSTEM_PROMPT = """You are Share-k's Advisory Fit evidence analyst.
Return exactly one finding for each supplied Requirement. Cite only allowed
evidence identifiers. Missing evidence means NOT_EVIDENCED, never incapability.
Do not rank, score, recommend, accept, decline, or change an Application.
Return only the schema-conforming structured result."""


def render_advisory_fit_prompt(input_data: AdvisoryFitInput) -> str:
    return "ASSESSMENT REQUEST DATA\n" + json.dumps(
        input_data.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )
