from __future__ import annotations

from sharek_agents.agents.contributor_matching.schemas import ContributorMatchingInput


SYSTEM_PROMPT = """You are Share-k's Contributor Matching Agent.

Return only one JSON object matching the supplied response schema. Produce
ranked recommendations, never an eligibility verdict or an owner decision.
Use only contributor IDs, approved skills, reputation facts, requirements, and
evidence capsules supplied in the request. Never invent a contributor, skill,
reputation fact, citation, or source URL. Every returned match and matched
skill must be supported by the supplied evidence IDs. A lower score means a
weaker recommendation, not that the contributor is incapable or ineligible.
"""


def render_contributor_matching_prompt(input_data: ContributorMatchingInput) -> str:
    return (
        "MATCHING REQUEST DATA\n"
        f"{input_data.model_dump_json(by_alias=True, indent=2)}\n\n"
        "Return the strongest matching contributors only. Explain the ranking "
        "in concise owner-facing language and cite the exact evidence IDs."
    )
