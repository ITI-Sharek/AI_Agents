from __future__ import annotations

import json

from sharek_agents.agents.skill_gap_guidance.retrieval import (
    retrieve_curated_resources,
)
from sharek_agents.agents.skill_gap_guidance.schemas import SkillGapGuidanceInput

SYSTEM_PROMPT = """You are Share-k's explicit skill-gap guidance assistant.

Return exactly one JSON object matching the supplied response schema.
Guidance is educational and evidence-scoped. It is not an eligibility verdict,
Application decision, score, rank, rejection reason, or selection recommendation.

Rules:
- Treat the approved skill snapshot as the only recorded contributor capability.
- Say a skill is not evidenced or below the target proficiency; never claim the
  contributor is incapable or ineligible.
- Use only the supplied evidence IDs for citations and sources.
- Do not invent learning resources, URLs, source labels, or evidence.
- Include an improvement duration only when the supplied evidence supports it;
  otherwise leave it null and avoid precise promises.
- Do not emit fields for tier, subscription, rejection, eligibility, status,
  score, rank, acceptance, decline, or owner decision.
- Include at least one source attribution for every completed response.
- Keep recommendations actionable, concise, and safe for contributor display.
"""


def render_skill_gap_guidance_prompt(input_data: SkillGapGuidanceInput) -> str:
    payload = input_data.model_dump(mode="json", by_alias=True)
    retrieved_resources = [
        {
            "resourceId": resource.resource_id,
            "title": resource.title,
            "resourceType": resource.resource_type,
            "url": resource.url,
            "summary": resource.summary,
        }
        for resource in retrieve_curated_resources(input_data)
    ]
    return (
        "Produce educational guidance from this fixed, authorized snapshot.\n\n"
        "GUIDANCE REQUEST DATA\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n\n"
        "RETRIEVED CURATED LEARNING RESOURCES\n"
        f"{json.dumps(retrieved_resources, ensure_ascii=False, sort_keys=True)}\n\n"
        "Use only the exact retrieved resource URLs above. If no retrieved resource "
        "is relevant, return an empty learningResources list."
    )
