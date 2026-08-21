from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from sharek_agents.agents.skill_gap_guidance.schemas import (
    SkillGapGuidanceInput,
    SkillGapGuidanceResult,
)
from sharek_agents.agents.skill_gap_guidance.service import (
    SkillGapGuidanceProviderError,
    SkillGapGuidanceProviderTimeout,
    generate_skill_gap_guidance,
)
from sharek_agents.common.logging import get_logger

logger = get_logger(__name__)


async def analyze_skill_gap_guidance(
    body: SkillGapGuidanceInput,
) -> SkillGapGuidanceResult:
    try:
        return await generate_skill_gap_guidance(body)
    except SkillGapGuidanceProviderTimeout as exc:
        logger.warning("Skill-gap guidance provider timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Skill-gap guidance provider timed out",
        ) from exc
    except SkillGapGuidanceProviderError as exc:
        logger.warning("Skill-gap guidance provider returned invalid output")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Skill-gap guidance provider returned an invalid response",
        ) from exc


async def stream_skill_gap_guidance(
    body: SkillGapGuidanceInput,
) -> StreamingResponse:
    result = await analyze_skill_gap_guidance(body)

    async def events() -> AsyncIterator[str]:
        payload = json.dumps(result.model_dump(mode="json", by_alias=True))
        yield f"event: guidance.completed\ndata: {payload}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
