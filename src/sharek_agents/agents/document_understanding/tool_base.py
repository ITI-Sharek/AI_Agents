from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from sharek_agents.agents.document_understanding.tools import (
    ToolDefinition,
    ToolInput,
)


class BaseTool:
    """Concrete base for typed tools that satisfy the ``Tool`` protocol.

    Subclasses must set ``name``, ``description``, and ``input_schema``
    as class attributes, and implement :meth:`run`.

    Example::

        class SearchInput(ToolInput):
            query: str
            limit: int = 5

        class SearchTool(BaseTool):
            name = "search_project_documents"
            description = "Search uploaded project documents"
            input_schema = SearchInput

            async def run(self, args: SearchInput) -> str:
                ...
    """

    name: str = ""
    description: str = ""
    input_schema: type[BaseModel] = ToolInput
    output_schema: type[BaseModel] | None = None

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_schema.model_json_schema(),
            output_schema=self.output_schema.model_json_schema()
            if self.output_schema is not None
            else None,
        )

    async def execute(self, **kwargs: Any) -> str:
        validated = self.input_schema(**kwargs)
        return await self.run(validated)

    async def run(self, args: BaseModel) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement run()"
        )
