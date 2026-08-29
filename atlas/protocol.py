from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCallAction(StrictModel):
    type: Literal["tool_call"]
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class ClarifyAction(StrictModel):
    type: Literal["clarify"]
    text: str = Field(min_length=1)


class DelegateAction(StrictModel):
    type: Literal["delegate"]
    target: Literal[
        "general_ai",
        "computer_ai",
        "research_ai",
    ]
    request: str = Field(min_length=1)


class FinalAction(StrictModel):
    type: Literal["final"]
    text: str = Field(min_length=1)


Action = Annotated[
    Union[
        ToolCallAction,
        ClarifyAction,
        DelegateAction,
        FinalAction,
    ],
    Field(discriminator="type"),
]


ACTION_ADAPTER = TypeAdapter(Action)


class ToolError(StrictModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ToolResult(StrictModel):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: Literal["success", "error"]
    result: dict[str, Any] | None = None
    error: ToolError | None = None


def action_json_schema() -> dict[str, Any]:
    return ACTION_ADAPTER.json_schema()


def parse_action(raw_json: str) -> Action:
    return ACTION_ADAPTER.validate_json(raw_json)
