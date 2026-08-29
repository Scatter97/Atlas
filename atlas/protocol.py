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


# ----------------------------------------------------------------------
# New controller / tool specialist protocol extensions
# ----------------------------------------------------------------------


class RouteAction(StrictModel):
    type: Literal["route"]
    target: Literal[
        "atlas_tools",
        "general_ai",
        "computer_ai",
        "research_ai",
    ]
    request: str = Field(min_length=1)


ControllerAction = Annotated[
    Union[
        RouteAction,
        ClarifyAction,
        FinalAction,
    ],
    Field(discriminator="type"),
]


ToolSpecialistAction = Annotated[
    Union[
        ToolCallAction,
        ClarifyAction,
        FinalAction,
    ],
    Field(discriminator="type"),
]


CONTROLLER_ACTION_ADAPTER = TypeAdapter(
    ControllerAction
)

TOOL_SPECIALIST_ACTION_ADAPTER = TypeAdapter(
    ToolSpecialistAction
)


def controller_action_json_schema() -> dict[str, Any]:
    return CONTROLLER_ACTION_ADAPTER.json_schema()


def parse_controller_action(
    raw_json: str,
) -> ControllerAction:
    return CONTROLLER_ACTION_ADAPTER.validate_json(
        raw_json
    )


def tool_specialist_action_json_schema() -> dict[str, Any]:
    return TOOL_SPECIALIST_ACTION_ADAPTER.json_schema()


def parse_tool_specialist_action(
    raw_json: str,
) -> ToolSpecialistAction:
    return TOOL_SPECIALIST_ACTION_ADAPTER.validate_json(
        raw_json
    )
