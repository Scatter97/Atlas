from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atlas.protocol import ToolError, ToolResult
from atlas.registries import ApplicationRegistry, DeviceRegistry


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolExecutionError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


ToolExecutor = Callable[
    [BaseModel],
    dict[str, Any],
]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    category: str
    arguments_model: type[BaseModel]
    output_model: type[BaseModel]
    executor: ToolExecutor
    side_effect: bool = False
    risk: str = "none"
    confirmation: str = "never"
    timeout_seconds: int = 15

    def public_schema(
        self,
    ) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "input_schema": self.arguments_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
            "side_effect": self.side_effect,
            "risk": self.risk,
            "confirmation": self.confirmation,
            "timeout_seconds": self.timeout_seconds,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        tool: ToolDefinition,
    ) -> None:
        if tool.name in self._tools:
            raise ValueError(
                f"Tool already registered: {tool.name}"
            )

        self._tools[tool.name] = tool

    def catalog(
        self,
    ) -> list[dict[str, Any]]:
        return [
            tool.public_schema()
            for tool in self._tools.values()
        ]

    def execute(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        tool = self._tools.get(name)

        if tool is None:
            return ToolResult(
                call_id=call_id,
                name=name,
                status="error",
                result=None,
                error=ToolError(
                    code="TOOL_NOT_FOUND",
                    message=f"Unknown tool: {name}",
                ),
            )

        try:
            validated_arguments = tool.arguments_model.model_validate(
                arguments
            )

            raw_result = tool.executor(
                validated_arguments
            )

            validated_result = tool.output_model.model_validate(
                raw_result
            )

        except ValidationError as exc:
            return ToolResult(
                call_id=call_id,
                name=name,
                status="error",
                result=None,
                error=ToolError(
                    code="INVALID_ARGUMENT",
                    message=str(exc),
                ),
            )

        except ToolExecutionError as exc:
            return ToolResult(
                call_id=call_id,
                name=name,
                status="error",
                result=None,
                error=ToolError(
                    code=exc.code,
                    message=exc.message,
                ),
            )

        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                name=name,
                status="error",
                result=None,
                error=ToolError(
                    code="TOOL_EXECUTION_FAILED",
                    message=str(exc),
                ),
            )

        return ToolResult(
            call_id=call_id,
            name=name,
            status="success",
            result=validated_result.model_dump(
                mode="json"
            ),
            error=None,
        )


class DeviceListArguments(ToolArguments):
    pass


class DeviceListOutput(ToolOutput):
    devices: list[dict[str, Any]]


class DeviceGetStatusArguments(ToolArguments):
    device_id: str = Field(min_length=1)


class DeviceGetStatusOutput(ToolOutput):
    device_id: str
    online: bool


class ComputerLaunchApplicationArguments(ToolArguments):
    device_id: str = Field(min_length=1)
    application_id: str = Field(min_length=1)


class ComputerLaunchApplicationOutput(ToolOutput):
    launched: bool
    already_running: bool


class TimerCreateArguments(ToolArguments):
    duration_seconds: int = Field(
        gt=0,
        le=86_400,
    )
    label: str | None = Field(
        default=None,
        max_length=80,
    )


class TimerCreateOutput(ToolOutput):
    timer_id: str
    duration_seconds: int
    label: str | None


def build_mock_tool_registry(
    devices: DeviceRegistry,
    applications: ApplicationRegistry,
) -> ToolRegistry:
    registry = ToolRegistry()

    def device_list(
        _: DeviceListArguments,
    ) -> dict[str, Any]:
        return {
            "devices": devices.list_public(),
        }

    def device_get_status(
        args: DeviceGetStatusArguments,
    ) -> dict[str, Any]:
        device = devices.get(
            args.device_id
        )

        if device is None:
            raise ToolExecutionError(
                "DEVICE_NOT_FOUND",
                f"Unknown device: {args.device_id}",
            )

        return {
            "device_id": args.device_id,
            "online": bool(
                device.get(
                    "online",
                    False,
                )
            ),
        }

    def computer_launch_application(
        args: ComputerLaunchApplicationArguments,
    ) -> dict[str, Any]:
        device = devices.get(
            args.device_id
        )

        if device is None:
            raise ToolExecutionError(
                "DEVICE_NOT_FOUND",
                f"Unknown device: {args.device_id}",
            )

        if not bool(
            device.get(
                "online",
                False,
            )
        ):
            raise ToolExecutionError(
                "DEVICE_OFFLINE",
                f"Device is offline: {args.device_id}",
            )

        if not applications.exists(
            args.application_id
        ):
            raise ToolExecutionError(
                "APPLICATION_NOT_FOUND",
                f"Unknown application: {args.application_id}",
            )

        return {
            "launched": True,
            "already_running": False,
        }

    def timer_create(
        args: TimerCreateArguments,
    ) -> dict[str, Any]:
        timer_id = (
            f"timer_{uuid.uuid4().hex[:8]}"
        )

        return {
            "timer_id": timer_id,
            "duration_seconds": args.duration_seconds,
            "label": args.label,
        }

    registry.register(
        ToolDefinition(
            name="device_list",
            description=(
                "List the devices currently known to Atlas."
            ),
            category="device",
            arguments_model=DeviceListArguments,
            output_model=DeviceListOutput,
            executor=device_list,
        )
    )

    registry.register(
        ToolDefinition(
            name="device_get_status",
            description=(
                "Get the current online state of a known device."
            ),
            category="device",
            arguments_model=DeviceGetStatusArguments,
            output_model=DeviceGetStatusOutput,
            executor=device_get_status,
        )
    )

    registry.register(
        ToolDefinition(
            name="computer_launch_application",
            description=(
                "Mock-launch a known application on a known online computer."
            ),
            category="computer",
            arguments_model=ComputerLaunchApplicationArguments,
            output_model=ComputerLaunchApplicationOutput,
            executor=computer_launch_application,
            side_effect=True,
            risk="low",
            confirmation="never",
        )
    )

    registry.register(
        ToolDefinition(
            name="timer_create",
            description=(
                "Create a mock timer for the requested duration."
            ),
            category="timer",
            arguments_model=TimerCreateArguments,
            output_model=TimerCreateOutput,
            executor=timer_create,
            side_effect=True,
            risk="low",
            confirmation="never",
        )
    )

    return registry
