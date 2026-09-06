from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from atlas.protocol import (
    ToolError,
    ToolResult,
)
from atlas.registries import (
    ApplicationRegistry,
    DeviceRegistry,
)


class ToolArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class ToolOutput(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class ToolExecutionError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
    ) -> None:
        super().__init__(
            message
        )
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
            "input_schema": (
                self.arguments_model.model_json_schema()
            ),
            "output_schema": (
                self.output_model.model_json_schema()
            ),
            "side_effect": self.side_effect,
            "risk": self.risk,
            "confirmation": self.confirmation,
            "timeout_seconds": (
                self.timeout_seconds
            ),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[
            str,
            ToolDefinition,
        ] = {}

    def register(
        self,
        tool: ToolDefinition,
    ) -> None:
        if tool.name in self._tools:
            raise ValueError(
                "Tool already registered: "
                f"{tool.name}"
            )

        self._tools[
            tool.name
        ] = tool

    def catalog(
        self,
    ) -> list[
        dict[str, Any]
    ]:
        return [
            tool.public_schema()
            for tool
            in self._tools.values()
        ]

    def execute(
        self,
        call_id: str,
        name: str,
        arguments: dict[
            str,
            Any,
        ],
    ) -> ToolResult:
        tool = self._tools.get(
            name
        )

        if tool is None:
            return ToolResult(
                call_id=call_id,
                name=name,
                status="error",
                result=None,
                error=ToolError(
                    code="TOOL_NOT_FOUND",
                    message=(
                        f"Unknown tool: {name}"
                    ),
                ),
            )

        try:
            validated_arguments = (
                tool.arguments_model.model_validate(
                    arguments
                )
            )

            raw_result = (
                tool.executor(
                    validated_arguments
                )
            )

            validated_result = (
                tool.output_model.model_validate(
                    raw_result
                )
            )

        except ValidationError as exc:
            return ToolResult(
                call_id=call_id,
                name=name,
                status="error",
                result=None,
                error=ToolError(
                    code="INVALID_ARGUMENT",
                    message=str(
                        exc
                    ),
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
                    message=str(
                        exc
                    ),
                ),
            )

        return ToolResult(
            call_id=call_id,
            name=name,
            status="success",
            result=(
                validated_result.model_dump(
                    mode="json"
                )
            ),
            error=None,
        )


class DeviceListArguments(
    ToolArguments
):
    pass


class DeviceListOutput(
    ToolOutput
):
    devices: list[
        dict[str, Any]
    ]


class DeviceGetStatusArguments(
    ToolArguments
):
    device_id: str = Field(
        min_length=1
    )


class DeviceGetStatusOutput(
    ToolOutput
):
    device_id: str
    online: bool


class ApplicationListArguments(
    ToolArguments
):
    pass


class ApplicationListOutput(
    ToolOutput
):
    applications: list[
        dict[str, Any]
    ]


class ComputerApplicationArguments(
    ToolArguments
):
    device_id: str = Field(
        min_length=1
    )

    application_id: str = Field(
        min_length=1
    )


class ComputerLaunchApplicationOutput(
    ToolOutput
):
    launched: bool
    already_running: bool


class ComputerCloseApplicationOutput(
    ToolOutput
):
    closed: bool
    was_running: bool


class ComputerListRunningApplicationsArguments(
    ToolArguments
):
    device_id: str = Field(
        min_length=1
    )


class ComputerListRunningApplicationsOutput(
    ToolOutput
):
    device_id: str
    application_ids: list[str]


class ComputerGetVolumeArguments(
    ToolArguments
):
    device_id: str = Field(
        min_length=1
    )


class ComputerGetVolumeOutput(
    ToolOutput
):
    device_id: str
    volume_percent: int


class ComputerSetVolumeArguments(
    ToolArguments
):
    device_id: str = Field(
        min_length=1
    )

    volume_percent: int = Field(
        ge=0,
        le=100,
    )


class ComputerSetVolumeOutput(
    ToolOutput
):
    device_id: str
    volume_percent: int
    changed: bool


class TimerCreateArguments(
    ToolArguments
):
    duration_seconds: int = Field(
        gt=0,
        le=86_400,
    )

    label: str | None = Field(
        default=None,
        max_length=80,
    )


class TimerCreateOutput(
    ToolOutput
):
    timer_id: str
    duration_seconds: int
    label: str | None


class TimerListArguments(
    ToolArguments
):
    pass


class TimerListOutput(
    ToolOutput
):
    timers: list[
        dict[str, Any]
    ]


class TimerCancelArguments(
    ToolArguments
):
    timer_id: str = Field(
        min_length=1
    )


class TimerCancelOutput(
    ToolOutput
):
    timer_id: str
    cancelled: bool


def build_mock_tool_registry(
    devices: DeviceRegistry,
    applications: ApplicationRegistry,
) -> ToolRegistry:
    registry = ToolRegistry()

    running_applications: dict[
        str,
        set[str],
    ] = {}

    volume_levels: dict[
        str,
        int,
    ] = {}

    timers: dict[
        str,
        dict[str, Any],
    ] = {}

    def require_online_device(
        device_id: str,
    ) -> dict[str, Any]:
        device = devices.get(
            device_id
        )

        if device is None:
            raise ToolExecutionError(
                "DEVICE_NOT_FOUND",
                f"Unknown device: {device_id}",
            )

        if not bool(
            device.get(
                "online",
                False,
            )
        ):
            raise ToolExecutionError(
                "DEVICE_OFFLINE",
                f"Device is offline: {device_id}",
            )

        return device

    def device_list(
        _: DeviceListArguments,
    ) -> dict[str, Any]:
        return {
            "devices": (
                devices.list_public()
            ),
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
                "Unknown device: "
                f"{args.device_id}",
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

    def application_list(
        _: ApplicationListArguments,
    ) -> dict[str, Any]:
        return {
            "applications": (
                applications.list_public()
            ),
        }

    def computer_launch_application(
        args: ComputerApplicationArguments,
    ) -> dict[str, Any]:
        require_online_device(
            args.device_id
        )

        if not applications.exists(
            args.application_id
        ):
            raise ToolExecutionError(
                "APPLICATION_NOT_FOUND",
                "Unknown application: "
                f"{args.application_id}",
            )

        running = (
            running_applications.setdefault(
                args.device_id,
                set(),
            )
        )

        already_running = (
            args.application_id
            in running
        )

        running.add(
            args.application_id
        )

        return {
            "launched": True,
            "already_running": (
                already_running
            ),
        }

    def computer_close_application(
        args: ComputerApplicationArguments,
    ) -> dict[str, Any]:
        require_online_device(
            args.device_id
        )

        if not applications.exists(
            args.application_id
        ):
            raise ToolExecutionError(
                "APPLICATION_NOT_FOUND",
                "Unknown application: "
                f"{args.application_id}",
            )

        running = (
            running_applications.setdefault(
                args.device_id,
                set(),
            )
        )

        was_running = (
            args.application_id
            in running
        )

        if was_running:
            running.remove(
                args.application_id
            )

        return {
            "closed": was_running,
            "was_running": was_running,
        }

    def computer_list_running_applications(
        args: ComputerListRunningApplicationsArguments,
    ) -> dict[str, Any]:
        require_online_device(
            args.device_id
        )

        running = (
            running_applications.setdefault(
                args.device_id,
                set(),
            )
        )

        return {
            "device_id": args.device_id,
            "application_ids": sorted(
                running
            ),
        }

    def computer_get_volume(
        args: ComputerGetVolumeArguments,
    ) -> dict[str, Any]:
        require_online_device(
            args.device_id
        )

        volume = (
            volume_levels.setdefault(
                args.device_id,
                50,
            )
        )

        return {
            "device_id": args.device_id,
            "volume_percent": volume,
        }

    def computer_set_volume(
        args: ComputerSetVolumeArguments,
    ) -> dict[str, Any]:
        require_online_device(
            args.device_id
        )

        previous = (
            volume_levels.setdefault(
                args.device_id,
                50,
            )
        )

        volume_levels[
            args.device_id
        ] = args.volume_percent

        return {
            "device_id": args.device_id,
            "volume_percent": (
                args.volume_percent
            ),
            "changed": (
                previous
                != args.volume_percent
            ),
        }

    def timer_create(
        args: TimerCreateArguments,
    ) -> dict[str, Any]:
        timer_id = (
            f"timer_{uuid.uuid4().hex[:8]}"
        )

        timer = {
            "timer_id": timer_id,
            "duration_seconds": (
                args.duration_seconds
            ),
            "label": args.label,
        }

        timers[
            timer_id
        ] = timer

        return timer

    def timer_list(
        _: TimerListArguments,
    ) -> dict[str, Any]:
        return {
            "timers": [
                dict(
                    timer
                )
                for timer
                in timers.values()
            ],
        }

    def timer_cancel(
        args: TimerCancelArguments,
    ) -> dict[str, Any]:
        if args.timer_id not in timers:
            raise ToolExecutionError(
                "TIMER_NOT_FOUND",
                "Unknown timer: "
                f"{args.timer_id}",
            )

        del timers[
            args.timer_id
        ]

        return {
            "timer_id": args.timer_id,
            "cancelled": True,
        }

    registry.register(
        ToolDefinition(
            name="device_list",
            description=(
                "List the devices currently "
                "known to Atlas."
            ),
            category="device",
            arguments_model=(
                DeviceListArguments
            ),
            output_model=(
                DeviceListOutput
            ),
            executor=device_list,
        )
    )

    registry.register(
        ToolDefinition(
            name="device_get_status",
            description=(
                "Get the current online state "
                "of a known device."
            ),
            category="device",
            arguments_model=(
                DeviceGetStatusArguments
            ),
            output_model=(
                DeviceGetStatusOutput
            ),
            executor=device_get_status,
        )
    )

    registry.register(
        ToolDefinition(
            name="application_list",
            description=(
                "List applications currently "
                "known to Atlas."
            ),
            category="application",
            arguments_model=(
                ApplicationListArguments
            ),
            output_model=(
                ApplicationListOutput
            ),
            executor=application_list,
        )
    )

    registry.register(
        ToolDefinition(
            name="computer_launch_application",
            description=(
                "Launch a known application "
                "on a known online computer."
            ),
            category="computer",
            arguments_model=(
                ComputerApplicationArguments
            ),
            output_model=(
                ComputerLaunchApplicationOutput
            ),
            executor=(
                computer_launch_application
            ),
            side_effect=True,
            risk="low",
            confirmation="never",
        )
    )

    registry.register(
        ToolDefinition(
            name="computer_close_application",
            description=(
                "Close a known application "
                "on a known online computer."
            ),
            category="computer",
            arguments_model=(
                ComputerApplicationArguments
            ),
            output_model=(
                ComputerCloseApplicationOutput
            ),
            executor=(
                computer_close_application
            ),
            side_effect=True,
            risk="low",
            confirmation="never",
        )
    )

    registry.register(
        ToolDefinition(
            name=(
                "computer_list_running_applications"
            ),
            description=(
                "List Atlas-known applications "
                "currently running on a computer."
            ),
            category="computer",
            arguments_model=(
                ComputerListRunningApplicationsArguments
            ),
            output_model=(
                ComputerListRunningApplicationsOutput
            ),
            executor=(
                computer_list_running_applications
            ),
        )
    )

    registry.register(
        ToolDefinition(
            name="computer_get_volume",
            description=(
                "Get the current mock system "
                "volume of an online computer."
            ),
            category="computer",
            arguments_model=(
                ComputerGetVolumeArguments
            ),
            output_model=(
                ComputerGetVolumeOutput
            ),
            executor=computer_get_volume,
        )
    )

    registry.register(
        ToolDefinition(
            name="computer_set_volume",
            description=(
                "Set the mock system volume "
                "of an online computer from "
                "0 to 100 percent."
            ),
            category="computer",
            arguments_model=(
                ComputerSetVolumeArguments
            ),
            output_model=(
                ComputerSetVolumeOutput
            ),
            executor=computer_set_volume,
            side_effect=True,
            risk="low",
            confirmation="never",
        )
    )

    registry.register(
        ToolDefinition(
            name="timer_create",
            description=(
                "Create a timer for the "
                "requested duration."
            ),
            category="timer",
            arguments_model=(
                TimerCreateArguments
            ),
            output_model=(
                TimerCreateOutput
            ),
            executor=timer_create,
            side_effect=True,
            risk="low",
            confirmation="never",
        )
    )

    registry.register(
        ToolDefinition(
            name="timer_list",
            description=(
                "List timers currently known "
                "to Atlas."
            ),
            category="timer",
            arguments_model=(
                TimerListArguments
            ),
            output_model=(
                TimerListOutput
            ),
            executor=timer_list,
        )
    )

    registry.register(
        ToolDefinition(
            name="timer_cancel",
            description=(
                "Cancel a known timer by "
                "timer ID."
            ),
            category="timer",
            arguments_model=(
                TimerCancelArguments
            ),
            output_model=(
                TimerCancelOutput
            ),
            executor=timer_cancel,
            side_effect=True,
            risk="low",
            confirmation="never",
        )
    )

    return registry
