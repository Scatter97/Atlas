from __future__ import annotations

import json
import uuid
from pathlib import Path

from atlas.protocol import (
    Action,
    ClarifyAction,
    DelegateAction,
    FinalAction,
    RouteAction,
    ToolCallAction,
    controller_action_json_schema,
    parse_controller_action,
    parse_tool_specialist_action,
    tool_specialist_action_json_schema,
)
from atlas.registries import (
    ApplicationRegistry,
    DeviceRegistry,
)
from atlas.runtime import (
    ModelRuntimeManager,
)
from atlas.tools import ToolRegistry
from atlas.trace import AtlasTracer


class AtlasCore:
    def __init__(
        self,
        runtime: ModelRuntimeManager,
        tools: ToolRegistry,
        devices: DeviceRegistry,
        applications: ApplicationRegistry,
        controller_prompt_path: Path,
        tool_prompt_path: Path,
        general_prompt_path: Path,
        tracer: AtlasTracer,
        max_actions_per_request: int = 12,
    ) -> None:
        self.runtime = runtime
        self.tools = tools
        self.devices = devices
        self.applications = applications
        self.tracer = tracer
        self.max_actions_per_request = (
            max_actions_per_request
        )

        self.controller_prompt = (
            controller_prompt_path.read_text(
                encoding="utf-8"
            ).strip()
        )

        self.tool_prompt = (
            tool_prompt_path.read_text(
                encoding="utf-8"
            ).strip()
        )

        self.general_prompt = (
            general_prompt_path.read_text(
                encoding="utf-8"
            ).strip()
        )

    def preload_models(
        self,
    ) -> None:
        self.runtime.preload_persistent()

    def handle(
        self,
        user_text: str,
    ) -> Action:
        request_id = (
            f"req_{uuid.uuid4().hex[:8]}"
        )

        controller_messages = [
            {
                "role": "system",
                "content": self.controller_prompt,
            },
            {
                "role": "user",
                "content": user_text,
            },
        ]

        controller_raw = (
            self.runtime.generate_structured(
                role="controller",
                messages=controller_messages,
                response_schema=(
                    controller_action_json_schema()
                ),
                request_id=request_id,
            )
        )

        controller_action = (
            parse_controller_action(
                controller_raw
            )
        )

        if isinstance(
            controller_action,
            RouteAction,
        ):
            self.tracer.emit(
                "controller_action",
                {
                    "request_id": request_id,
                    "type": "route",
                    "target": (
                        controller_action.target
                    ),
                },
            )

            return self._handle_route(
                route=controller_action,
                request_id=request_id,
            )

        self.tracer.emit(
            "controller_action",
            {
                "request_id": request_id,
                "type": controller_action.type,
            },
        )

        self._trace_complete(
            request_id=request_id,
            action=controller_action,
        )

        return controller_action

    def _handle_route(
        self,
        route: RouteAction,
        request_id: str,
    ) -> Action:
        if route.target == "atlas_tools":
            return self._handle_tools(
                request_text=route.request,
                request_id=request_id,
            )

        if route.target == "general_ai":
            return self._handle_general(
                request_text=route.request,
                request_id=request_id,
            )

        if route.target == "computer_ai":
            action = DelegateAction(
                type="delegate",
                target="computer_ai",
                request=route.request,
            )

            self._trace_complete(
                request_id=request_id,
                action=action,
            )

            return action

        if route.target == "research_ai":
            action = DelegateAction(
                type="delegate",
                target="research_ai",
                request=route.request,
            )

            self._trace_complete(
                request_id=request_id,
                action=action,
            )

            return action

        action = FinalAction(
            type="final",
            text=(
                "I couldn't route that request "
                "reliably."
            ),
        )

        self._trace_complete(
            request_id=request_id,
            action=action,
        )

        return action

    def _handle_general(
        self,
        request_text: str,
        request_id: str,
    ) -> FinalAction:
        messages = [
            {
                "role": "system",
                "content": self.general_prompt,
            },
            {
                "role": "user",
                "content": request_text,
            },
        ]

        response_text = (
            self.runtime.generate_text(
                role="general",
                messages=messages,
                request_id=request_id,
            )
        ).strip()

        if not response_text:
            response_text = (
                "I couldn't produce a response."
            )

        action = FinalAction(
            type="final",
            text=response_text,
        )

        self._trace_complete(
            request_id=request_id,
            action=action,
        )

        return action

    def _handle_tools(
        self,
        request_text: str,
        request_id: str,
    ) -> Action:
        messages: list[
            dict[str, str]
        ] = [
            {
                "role": "system",
                "content": (
                    self._build_tool_prompt()
                ),
            },
            {
                "role": "user",
                "content": request_text,
            },
        ]

        for _ in range(
            self.max_actions_per_request
        ):
            raw_action = (
                self.runtime.generate_structured(
                    role="tools",
                    messages=messages,
                    response_schema=(
                        tool_specialist_action_json_schema()
                    ),
                    request_id=request_id,
                )
            )

            action = (
                parse_tool_specialist_action(
                    raw_action
                )
            )

            action_data = action.model_dump(
                mode="json"
            )

            self.tracer.emit(
                "model_action",
                {
                    "request_id": request_id,
                    "role": "tools",
                    **action_data,
                },
            )

            if not isinstance(
                action,
                ToolCallAction,
            ):
                self._trace_complete(
                    request_id=request_id,
                    action=action,
                )

                return action

            call_id = (
                f"call_{uuid.uuid4().hex[:8]}"
            )

            self.tracer.emit(
                "tool_call",
                {
                    "request_id": request_id,
                    "call_id": call_id,
                    "name": action.name,
                    "arguments": (
                        action.arguments
                    ),
                },
            )

            result = self.tools.execute(
                call_id=call_id,
                name=action.name,
                arguments=action.arguments,
            )

            self.tracer.emit(
                "tool_result",
                {
                    "request_id": request_id,
                    **result.model_dump(
                        mode="json"
                    ),
                },
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        action.model_dump_json()
                    ),
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "TOOL_RESULT\n"
                        + result.model_dump_json()
                    ),
                }
            )

        self.tracer.emit(
            "action_limit_reached",
            {
                "request_id": request_id,
                "max_actions": (
                    self.max_actions_per_request
                ),
            },
        )

        action = FinalAction(
            type="final",
            text=(
                "I couldn't complete that reliably "
                "within the action limit."
            ),
        )

        self._trace_complete(
            request_id=request_id,
            action=action,
        )

        return action

    def _build_tool_prompt(
        self,
    ) -> str:
        runtime_context = {
            "available_tools": (
                self.tools.catalog()
            ),
            "devices": (
                self.devices.list_public()
            ),
            "applications": (
                self.applications.list_public()
            ),
        }

        context_json = json.dumps(
            runtime_context,
            indent=2,
            ensure_ascii=False,
        )

        return (
            f"{self.tool_prompt}\n\n"
            "RUNTIME CONTEXT:\n"
            f"{context_json}"
        )

    def _trace_complete(
        self,
        request_id: str,
        action: Action,
    ) -> None:
        self.tracer.emit(
            "request_complete",
            {
                "request_id": request_id,
                "type": action.type,
            },
        )
