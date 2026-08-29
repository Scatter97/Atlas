from __future__ import annotations

import json
import uuid
from pathlib import Path

from atlas.model import ModelBackend
from atlas.protocol import (
    Action,
    FinalAction,
    ToolCallAction,
    action_json_schema,
)
from atlas.registries import ApplicationRegistry, DeviceRegistry
from atlas.tools import ToolRegistry
from atlas.trace import AtlasTracer


class AtlasCore:
    def __init__(
        self,
        model: ModelBackend,
        tools: ToolRegistry,
        devices: DeviceRegistry,
        applications: ApplicationRegistry,
        prompt_path: Path,
        tracer: AtlasTracer,
        max_actions_per_request: int = 12,
    ) -> None:
        self.model = model
        self.tools = tools
        self.devices = devices
        self.applications = applications
        self.tracer = tracer
        self.max_actions_per_request = max_actions_per_request

        self.base_prompt = prompt_path.read_text(
            encoding="utf-8"
        ).strip()

    def handle(
        self,
        user_text: str,
    ) -> Action:
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self._build_system_prompt(),
            },
            {
                "role": "user",
                "content": user_text,
            },
        ]

        for _ in range(
            self.max_actions_per_request
        ):
            action = self.model.generate_action(
                messages=messages,
                response_schema=action_json_schema(),
            )

            action_data = action.model_dump(
                mode="json"
            )

            self.tracer.emit(
                "model_action",
                {
                    "request_id": request_id,
                    **action_data,
                },
            )

            if not isinstance(
                action,
                ToolCallAction,
            ):
                self.tracer.emit(
                    "request_complete",
                    {
                        "request_id": request_id,
                        "type": action.type,
                    },
                )
                return action

            call_id = f"call_{uuid.uuid4().hex[:8]}"

            self.tracer.emit(
                "tool_call",
                {
                    "request_id": request_id,
                    "call_id": call_id,
                    "name": action.name,
                    "arguments": action.arguments,
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
                    "content": action.model_dump_json(),
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
                "max_actions": self.max_actions_per_request,
            },
        )

        return FinalAction(
            type="final",
            text=(
                "I couldn't complete that reliably "
                "within the action limit."
            ),
        )

    def _build_system_prompt(
        self,
    ) -> str:
        runtime_context = {
            "available_tools": self.tools.catalog(),
            "devices": self.devices.list_public(),
            "applications": self.applications.list_public(),
        }

        context_json = json.dumps(
            runtime_context,
            indent=2,
            ensure_ascii=False,
        )

        return (
            f"{self.base_prompt}\n\n"
            "RUNTIME CONTEXT:\n"
            f"{context_json}"
        )
