from __future__ import annotations

from pathlib import Path

from atlas.config import Config
from atlas.core import AtlasCore
from atlas.model import ModelBackendError, OllamaBackend
from atlas.protocol import (
    ClarifyAction,
    DelegateAction,
    FinalAction,
)
from atlas.registries import ApplicationRegistry, DeviceRegistry
from atlas.tools import build_mock_tool_registry
from atlas.trace import AtlasTracer


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


def build_core(
    config: Config,
) -> AtlasCore:
    devices = DeviceRegistry.from_file(
        PROJECT_ROOT
        / "data"
        / "devices.json"
    )

    applications = ApplicationRegistry.from_file(
        PROJECT_ROOT
        / "data"
        / "applications.json"
    )

    tools = build_mock_tool_registry(
        devices=devices,
        applications=applications,
    )

    model = OllamaBackend(
        base_url=config.ollama_base_url,
        model=config.ollama_model,
        timeout_seconds=config.model_timeout_seconds,
    )

    tracer = AtlasTracer(
        terminal_enabled=config.trace_enabled,
    )

    return AtlasCore(
        model=model,
        tools=tools,
        devices=devices,
        applications=applications,
        prompt_path=(
            PROJECT_ROOT
            / "prompts"
            / "tool_controller.txt"
        ),
        tracer=tracer,
        max_actions_per_request=config.max_actions_per_request,
    )


def main() -> None:
    config = Config.from_env()
    core = build_core(config)

    print("Atlas Core v0.1.1")
    print(f"Model: {config.ollama_model}")
    print("Mock tools enabled.")
    print(
        "Developer trace: "
        + (
            "enabled"
            if config.trace_enabled
            else "disabled"
        )
    )
    print("Type 'exit' to quit.")

    while True:
        try:
            user_text = input(
                "\nYou: "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print("\nGoodbye.")
            return

        if not user_text:
            continue

        if user_text.lower() in {
            "exit",
            "quit",
        }:
            print("Goodbye.")
            return

        try:
            action = core.handle(
                user_text
            )

        except ModelBackendError as exc:
            print(
                f"Core error: {exc}"
            )
            continue

        if isinstance(
            action,
            FinalAction,
        ):
            print(
                f"Atlas: {action.text}"
            )

        elif isinstance(
            action,
            ClarifyAction,
        ):
            print(
                f"Atlas: {action.text}"
            )

        elif isinstance(
            action,
            DelegateAction,
        ):
            print(
                "Atlas delegation: "
                f"{action.target} -> {action.request}"
            )

        else:
            print(
                f"Unexpected action: {action}"
            )


if __name__ == "__main__":
    main()
