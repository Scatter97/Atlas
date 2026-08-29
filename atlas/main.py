from __future__ import annotations

from pathlib import Path

from atlas.config import Config
from atlas.core import AtlasCore
from atlas.model import ModelBackendError
from atlas.protocol import (
    ClarifyAction,
    DelegateAction,
    FinalAction,
)
from atlas.registries import (
    ApplicationRegistry,
    DeviceRegistry,
)
from atlas.runtime import (
    ModelRuntimeManager,
)
from atlas.tools import (
    build_mock_tool_registry,
)
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

    applications = (
        ApplicationRegistry.from_file(
            PROJECT_ROOT
            / "data"
            / "applications.json"
        )
    )

    tools = build_mock_tool_registry(
        devices=devices,
        applications=applications,
    )

    tracer = AtlasTracer(
        terminal_enabled=(
            config.trace_enabled
        ),
    )

    runtime = ModelRuntimeManager.from_file(
        path=(
            PROJECT_ROOT
            / "data"
            / "model_profiles.json"
        ),
        ollama_base_url=(
            config.ollama_base_url
        ),
        timeout_seconds=(
            config.model_timeout_seconds
        ),
        tracer=tracer,
        selected_profile=(
            config.model_profile
        ),
    )

    return AtlasCore(
        runtime=runtime,
        tools=tools,
        devices=devices,
        applications=applications,
        controller_prompt_path=(
            PROJECT_ROOT
            / "prompts"
            / "controller.txt"
        ),
        tool_prompt_path=(
            PROJECT_ROOT
            / "prompts"
            / "tool_controller.txt"
        ),
        general_prompt_path=(
            PROJECT_ROOT
            / "prompts"
            / "general_ai.txt"
        ),
        tracer=tracer,
        max_actions_per_request=(
            config.max_actions_per_request
        ),
    )


def print_runtime_summary(
    core: AtlasCore,
) -> None:
    print(
        "Model profile: "
        f"{core.runtime.profile_name}"
    )

    for item in core.runtime.summary():
        role = item["role"]
        model = item["model"]
        residency = item["residency"]
        memory = item["memory"]

        print(
            f"  {role}: {model} "
            f"[{residency}, {memory}]"
        )


def main() -> None:
    config = Config.from_env()

    try:
        core = build_core(
            config
        )
    except (
        ValueError,
        OSError,
    ) as exc:
        print(
            f"Configuration error: {exc}"
        )
        return

    print("Atlas Core v0.2.0")
    print_runtime_summary(
        core
    )
    print("Mock tools enabled.")
    print(
        "Developer trace: "
        + (
            "enabled"
            if config.trace_enabled
            else "disabled"
        )
    )

    print(
        "Preloading persistent models..."
    )

    try:
        core.preload_models()
    except ModelBackendError as exc:
        print(
            f"Core error while preloading: {exc}"
        )
        return

    print("Atlas ready.")
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

        except Exception as exc:
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
                f"{action.target} -> "
                f"{action.request}"
            )

        else:
            print(
                f"Unexpected action: {action}"
            )


if __name__ == "__main__":
    main()
