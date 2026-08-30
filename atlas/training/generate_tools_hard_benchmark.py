from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "tools_hard_eval.jsonl"
)


DEVICES = [
    {
        "device_id": "gaming_pc",
        "name": "Gaming PC",
        "type": "computer",
        "aliases": [
            "desktop",
            "main pc",
            "gaming computer",
        ],
        "online": True,
    },
    {
        "device_id": "laptop",
        "name": "Laptop",
        "type": "computer",
        "aliases": [
            "notebook",
            "portable computer",
        ],
        "online": True,
    },
]


APPLICATIONS = [
    {
        "application_id": "steam",
        "name": "Steam",
        "aliases": [
            "steam client",
        ],
    },
    {
        "application_id": "firefox",
        "name": "Firefox",
        "aliases": [
            "browser",
            "mozilla",
        ],
    },
]


KNOWN_TOOLS = [
    {
        "name": "device_list",
        "description": (
            "List devices currently known "
            "to Atlas."
        ),
        "arguments": {},
    },
    {
        "name": "device_get_status",
        "description": (
            "Get whether a known device "
            "is online."
        ),
        "arguments": {
            "device_id": "string",
        },
    },
    {
        "name": "application_list",
        "description": (
            "List applications currently "
            "known to Atlas."
        ),
        "arguments": {},
    },
    {
        "name": "computer_launch_application",
        "description": (
            "Launch a known application "
            "on a known online computer."
        ),
        "arguments": {
            "device_id": "string",
            "application_id": "string",
        },
    },
    {
        "name": "computer_close_application",
        "description": (
            "Close a known application "
            "on a known online computer."
        ),
        "arguments": {
            "device_id": "string",
            "application_id": "string",
        },
    },
    {
        "name": (
            "computer_list_running_applications"
        ),
        "description": (
            "List known applications "
            "running on a computer."
        ),
        "arguments": {
            "device_id": "string",
        },
    },
    {
        "name": "computer_get_volume",
        "description": (
            "Get computer volume."
        ),
        "arguments": {
            "device_id": "string",
        },
    },
    {
        "name": "computer_set_volume",
        "description": (
            "Set computer volume from "
            "0 through 100 percent."
        ),
        "arguments": {
            "device_id": "string",
            "volume_percent": "integer",
        },
    },
    {
        "name": "timer_create",
        "description": (
            "Create a timer."
        ),
        "arguments": {
            "duration_seconds": "integer",
            "label": "optional string",
        },
    },
    {
        "name": "timer_list",
        "description": (
            "List current timers."
        ),
        "arguments": {},
    },
    {
        "name": "timer_cancel",
        "description": (
            "Cancel a timer by timer ID."
        ),
        "arguments": {
            "timer_id": "string",
        },
    },
]


UNSEEN_TOOLS = [
    {
        "name": "computer_get_battery",
        "description": (
            "Get the battery percentage "
            "of a portable computer."
        ),
        "arguments": {
            "device_id": "string",
        },
    },
    {
        "name": "computer_get_storage",
        "description": (
            "Get free and total storage "
            "for a computer."
        ),
        "arguments": {
            "device_id": "string",
        },
    },
    {
        "name": "computer_get_cpu_temperature",
        "description": (
            "Get the current CPU "
            "temperature of a computer."
        ),
        "arguments": {
            "device_id": "string",
        },
    },
    {
        "name": "application_get_version",
        "description": (
            "Get the installed version "
            "of a known application "
            "on a computer."
        ),
        "arguments": {
            "device_id": "string",
            "application_id": "string",
        },
    },
    {
        "name": "timer_pause",
        "description": (
            "Pause a currently running "
            "timer by timer ID."
        ),
        "arguments": {
            "timer_id": "string",
        },
    },
]


BASE_PROMPT = """You are Atlas Tools.

Choose exactly one next action.

Allowed actions:
- tool_call
- clarify
- final

Use only tools, devices, applications, and IDs in RUNTIME CONTEXT.
Never invent tools.
Never invent resources.
Never claim success before a successful TOOL_RESULT.
Use one tool per turn.
do not delegate.
A request that merely mentions an action is not automatically permission to perform that action.
Ask for clarification when an essential target is genuinely ambiguous.
Keep responses concise.
"""


def action(
    action_type: str,
    **values: Any,
) -> str:
    payload = {
        "type": action_type,
        **values,
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def tool_call(
    name: str,
    arguments: dict[str, Any],
) -> str:
    return action(
        "tool_call",
        name=name,
        arguments=arguments,
    )


def clarify(
    text: str,
) -> str:
    return action(
        "clarify",
        text=text,
    )


def final(
    text: str,
) -> str:
    return action(
        "final",
        text=text,
    )


def system_prompt(
    include_unseen: bool = False,
) -> str:
    tools = list(
        KNOWN_TOOLS
    )

    if include_unseen:
        tools.extend(
            UNSEEN_TOOLS
        )

    runtime = {
        "available_tools": tools,
        "devices": DEVICES,
        "applications": APPLICATIONS,
    }

    return (
        BASE_PROMPT
        + "\nRUNTIME CONTEXT:\n"
        + json.dumps(
            runtime,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def example(
    category: str,
    user_text: str,
    expected: str,
    *,
    include_unseen: bool = False,
    prior_messages: list[
        dict[str, str]
    ] | None = None,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": system_prompt(
                include_unseen=include_unseen,
            ),
        },
    ]

    if prior_messages:
        messages.extend(
            prior_messages
        )

    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": expected,
        }
    )

    return {
        "category": category,
        "messages": messages,
    }


def build_unseen_phrasing() -> list[
    dict[str, Any]
]:
    cases: list[
        tuple[
            str,
            str,
            dict[str, Any],
        ]
    ] = []

    launch_phrases = [
        "Get {app} going over on the {device}.",
        "Fire up {app} on the {device}.",
        "I need {app} running on the {device}.",
        "{app} on the {device}, please.",
        "Can you get {app} started over there on the {device}?",
    ]

    close_phrases = [
        "Kill {app} on the {device}.",
        "Shut down {app} on the {device}.",
        "Get {app} out of the way on the {device}.",
        "I don't need {app} running on the {device} anymore.",
        "Exit {app} over on the {device}.",
    ]

    for (
        device_name,
        device_id,
    ) in [
        (
            "main pc",
            "gaming_pc",
        ),
        (
            "notebook",
            "laptop",
        ),
    ]:
        for (
            app_name,
            app_id,
        ) in [
            (
                "Steam",
                "steam",
            ),
            (
                "Mozilla",
                "firefox",
            ),
        ]:
            for phrase in launch_phrases:
                cases.append(
                    (
                        phrase.format(
                            app=app_name,
                            device=device_name,
                        ),
                        (
                            "computer_launch_"
                            "application"
                        ),
                        {
                            "device_id": device_id,
                            "application_id": app_id,
                        },
                    )
                )

            for phrase in close_phrases:
                cases.append(
                    (
                        phrase.format(
                            app=app_name,
                            device=device_name,
                        ),
                        (
                            "computer_close_"
                            "application"
                        ),
                        {
                            "device_id": device_id,
                            "application_id": app_id,
                        },
                    )
                )

    extras = [
        (
            "How loud is the desktop set right now?",
            "computer_get_volume",
            {
                "device_id": "gaming_pc",
            },
        ),
        (
            "What's the notebook audio level at?",
            "computer_get_volume",
            {
                "device_id": "laptop",
            },
        ),
        (
            "Drop the desktop sound to 18 percent.",
            "computer_set_volume",
            {
                "device_id": "gaming_pc",
                "volume_percent": 18,
            },
        ),
        (
            "Put the notebook audio at 63 percent.",
            "computer_set_volume",
            {
                "device_id": "laptop",
                "volume_percent": 63,
            },
        ),
        (
            "Give me ninety seconds on a countdown.",
            "timer_create",
            {
                "duration_seconds": 90,
                "label": None,
            },
        ),
        (
            "I need a quarter-hour countdown.",
            "timer_create",
            {
                "duration_seconds": 900,
                "label": None,
            },
        ),
        (
            "Which countdowns are active?",
            "timer_list",
            {},
        ),
        (
            "What software can you work with?",
            "application_list",
            {},
        ),
        (
            "Which machines are registered with you?",
            "device_list",
            {},
        ),
        (
            "See whether the notebook is reachable.",
            "device_get_status",
            {
                "device_id": "laptop",
            },
        ),
    ]

    cases.extend(
        extras
    )

    return [
        example(
            "unseen_phrasing",
            text,
            tool_call(
                name,
                arguments,
            ),
        )
        for (
            text,
            name,
            arguments,
        ) in cases[:50]
    ]


def build_ambiguity() -> list[
    dict[str, Any]
]:
    stems = [
        "Open Steam on my computer.",
        "Get Firefox running on one of my machines.",
        "Close Steam for me.",
        "What is the volume?",
        "Set the volume to 33 percent.",
        "Which apps are running?",
        "Start Firefox on my PC.",
        "Shut Steam down on my computer.",
        "Check whether my computer is online.",
        "Get the browser running.",
    ]

    questions = [
        "Which computer should I use?",
        "Which of your computers do you mean?",
        "Which device should I use?",
        "Which computer should I apply that to?",
        "Which machine do you mean?",
    ]

    output = []

    for index in range(
        50
    ):
        output.append(
            example(
                "ambiguity",
                stems[
                    index
                    % len(stems)
                ],
                clarify(
                    questions[
                        index
                        % len(questions)
                    ]
                ),
            )
        )

    return output


def build_hard_negatives() -> list[
    dict[str, Any]
]:
    statements = [
        "I might open Steam later.",
        "Steam is useful for games.",
        "Don't open Firefox.",
        "I was thinking about closing Steam, but leave it alone.",
        "Would 20 percent volume be too quiet?",
        "I'm considering changing my laptop volume.",
        "Don't change the volume.",
        "I may set a timer after dinner.",
        "Timers are useful when I study.",
        "Do not cancel any timers.",
        "I wonder whether Firefox is already open.",
        "Steam was running yesterday.",
        "I don't want you to start anything.",
        "Maybe I'll close Firefox myself.",
        "Leave Steam running.",
        "Keep my current volume exactly as it is.",
        "Don't start a countdown yet.",
        "I'm deciding whether I need a timer.",
        "No need to touch my computer.",
        "I was talking about opening Steam, not asking you to do it.",
    ]

    output = []

    for index in range(
        50
    ):
        statement = statements[
            index
            % len(statements)
        ]

        output.append(
            example(
                "hard_negative",
                statement,
                final(
                    "No action taken."
                ),
            )
        )

    return output


def build_failure_handling() -> list[
    dict[str, Any]
]:
    failure_templates = [
        (
            (
                "computer_launch_"
                "application"
            ),
            {
                "device_id": "gaming_pc",
                "application_id": "steam",
            },
            "DEVICE_OFFLINE",
            "Device is offline: gaming_pc",
            (
                "Your gaming PC is offline, "
                "so I couldn't open Steam."
            ),
        ),
        (
            (
                "computer_launch_"
                "application"
            ),
            {
                "device_id": "laptop",
                "application_id": "firefox",
            },
            "APPLICATION_NOT_FOUND",
            "Unknown application: firefox",
            (
                "Firefox isn't available there, "
                "so I couldn't open it."
            ),
        ),
        (
            "timer_cancel",
            {
                "timer_id": "timer_missing",
            },
            "TIMER_NOT_FOUND",
            "Unknown timer: timer_missing",
            (
                "That timer couldn't be found."
            ),
        ),
        (
            "computer_set_volume",
            {
                "device_id": "gaming_pc",
                "volume_percent": 40,
            },
            "PERMISSION_DENIED",
            "Permission denied.",
            (
                "I couldn't change the volume "
                "because permission was denied."
            ),
        ),
        (
            "device_get_status",
            {
                "device_id": "laptop",
            },
            "TIMEOUT",
            "The device did not respond.",
            (
                "Your laptop didn't respond "
                "to the status check."
            ),
        ),
    ]

    output = []

    for index in range(
        50
    ):
        (
            name,
            arguments,
            code,
            message,
            response,
        ) = failure_templates[
            index
            % len(failure_templates)
        ]

        call_json = tool_call(
            name,
            arguments,
        )

        result = {
            "call_id": (
                f"call_hard_{index:03d}"
            ),
            "name": name,
            "status": "error",
            "result": None,
            "error": {
                "code": code,
                "message": message,
            },
        }

        prior = [
            {
                "role": "user",
                "content": (
                    "Carry out the requested action."
                ),
            },
            {
                "role": "assistant",
                "content": call_json,
            },
        ]

        output.append(
            example(
                "failure_handling",
                (
                    "TOOL_RESULT\n"
                    + json.dumps(
                        result,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
                final(
                    response
                ),
                prior_messages=prior,
            )
        )

    return output


def build_unseen_tools() -> list[
    dict[str, Any]
]:
    templates = [
        (
            "How much charge does my notebook have left?",
            "computer_get_battery",
            {
                "device_id": "laptop",
            },
        ),
        (
            "Check the remaining battery on the portable computer.",
            "computer_get_battery",
            {
                "device_id": "laptop",
            },
        ),
        (
            "How much disk space is left on my desktop?",
            "computer_get_storage",
            {
                "device_id": "gaming_pc",
            },
        ),
        (
            "Check storage capacity on the notebook.",
            "computer_get_storage",
            {
                "device_id": "laptop",
            },
        ),
        (
            "What's my gaming computer CPU temperature?",
            "computer_get_cpu_temperature",
            {
                "device_id": "gaming_pc",
            },
        ),
        (
            "How hot is the processor in my notebook?",
            "computer_get_cpu_temperature",
            {
                "device_id": "laptop",
            },
        ),
        (
            "Which Steam version is installed on my desktop?",
            "application_get_version",
            {
                "device_id": "gaming_pc",
                "application_id": "steam",
            },
        ),
        (
            "Tell me the Firefox version on the notebook.",
            "application_get_version",
            {
                "device_id": "laptop",
                "application_id": "firefox",
            },
        ),
        (
            "Pause timer_ab12cd34.",
            "timer_pause",
            {
                "timer_id": "timer_ab12cd34",
            },
        ),
        (
            "Freeze the countdown timer_deadbeef for now.",
            "timer_pause",
            {
                "timer_id": "timer_deadbeef",
            },
        ),
    ]

    output = []

    for index in range(
        50
    ):
        (
            text,
            name,
            arguments,
        ) = templates[
            index
            % len(templates)
        ]

        output.append(
            example(
                "unseen_tool",
                text,
                tool_call(
                    name,
                    arguments,
                ),
                include_unseen=True,
            )
        )

    return output


def main() -> None:
    benchmark = []

    benchmark.extend(
        build_unseen_phrasing()
    )

    benchmark.extend(
        build_ambiguity()
    )

    benchmark.extend(
        build_hard_negatives()
    )

    benchmark.extend(
        build_failure_handling()
    )

    benchmark.extend(
        build_unseen_tools()
    )

    if len(
        benchmark
    ) != 250:
        raise RuntimeError(
            "Hard benchmark must contain "
            "exactly 250 examples."
        )

    rng = random.Random(
        20260829
    )

    rng.shuffle(
        benchmark
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for index, item in enumerate(
            benchmark
        ):
            record = {
                "id": (
                    f"atlas_hard_{index:04d}"
                ),
                **item,
            }

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

            output_file.write(
                "\n"
            )

    print(
        "Wrote 250 hard benchmark examples to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Categories:"
    )
    print(
        "  unseen_phrasing: 50"
    )
    print(
        "  ambiguity: 50"
    )
    print(
        "  hard_negative: 50"
    )
    print(
        "  failure_handling: 50"
    )
    print(
        "  unseen_tool: 50"
    )


if __name__ == "__main__":
    main()
