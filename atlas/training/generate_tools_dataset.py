from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "training"
)


TOOL_SUMMARIES = [
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

TOOL_BY_NAME = {
    tool["name"]: tool
    for tool in TOOL_SUMMARIES
}

TOOL_NAMES = tuple(
    TOOL_BY_NAME
)

DEVICES = [
    {
        "device_id": "gaming_pc",
        "name": "Gaming PC",
        "aliases": [
            "gaming pc",
            "desktop",
            "main pc",
            "gaming computer",
            "desktop computer",
        ],
        "online": True,
    },
    {
        "device_id": "laptop",
        "name": "Laptop",
        "aliases": [
            "laptop",
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
            "Steam",
            "steam",
            "Steam client",
        ],
    },
    {
        "application_id": "firefox",
        "name": "Firefox",
        "aliases": [
            "Firefox",
            "firefox",
            "Mozilla",
            "browser",
        ],
    },
]

BASE_SYSTEM = """You are Atlas Tools.

Choose exactly one next action.

Allowed actions:
- tool_call
- clarify
- final

Use only tools, devices, and applications in runtime context.
Never invent tool results.
Never claim success before a successful TOOL_RESULT.
Use one tool per turn.
Do not delegate.
A request that merely mentions an action is not automatically permission to perform it.
Ask for clarification when an essential target is genuinely ambiguous.
Keep final responses concise.
User-owned resources are "your" resources.
"""


def build_runtime_context(
    rng: random.Random,
    required_tools: set[str],
    minimum_tools: int = 5,
) -> str:
    unknown = (
        required_tools
        - set(
            TOOL_BY_NAME
        )
    )

    if unknown:
        raise ValueError(
            "Unknown required tools: "
            f"{sorted(unknown)}"
        )

    visible = set(
        required_tools
    )

    candidates = [
        name
        for name in TOOL_NAMES
        if name not in visible
    ]

    rng.shuffle(
        candidates
    )

    while (
        len(visible)
        < minimum_tools
        and candidates
    ):
        visible.add(
            candidates.pop()
        )

    for name in candidates:
        if rng.random() < 0.45:
            visible.add(
                name
            )

    visible_tools = [
        TOOL_BY_NAME[name]
        for name in visible
    ]

    rng.shuffle(
        visible_tools
    )

    context = {
        "available_tools": (
            visible_tools
        ),
        "devices": DEVICES,
        "applications": (
            APPLICATIONS
        ),
    }

    return (
        BASE_SYSTEM
        + "\nRUNTIME CONTEXT:\n"
        + json.dumps(
            context,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def runtime_context() -> str:
    return build_runtime_context(
        random.Random(
            20260829
        ),
        set(
            TOOL_NAMES
        ),
        minimum_tools=len(
            TOOL_NAMES
        ),
    )


TRAIN_PHRASES = {
    "device_list": [
        "List my devices.",
        "What devices do I have?",
        "Show my available computers.",
        "Which devices does Atlas know about?",
        "Give me a list of devices.",
        "What machines are registered?",
        "Tell me all my devices.",
        "Show my device inventory.",
        "Which computers can you access?",
        "What computers are available to you?",
    ],
    "device_get_status": [
        "Is my {device} online?",
        "Check the status of my {device}.",
        "Is {device} connected?",
        "Tell me if the {device} is reachable.",
        "Is the {device} up?",
        "Check whether {device} is online.",
        "What's the status of {device}?",
        "Can you see my {device}?",
        "Is {device} responding?",
        "Check connectivity for my {device}.",
    ],
    "application_list": [
        "What applications can Atlas open?",
        "List my known applications.",
        "What apps are available?",
        "Give me a list of apps.",
        "Which programs can you launch?",
        "Show my available software.",
        "What programs do you know about?",
        "List the applications you can use.",
        "Which apps are registered?",
        "Show the application catalog.",
    ],
    "computer_launch_application": [
        "Open {app} on my {device}.",
        "Launch {app} on the {device}.",
        "Start {app} on my {device}.",
        "Can you run {app} on my {device}?",
        "Please open {app} on the {device}.",
        "Get {app} running on my {device}.",
        "Fire up {app} over on my {device}.",
        "I need {app} running on the {device}.",
        "Run {app} using my {device}.",
        "Start up {app} on my {device}.",
    ],
    "computer_close_application": [
        "Close {app} on my {device}.",
        "Quit {app} on the {device}.",
        "Stop {app} on my {device}.",
        "Please close {app} on my {device}.",
        "Shut down {app} on the {device}.",
        "End {app} on my {device}.",
        "Exit {app} on the {device}.",
        "Can you close {app} on my {device}?",
        "Get {app} closed on my {device}.",
        "Stop running {app} on the {device}.",
    ],
    "computer_list_running_applications": [
        "What apps are running on my {device}?",
        "List running applications on my {device}.",
        "Show me what is open on the {device}.",
        "Which programs are active on my {device}?",
        "Give me the running apps on my {device}.",
        "What is currently running on the {device}?",
        "List the open applications on my {device}.",
        "What software is open on my {device}?",
        "Show running apps on the {device}.",
        "Tell me which apps are active on my {device}.",
    ],
    "computer_get_volume": [
        "What is the volume on my {device}?",
        "Check my {device} volume.",
        "Tell me the current volume of the {device}.",
        "What volume level is set on my {device}?",
        "Give me the volume percentage for my {device}.",
        "How loud is my {device} right now?",
        "What is the sound level on the {device}?",
        "Report the volume of my {device}.",
        "What's the current volume on my {device}?",
        "Show me the volume setting for the {device}.",
    ],
    "computer_set_volume": [
        "Set my {device} volume to {volume}%.",
        "Change the volume on my {device} to {volume} percent.",
        "Set the {device} to {volume}% volume.",
        "Adjust my {device} volume to {volume} percent.",
        "Set the sound level on the {device} to {volume}%.",
        "Please set volume to {volume}% on my {device}.",
        "Make the {device} volume {volume} percent.",
        "Set volume {volume}% on my {device}.",
        "Change volume to {volume}% for the {device}.",
        "Put my {device} at {volume}% volume.",
    ],
    "timer_create": [
        "Set a timer for {duration}.",
        "Create a timer for {duration}.",
        "Start a {duration} timer.",
        "I need a timer for {duration}.",
        "Please set a timer for {duration}.",
        "Make a timer that lasts {duration}.",
        "Set a countdown for {duration}.",
        "Give me a {duration} countdown.",
        "Start a timer lasting {duration}.",
        "Create a {duration} countdown.",
    ],
    "timer_list": [
        "List my timers.",
        "What timers are running?",
        "Show my current timers.",
        "Give me a list of active timers.",
        "What countdowns do I have?",
        "Tell me about my timers.",
        "Do I have any timers set?",
        "Show all timers.",
        "What timers exist right now?",
        "List all my timer entries.",
    ],
    "timer_cancel": [
        "Cancel {timer_id}.",
        "Stop timer {timer_id}.",
        "Delete the timer {timer_id}.",
        "Remove timer {timer_id}.",
        "Please cancel timer {timer_id}.",
        "Terminate timer {timer_id}.",
        "Cancel the timer with ID {timer_id}.",
        "Stop the timer called {timer_id}.",
        "Get rid of timer {timer_id}.",
        "Cancel {timer_id} now.",
    ],
}


TRAIN_PHRASES["device_list"].extend([
    "What machines have I got?",
    "Show me what computers are hooked up.",
    "What can you see device-wise?",
    "Give me the devices you know about.",
    "What boxes are registered with Atlas?",
    "Which of my computers are available?",
])

TRAIN_PHRASES["device_get_status"].extend([
    "Is {device} alive?",
    "Can {device} hear you?",
    "See if {device} is there.",
    "Is {device} reachable right now?",
    "Check if {device} is still around.",
    "See whether {device} is responding for me.",
])

TRAIN_PHRASES["application_list"].extend([
    "What can you open for me?",
    "What software do you know how to start?",
    "Show me the apps you know.",
    "What programs are hooked into Atlas?",
    "What apps can you work with right now?",
    "Give me the software you have access to.",
])

TRAIN_PHRASES["computer_launch_application"].extend([
    "Throw {app} up on the {device}.",
    "Get {app} going on {device}.",
    "Pop {app} open on my {device}.",
    "Can you bring {app} up on {device}?",
    "I want {app} up on the {device}.",
    "Go ahead and start {app} over on {device}.",
    "{app} on my {device}, please.",
    "Get me into {app} on the {device}.",
])

TRAIN_PHRASES["computer_close_application"].extend([
    "Kill {app} on the {device}.",
    "Get rid of {app} on my {device}.",
    "Take {app} down on the {device}.",
    "Can you get {app} closed on {device}?",
    "I'm done with {app} on my {device}.",
    "Knock {app} off on the {device}.",
    "Close out of {app} on {device}.",
    "Get {app} off my {device}.",
])

TRAIN_PHRASES["computer_list_running_applications"].extend([
    "What's up on my {device} right now?",
    "Anything running on the {device}?",
    "What have I got open on {device}?",
    "What's currently up on my {device}?",
    "Tell me what's running over on {device}.",
    "What programs have I got going on the {device}?",
    "Show me what's open over there on {device}.",
    "What's active on my {device}?",
])

TRAIN_PHRASES["computer_get_volume"].extend([
    "What's my {device} sitting at volume-wise?",
    "How loud have I got the {device}?",
    "Where's the sound at on {device}?",
    "What's {device} at for audio?",
    "Check how loud my {device} is set.",
    "What percent is the sound on {device}?",
    "Give me the current audio level on my {device}.",
    "Where's my {device} volume sitting?",
])

TRAIN_PHRASES["computer_set_volume"].extend([
    "Put {device} at {volume}%.",
    "Bring the {device} down to {volume}% volume.",
    "Make my {device} {volume}% loud.",
    "Can you put {device} on {volume}% volume?",
    "I want the sound at {volume}% on {device}.",
    "Move my {device} audio to {volume}%.",
    "Set {device} audio at {volume}% for me.",
    "Make it {volume}% on my {device}.",
])

TRAIN_PHRASES["timer_create"].extend([
    "Give me {duration} on the clock.",
    "Count down {duration} for me.",
    "I need {duration} on a countdown.",
    "Can you give me a {duration} timer?",
    "Put {duration} on a timer for me.",
    "Start counting down {duration}.",
    "Give me a timer for the next {duration}.",
    "Clock {duration} for me.",
])

TRAIN_PHRASES["timer_list"].extend([
    "What countdowns have I got going?",
    "Anything on the clock right now?",
    "What timers have I got active?",
    "Show me what's counting down.",
    "What do I currently have timed?",
    "Any timers running right now?",
])

TRAIN_PHRASES["timer_cancel"].extend([
    "Kill {timer_id}.",
    "Get {timer_id} off the clock.",
    "Drop timer {timer_id}.",
    "I don't need {timer_id} anymore.",
    "Can you stop {timer_id} for me?",
    "End the countdown {timer_id}.",
    "Remove {timer_id} from my timers.",
    "Shut off {timer_id}.",
])

EVAL_PHRASES = {
    "device_list": [
        "Give me a rundown of my devices.",
        "What hardware do I have registered?",
        "Show me my machines.",
        "Which computers are in Atlas?",
        "Enumerate the devices you know.",
    ],
    "device_get_status": [
        "Is the {device} up and running?",
        "Can you tell me whether {device} is online?",
        "Is my {device} currently reachable?",
        "Check if {device} is responding.",
        "What's the connectivity state of {device}?",
    ],
    "application_list": [
        "What programs are at my disposal?",
        "Show the software Atlas knows.",
        "Which apps can you work with?",
        "Give me the application inventory.",
        "What software is registered?",
    ],
    "computer_launch_application": [
        "Bring up {app} on the {device}.",
        "Could you get {app} going over there on {device}?",
        "I want {app} running over on {device}.",
        "Have the {device} start {app}.",
        "Put {app} up on my {device}.",
    ],
    "computer_close_application": [
        "Get {app} out of the way on {device}.",
        "Please stop {app} over on {device}.",
        "End the {app} session on {device}.",
        "Have {device} exit {app}.",
        "Take {app} down on {device}.",
    ],
    "computer_list_running_applications": [
        "What programs are currently active on my {device}?",
        "Show which applications are open on {device}.",
        "What's running over on {device}?",
        "Give me the active software on {device}.",
        "Which apps does {device} have open?",
    ],
    "computer_get_volume": [
        "How loud is the {device} set right now?",
        "What's the audio level on {device}?",
        "Tell me the sound setting of {device}.",
        "Where is {device}'s volume currently set?",
        "Report the current audio percentage for {device}.",
    ],
    "computer_set_volume": [
        "Adjust {device} to {volume} percent volume.",
        "Put the audio on {device} at {volume}%.",
        "Make {device}'s volume {volume} percent.",
        "Set {device}'s sound to {volume}%.",
        "I want {device} at {volume}% volume.",
    ],
    "timer_create": [
        "Give me a {duration} countdown.",
        "Start counting down from {duration}.",
        "I need {duration} on a timer.",
        "Begin a timer that runs for {duration}.",
        "Put {duration} on the clock.",
    ],
    "timer_list": [
        "Which countdowns are active?",
        "What timers do I currently have?",
        "Show the countdowns that exist.",
        "Tell me which timers are set.",
        "What's on my timer list?",
    ],
    "timer_cancel": [
        "Get rid of {timer_id}.",
        "Stop countdown {timer_id}.",
        "Remove the timer identified as {timer_id}.",
        "End timer {timer_id}.",
        "I want {timer_id} cancelled.",
    ],
}


HARD_NEGATIVE_TRAIN = [
    "I might open {app} on my {device} later.",
    "Don't open {app} on my {device}.",
    "I was thinking about launching {app} on {device}, but don't do it.",
    "Would {volume}% volume on {device} be too quiet?",
    "I don't want you to change my {device} volume.",
    "Maybe I'll set a timer for {duration} later.",
    "Timers are useful when I study.",
    "Do not cancel {timer_id}.",
    "Leave {app} running on {device}.",
    "Don't close {app} on {device}.",
    "Don't check whether {app} is running on {device}.",
    "I'll probably open {app} on {device} myself.",
    "No need to touch {device}.",
    "I'm talking about opening {app}, not asking you to do it.",
    "Keep everything on {device} as it is.",
    "Don't start anything on {device} yet.",
    "I may change the volume on {device} later.",
    "Would setting a {duration} timer help?",
    "{app} should stay open on {device}.",
    "Don't make any changes to {device}.",
]


HARD_NEGATIVE_EVAL = [
    "For now, leave {app} alone on {device}.",
    "I'm only considering opening {app} on {device}.",
    "Do not alter the {device} volume from where it is.",
    "I'm curious whether a {duration} timer would be useful.",
    "No action needed on {device}; I was just thinking about {app}.",
    "Keep {app} exactly as it is on {device}.",
    "Don't cancel {timer_id}; I still need it.",
    "I might adjust {device} to {volume}% later, not now.",
    "I'm not asking you to launch {app}.",
    "Leave my timers unchanged.",
]


AMBIGUITY_TRAIN = [
    (
        "computer_launch_application",
        "Open {app} on my computer.",
    ),
    (
        "computer_launch_application",
        "Start {app} for me.",
    ),
    (
        "computer_close_application",
        "Close {app}.",
    ),
    (
        "computer_get_volume",
        "What is the volume?",
    ),
    (
        "computer_set_volume",
        "Set the volume to {volume} percent.",
    ),
    (
        "computer_list_running_applications",
        "What apps are running?",
    ),
    (
        "device_get_status",
        "Check whether my computer is online.",
    ),
]


AMBIGUITY_EVAL = [
    (
        "computer_launch_application",
        "Get {app} running on one of my machines.",
    ),
    (
        "computer_close_application",
        "Shut {app} down on my computer.",
    ),
    (
        "computer_get_volume",
        "How loud is my computer?",
    ),
    (
        "computer_set_volume",
        "Put my computer at {volume}% volume.",
    ),
    (
        "computer_list_running_applications",
        "Which programs are open on my machine?",
    ),
    (
        "device_get_status",
        "Is my machine reachable?",
    ),
]


CLARIFICATIONS = [
    "Which computer should I use?",
    "Which of your computers do you mean?",
    "Which device should I apply that to?",
    "Do you mean your gaming PC or laptop?",
    "Which machine should I use?",
]


AMBIGUITY_TRAIN_V021 = [
    {"tool_name": "computer_launch_application", "template": "Open {app} on my computer.", "missing": "device"},
    {"tool_name": "computer_launch_application", "template": "Start {app} for me.", "missing": "device"},
    {"tool_name": "computer_launch_application", "template": "Get {app} going somewhere for me.", "missing": "device"},
    {"tool_name": "computer_close_application", "template": "Close {app}.", "missing": "device"},
    {"tool_name": "computer_get_volume", "template": "What is the volume?", "missing": "device"},
    {"tool_name": "computer_set_volume", "template": "Set the volume to {volume} percent.", "missing": "device"},
    {"tool_name": "computer_list_running_applications", "template": "What apps are running?", "missing": "device"},
    {"tool_name": "device_get_status", "template": "Check whether my computer is online.", "missing": "device"},
    {"tool_name": "computer_launch_application", "template": "Open an app on my {device}.", "missing": "application"},
    {"tool_name": "computer_close_application", "template": "Close the app on my {device}.", "missing": "application"},
    {"tool_name": "timer_cancel", "template": "Cancel my timer.", "missing": "timer_id"},
    {"tool_name": "timer_cancel", "template": "Stop the countdown.", "missing": "timer_id"},
    {"tool_name": "computer_launch_application", "template": "Open something for me.", "missing": "application_and_device"}
]

AMBIGUITY_EVAL_V021 = [
    {"tool_name": "computer_launch_application", "template": "Put {app} up somewhere for me.", "missing": "device"},
    {"tool_name": "computer_close_application", "template": "Get {app} closed on one of my machines.", "missing": "device"},
    {"tool_name": "computer_get_volume", "template": "What's my sound set to?", "missing": "device"},
    {"tool_name": "computer_set_volume", "template": "Set my audio to {volume}%.", "missing": "device"},
    {"tool_name": "computer_list_running_applications", "template": "What is open at the moment?", "missing": "device"},
    {"tool_name": "device_get_status", "template": "Is one of my computers online?", "missing": "device"},
    {"tool_name": "computer_launch_application", "template": "Launch a program on my {device}.", "missing": "application"},
    {"tool_name": "computer_close_application", "template": "Close an application on my {device}.", "missing": "application"},
    {"tool_name": "timer_cancel", "template": "Cancel the timer I mean.", "missing": "timer_id"},
    {"tool_name": "computer_launch_application", "template": "Start something for me.", "missing": "application_and_device"}
]

CLARIFICATIONS_V021 = {
    "device": [
        "Which computer should I use?",
        "Which of your computers do you mean?",
        "Do you mean your gaming PC or laptop?",
        "Which machine should I use?",
        "Which device should I apply that to?"
    ],
    "application": [
        "Which application do you mean?",
        "Which app should I use?",
        "What application should I act on?",
        "Which program do you mean?"
    ],
    "timer_id": [
        "Which timer do you mean?",
        "Which timer should I cancel?",
        "What timer ID should I use?",
        "Which countdown are you referring to?"
    ],
    "application_and_device": [
        "Which application and computer do you mean?",
        "What should I open, and on which computer?",
        "Which app should I start on which device?"
    ]
}

def choose_device(
    rng: random.Random,
) -> dict[str, Any]:
    return rng.choice(
        DEVICES
    )


def choose_application(
    rng: random.Random,
) -> dict[str, Any]:
    return rng.choice(
        APPLICATIONS
    )


def choose_device_text(
    rng: random.Random,
    device: dict[str, Any],
) -> str:
    return rng.choice(
        [
            device["name"],
            *device["aliases"],
        ]
    )


def choose_application_text(
    rng: random.Random,
    application: dict[str, Any],
) -> str:
    return rng.choice(
        [
            application["name"],
            *application["aliases"],
        ]
    )


def random_timer_id(
    rng: random.Random,
) -> str:
    characters = (
        "0123456789abcdef"
    )

    suffix = "".join(
        rng.choice(
            characters
        )
        for _ in range(8)
    )

    return (
        f"timer_{suffix}"
    )


def random_call_id(
    rng: random.Random,
) -> str:
    return (
        random_timer_id(
            rng
        )
        .replace(
            "timer_",
            "call_",
            1,
        )
    )


def random_duration(
    rng: random.Random,
) -> tuple[int, str]:
    unit = rng.choice(
        [
            "seconds",
            "minutes",
            "hours",
        ]
    )

    if unit == "seconds":
        value = rng.randint(
            1,
            59,
        )

        seconds = value

    elif unit == "minutes":
        value = rng.randint(
            1,
            120,
        )

        seconds = (
            value * 60
        )

    else:
        value = rng.randint(
            1,
            24,
        )

        seconds = (
            value * 3600
        )

    if value == 1:
        display_unit = (
            unit[:-1]
        )
    else:
        display_unit = unit

    return (
        seconds,
        f"{value} {display_unit}",
    )


def sample_values(
    rng: random.Random,
) -> dict[str, Any]:
    device = choose_device(
        rng
    )

    application = (
        choose_application(
            rng
        )
    )

    (
        duration_seconds,
        duration_text,
    ) = random_duration(
        rng
    )

    return {
        "device": device,
        "device_text": (
            choose_device_text(
                rng,
                device,
            )
        ),
        "application": (
            application
        ),
        "application_text": (
            choose_application_text(
                rng,
                application,
            )
        ),
        "volume": rng.randint(
            0,
            100,
        ),
        "duration_seconds": (
            duration_seconds
        ),
        "duration_text": (
            duration_text
        ),
        "timer_id": (
            random_timer_id(
                rng
            )
        ),
    }


def format_phrase(
    template: str,
    values: dict[str, Any],
) -> str:
    return template.format(
        device=(
            values[
                "device_text"
            ]
        ),
        app=(
            values[
                "application_text"
            ]
        ),
        volume=(
            values[
                "volume"
            ]
        ),
        duration=(
            values[
                "duration_text"
            ]
        ),
        timer_id=(
            values[
                "timer_id"
            ]
        ),
    )


def arguments_for_tool(
    tool_name: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    if tool_name in {
        "device_list",
        "application_list",
        "timer_list",
    }:
        return {}

    if tool_name == (
        "device_get_status"
    ):
        return {
            "device_id": (
                values[
                    "device"
                ][
                    "device_id"
                ]
            )
        }

    if tool_name in {
        "computer_launch_application",
        "computer_close_application",
    }:
        return {
            "device_id": (
                values[
                    "device"
                ][
                    "device_id"
                ]
            ),
            "application_id": (
                values[
                    "application"
                ][
                    "application_id"
                ]
            ),
        }

    if tool_name in {
        "computer_list_running_applications",
        "computer_get_volume",
    }:
        return {
            "device_id": (
                values[
                    "device"
                ][
                    "device_id"
                ]
            )
        }

    if tool_name == (
        "computer_set_volume"
    ):
        return {
            "device_id": (
                values[
                    "device"
                ][
                    "device_id"
                ]
            ),
            "volume_percent": (
                values[
                    "volume"
                ]
            ),
        }

    if tool_name == (
        "timer_create"
    ):
        return {
            "duration_seconds": (
                values[
                    "duration_seconds"
                ]
            ),
            "label": None,
        }

    if tool_name == (
        "timer_cancel"
    ):
        return {
            "timer_id": (
                values[
                    "timer_id"
                ]
            )
        }

    raise ValueError(
        "Unsupported tool: "
        f"{tool_name}"
    )


def make_positive_example(
    rng: random.Random,
    phrases: dict[
        str,
        list[str],
    ],
) -> dict[str, Any]:
    tool_name = rng.choice(
        TOOL_NAMES
    )

    values = sample_values(
        rng
    )

    user_text = format_phrase(
        rng.choice(
            phrases[
                tool_name
            ]
        ),
        values,
    )

    return {
        "category": (
            "positive_tool_call"
        ),
        "messages": [
            {
                "role": "system",
                "content": (
                    build_runtime_context(
                        rng,
                        {
                            tool_name
                        },
                    )
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
            {
                "role": "assistant",
                "content": tool_call(
                    tool_name,
                    arguments_for_tool(
                        tool_name,
                        values,
                    ),
                ),
            },
        ],
    }


def make_hard_negative_example(
    rng: random.Random,
    phrases: list[str],
) -> dict[str, Any]:
    values = sample_values(
        rng
    )

    user_text = format_phrase(
        rng.choice(
            phrases
        ),
        values,
    )

    return {
        "category": (
            "hard_negative"
        ),
        "messages": [
            {
                "role": "system",
                "content": (
                    build_runtime_context(
                        rng,
                        set(),
                    )
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
            {
                "role": "assistant",
                "content": final(
                    "No action taken."
                ),
            },
        ],
    }


def make_ambiguity_example(
    rng: random.Random,
    phrases: list[
        tuple[
            str,
            str,
        ]
    ],
) -> dict[str, Any]:
    (
        tool_name,
        template,
    ) = rng.choice(
        phrases
    )

    values = sample_values(
        rng
    )

    user_text = format_phrase(
        template,
        values,
    )

    return {
        "category": (
            "ambiguity"
        ),
        "messages": [
            {
                "role": "system",
                "content": (
                    build_runtime_context(
                        rng,
                        {
                            tool_name
                        },
                    )
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
            {
                "role": "assistant",
                "content": clarify(
                    rng.choice(
                        CLARIFICATIONS
                    )
                ),
            },
        ],
    }


def ambiguity_resolution_text(
    missing: str,
    values: dict[str, Any],
) -> str:
    if missing == "device":
        return values["device_text"]

    if missing == "application":
        return values["application_text"]

    if missing == "timer_id":
        return values["timer_id"]

    if missing == "application_and_device":
        return (
            f"{values['application_text']} on my "
            f"{values['device_text']}"
        )

    raise ValueError(
        "Unsupported ambiguity type: "
        f"{missing}"
    )


def make_ambiguity_example_v021(
    rng: random.Random,
    phrases: list[dict[str, str]],
) -> dict[str, Any]:
    specification = rng.choice(phrases)
    tool_name = specification["tool_name"]
    template = specification["template"]
    missing = specification["missing"]

    values = sample_values(rng)
    user_text = format_phrase(template, values)

    messages = [
        {
            "role": "system",
            "content": build_runtime_context(rng, {tool_name}),
        },
        {
            "role": "user",
            "content": user_text,
        },
        {
            "role": "assistant",
            "content": clarify(
                rng.choice(CLARIFICATIONS_V021[missing])
            ),
        },
    ]

    if rng.random() < 0.55:
        messages.extend([
            {
                "role": "user",
                "content": ambiguity_resolution_text(
                    missing,
                    values,
                ),
            },
            {
                "role": "assistant",
                "content": tool_call(
                    tool_name,
                    arguments_for_tool(
                        tool_name,
                        values,
                    ),
                ),
            },
        ])

    return {
        "category": "ambiguity",
        "messages": messages,
    }


def tool_call(
    name: str,
    arguments: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "name": name,
            "arguments": arguments,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def final(
    text: str,
) -> str:
    return json.dumps(
        {
            "type": "final",
            "text": text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def clarify(
    text: str,
) -> str:
    return json.dumps(
        {
            "type": "clarify",
            "text": text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def make_single_turn_examples() -> list[
    tuple[str, str]
]:
    examples: list[
        tuple[str, str]
    ] = []

    for user_text in [
        "List my devices.",
        "What devices do I have?",
        "Show my available computers.",
        "Which devices does Atlas know about?",
    ]:
        examples.append(
            (
                user_text,
                tool_call(
                    "device_list",
                    {},
                ),
            )
        )

    for device in DEVICES:
        device_id = device[
            "device_id"
        ]

        name = device[
            "name"
        ]

        for user_text in [
            f"Is my {name} online?",
            f"Check the status of my {name}.",
            f"Is {name} connected?",
        ]:
            examples.append(
                (
                    user_text,
                    tool_call(
                        "device_get_status",
                        {
                            "device_id": (
                                device_id
                            )
                        },
                    )
                )
            )

    for user_text in [
        "What applications can Atlas open?",
        "List my known applications.",
        "What apps are available?",
    ]:
        examples.append(
            (
                user_text,
                tool_call(
                    "application_list",
                    {},
                ),
            )
        )

    for device in DEVICES:
        for app in APPLICATIONS:
            device_id = device[
                "device_id"
            ]

            device_name = device[
                "name"
            ]

            app_id = app[
                "application_id"
            ]

            app_name = app[
                "name"
            ]

            launch_variants = [
                (
                    f"Open {app_name} on my "
                    f"{device_name}."
                ),
                (
                    f"Launch {app_name} on the "
                    f"{device_name}."
                ),
                (
                    f"Start {app_name} on my "
                    f"{device_name}."
                ),
            ]

            close_variants = [
                (
                    f"Close {app_name} on my "
                    f"{device_name}."
                ),
                (
                    f"Quit {app_name} on the "
                    f"{device_name}."
                ),
                (
                    f"Stop {app_name} on my "
                    f"{device_name}."
                ),
            ]

            for user_text in launch_variants:
                examples.append(
                    (
                        user_text,
                        tool_call(
                            (
                                "computer_launch_"
                                "application"
                            ),
                            {
                                "device_id": (
                                    device_id
                                ),
                                "application_id": (
                                    app_id
                                ),
                            },
                        ),
                    )
                )

            for user_text in close_variants:
                examples.append(
                    (
                        user_text,
                        tool_call(
                            (
                                "computer_close_"
                                "application"
                            ),
                            {
                                "device_id": (
                                    device_id
                                ),
                                "application_id": (
                                    app_id
                                ),
                            },
                        ),
                    )
                )

        for user_text in [
            (
                "What apps are running on my "
                f"{device['name']}?"
            ),
            (
                "List running applications on "
                f"my {device['name']}."
            ),
        ]:
            examples.append(
                (
                    user_text,
                    tool_call(
                        (
                            "computer_list_"
                            "running_applications"
                        ),
                        {
                            "device_id": (
                                device[
                                    "device_id"
                                ]
                            )
                        },
                    ),
                )
            )

        for user_text in [
            (
                "What is the volume on my "
                f"{device['name']}?"
            ),
            (
                "Check my "
                f"{device['name']} volume."
            ),
        ]:
            examples.append(
                (
                    user_text,
                    tool_call(
                        "computer_get_volume",
                        {
                            "device_id": (
                                device[
                                    "device_id"
                                ]
                            ),
                        },
                    ),
                )
            )

        for volume in [
            0,
            10,
            25,
            50,
            75,
            100,
        ]:
            examples.append(
                (
                    (
                        "Set my "
                        f"{device['name']} "
                        f"volume to {volume}%."
                    ),
                    tool_call(
                        "computer_set_volume",
                        {
                            "device_id": (
                                device[
                                    "device_id"
                                ]
                            ),
                            "volume_percent": (
                                volume
                            ),
                        },
                    ),
                )
            )

    timer_cases = [
        (
            "Set a timer for 30 seconds.",
            30,
            None,
        ),
        (
            "Set a timer for 1 minute.",
            60,
            None,
        ),
        (
            "Set a timer for 5 minutes.",
            300,
            None,
        ),
        (
            "Set a 10 minute timer.",
            600,
            None,
        ),
        (
            "Start a 20 minute study timer.",
            1200,
            "study",
        ),
        (
            "Set a 45 minute timer for homework.",
            2700,
            "homework",
        ),
    ]

    for (
        user_text,
        duration,
        label,
    ) in timer_cases:
        examples.append(
            (
                user_text,
                tool_call(
                    "timer_create",
                    {
                        "duration_seconds": (
                            duration
                        ),
                        "label": label,
                    },
                )
            )
        )

    for user_text in [
        "List my timers.",
        "What timers are running?",
        "Show my current timers.",
    ]:
        examples.append(
            (
                user_text,
                tool_call(
                    "timer_list",
                    {},
                ),
            )
        )

    for timer_id in [
        "timer_ab12cd34",
        "timer_1234abcd",
        "timer_deadbeef",
    ]:
        examples.append(
            (
                f"Cancel {timer_id}.",
                tool_call(
                    "timer_cancel",
                    {
                        "timer_id": timer_id,
                    },
                ),
            )
        )

    examples.extend(
        [
            (
                "Open Steam on my computer.",
                clarify(
                    (
                        "Which computer should "
                        "I use?"
                    )
                ),
            ),
            (
                "Close Firefox.",
                clarify(
                    (
                        "Which computer should "
                        "I use?"
                    )
                ),
            ),
            (
                "Set the volume to 40%.",
                clarify(
                    (
                        "Which computer should "
                        "I change?"
                    )
                ),
            ),
        ]
    )

    return examples


def success_followup(
    user_text: str,
    action_json: str,
) -> dict[str, Any]:
    action = json.loads(
        action_json
    )

    name = action[
        "name"
    ]

    arguments = action[
        "arguments"
    ]

    if name == "computer_launch_application":
        result = {
            "call_id": "call_training",
            "name": name,
            "status": "success",
            "result": {
                "launched": True,
                "already_running": False,
            },
            "error": None,
        }

        response = final(
            "The application has been launched."
        )

    elif name == "computer_close_application":
        result = {
            "call_id": "call_training",
            "name": name,
            "status": "success",
            "result": {
                "closed": True,
                "was_running": True,
            },
            "error": None,
        }

        response = final(
            "The application has been closed."
        )

    elif name == "computer_set_volume":
        result = {
            "call_id": "call_training",
            "name": name,
            "status": "success",
            "result": {
                "device_id": (
                    arguments[
                        "device_id"
                    ]
                ),
                "volume_percent": (
                    arguments[
                        "volume_percent"
                    ]
                ),
                "changed": True,
            },
            "error": None,
        }

        response = final(
            (
                "Your volume has been set to "
                f"{arguments['volume_percent']}%."
            )
        )

    elif name == "timer_create":
        result = {
            "call_id": "call_training",
            "name": name,
            "status": "success",
            "result": {
                "timer_id": (
                    "timer_training"
                ),
                "duration_seconds": (
                    arguments[
                        "duration_seconds"
                    ]
                ),
                "label": (
                    arguments.get(
                        "label"
                    )
                ),
            },
            "error": None,
        }

        response = final(
            "Your timer has been created."
        )

    else:
        raise ValueError(
            "Unsupported followup tool: "
            f"{name}"
        )

    return {
        "messages": [
            {
                "role": "system",
                "content": runtime_context(),
            },
            {
                "role": "user",
                "content": user_text,
            },
            {
                "role": "assistant",
                "content": action_json,
            },
            {
                "role": "user",
                "content": (
                    "TOOL_RESULT\n"
                    + json.dumps(
                        result,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]
    }


ERRORS = [
    (
        "DEVICE_OFFLINE",
        "The device is offline.",
    ),
    (
        "DEVICE_NOT_FOUND",
        "The device could not be found.",
    ),
    (
        "APPLICATION_NOT_FOUND",
        "The application could not be found.",
    ),
    (
        "TIMER_NOT_FOUND",
        "The timer could not be found.",
    ),
    (
        "INVALID_ARGUMENT",
        "One or more arguments were invalid.",
    ),
    (
        "PERMISSION_DENIED",
        "Permission was denied.",
    ),
    (
        "TIMEOUT",
        "The operation timed out.",
    ),
    (
        "TOOL_EXECUTION_FAILED",
        "The operation failed.",
    ),
]


ERROR_BY_CODE = {
    code: message
    for code, message in ERRORS
}

TOOL_FAILURE_CODES = {
    "device_list": [
        "TIMEOUT",
        "PERMISSION_DENIED",
        "TOOL_EXECUTION_FAILED",
    ],
    "device_get_status": [
        "DEVICE_NOT_FOUND",
        "DEVICE_OFFLINE",
        "TIMEOUT",
        "PERMISSION_DENIED",
        "TOOL_EXECUTION_FAILED",
    ],
    "application_list": [
        "TIMEOUT",
        "PERMISSION_DENIED",
        "TOOL_EXECUTION_FAILED",
    ],
    "computer_launch_application": [
        "DEVICE_NOT_FOUND",
        "DEVICE_OFFLINE",
        "APPLICATION_NOT_FOUND",
        "PERMISSION_DENIED",
        "TIMEOUT",
        "TOOL_EXECUTION_FAILED",
    ],
    "computer_close_application": [
        "DEVICE_NOT_FOUND",
        "DEVICE_OFFLINE",
        "APPLICATION_NOT_FOUND",
        "PERMISSION_DENIED",
        "TIMEOUT",
        "TOOL_EXECUTION_FAILED",
    ],
    "computer_list_running_applications": [
        "DEVICE_NOT_FOUND",
        "DEVICE_OFFLINE",
        "PERMISSION_DENIED",
        "TIMEOUT",
        "TOOL_EXECUTION_FAILED",
    ],
    "computer_get_volume": [
        "DEVICE_NOT_FOUND",
        "DEVICE_OFFLINE",
        "PERMISSION_DENIED",
        "TIMEOUT",
        "TOOL_EXECUTION_FAILED",
    ],
    "computer_set_volume": [
        "DEVICE_NOT_FOUND",
        "DEVICE_OFFLINE",
        "INVALID_ARGUMENT",
        "PERMISSION_DENIED",
        "TIMEOUT",
        "TOOL_EXECUTION_FAILED",
    ],
    "timer_create": [
        "INVALID_ARGUMENT",
        "PERMISSION_DENIED",
        "TIMEOUT",
        "TOOL_EXECUTION_FAILED",
    ],
    "timer_list": [
        "PERMISSION_DENIED",
        "TIMEOUT",
        "TOOL_EXECUTION_FAILED",
    ],
    "timer_cancel": [
        "TIMER_NOT_FOUND",
        "PERMISSION_DENIED",
        "TIMEOUT",
        "TOOL_EXECUTION_FAILED",
    ],
}


def choose_failure_for_tool(
    rng: random.Random,
    tool_name: str,
) -> tuple[str, str]:
    try:
        codes = TOOL_FAILURE_CODES[
            tool_name
        ]
    except KeyError as exc:
        raise ValueError(
            "Unsupported tool for failure generation: "
            f"{tool_name}"
        ) from exc

    error_code = rng.choice(
        codes
    )

    return (
        error_code,
        ERROR_BY_CODE[
            error_code
        ],
    )


def success_result_for_tool(
    tool_name: str,
    arguments: dict[str, Any],
    values: dict[str, Any],
) -> tuple[
    dict[str, Any],
    str,
]:
    if tool_name == "device_list":
        return (
            {
                "devices": DEVICES,
            },
            "I found your devices.",
        )

    if tool_name == "device_get_status":
        return (
            {
                "device_id": (
                    arguments[
                        "device_id"
                    ]
                ),
                "online": True,
            },
            "Your device is online.",
        )

    if tool_name == "application_list":
        return (
            {
                "applications": (
                    APPLICATIONS
                ),
            },
            "I found your applications.",
        )

    if tool_name == (
        "computer_launch_application"
    ):
        return (
            {
                **arguments,
                "launched": True,
                "already_running": False,
            },
            (
                f"{values['application']['name']} "
                f"is open on your "
                f"{values['device']['name']}."
            ),
        )

    if tool_name == (
        "computer_close_application"
    ):
        return (
            {
                **arguments,
                "closed": True,
                "was_running": True,
            },
            (
                f"{values['application']['name']} "
                f"is closed on your "
                f"{values['device']['name']}."
            ),
        )

    if tool_name == (
        "computer_list_running_applications"
    ):
        return (
            {
                "device_id": (
                    arguments[
                        "device_id"
                    ]
                ),
                "applications": [
                    "steam",
                ],
            },
            "I found the running applications.",
        )

    if tool_name == (
        "computer_get_volume"
    ):
        return (
            {
                "device_id": (
                    arguments[
                        "device_id"
                    ]
                ),
                "volume_percent": (
                    values[
                        "volume"
                    ]
                ),
            },
            (
                f"Your volume is "
                f"{values['volume']}%."
            ),
        )

    if tool_name == (
        "computer_set_volume"
    ):
        return (
            {
                **arguments,
                "changed": True,
            },
            (
                f"Your volume is set to "
                f"{arguments['volume_percent']}%."
            ),
        )

    if tool_name == "timer_create":
        return (
            {
                "timer_id": (
                    values[
                        "timer_id"
                    ]
                ),
                "duration_seconds": (
                    arguments[
                        "duration_seconds"
                    ]
                ),
                "label": (
                    arguments.get(
                        "label"
                    )
                ),
            },
            "Your timer is set.",
        )

    if tool_name == "timer_list":
        return (
            {
                "timers": [
                    {
                        "timer_id": (
                            values[
                                "timer_id"
                            ]
                        ),
                        "duration_seconds": 300,
                    },
                ],
            },
            "I found your active timers.",
        )

    if tool_name == "timer_cancel":
        return (
            {
                "timer_id": (
                    arguments[
                        "timer_id"
                    ]
                ),
                "cancelled": True,
            },
            "Your timer is cancelled.",
        )

    raise ValueError(
        "Unsupported tool: "
        f"{tool_name}"
    )


def failure_response(
    error_code: str,
) -> str:
    responses = {
        "DEVICE_OFFLINE": (
            "The device is offline, "
            "so I couldn't complete that."
        ),
        "DEVICE_NOT_FOUND": (
            "I couldn't find that device."
        ),
        "APPLICATION_NOT_FOUND": (
            "I couldn't find that application."
        ),
        "TIMER_NOT_FOUND": (
            "I couldn't find that timer."
        ),
        "INVALID_ARGUMENT": (
            "I couldn't complete that "
            "because an argument was invalid."
        ),
        "PERMISSION_DENIED": (
            "I couldn't complete that "
            "because permission was denied."
        ),
        "TIMEOUT": (
            "The operation timed out."
        ),
        "TOOL_EXECUTION_FAILED": (
            "The operation failed to complete."
        ),
    }

    return responses[
        error_code
    ]


def make_success_followup_example(
    rng: random.Random,
    phrases: dict[
        str,
        list[str],
    ],
) -> dict[str, Any]:
    tool_name = rng.choice(
        TOOL_NAMES
    )

    values = sample_values(
        rng
    )

    arguments = arguments_for_tool(
        tool_name,
        values,
    )

    user_text = format_phrase(
        rng.choice(
            phrases[
                tool_name
            ]
        ),
        values,
    )

    (
        result,
        response_text,
    ) = success_result_for_tool(
        tool_name,
        arguments,
        values,
    )

    tool_result = {
        "call_id": random_call_id(
            rng
        ),
        "name": tool_name,
        "status": "success",
        "result": result,
        "error": None,
    }

    return {
        "category": (
            "success_followup"
        ),
        "messages": [
            {
                "role": "system",
                "content": (
                    build_runtime_context(
                        rng,
                        {
                            tool_name
                        },
                    )
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
            {
                "role": "assistant",
                "content": tool_call(
                    tool_name,
                    arguments,
                ),
            },
            {
                "role": "user",
                "content": (
                    "TOOL_RESULT\n"
                    + json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            },
            {
                "role": "assistant",
                "content": final(
                    response_text
                ),
            },
        ],
    }


def make_failure_followup_example(
    rng: random.Random,
    phrases: dict[
        str,
        list[str],
    ],
) -> dict[str, Any]:
    tool_name = rng.choice(
        TOOL_NAMES
    )

    values = sample_values(
        rng
    )

    arguments = arguments_for_tool(
        tool_name,
        values,
    )

    user_text = format_phrase(
        rng.choice(
            phrases[
                tool_name
            ]
        ),
        values,
    )

    (
        error_code,
        error_message,
    ) = choose_failure_for_tool(
        rng,
        tool_name,
    )

    tool_result = {
        "call_id": random_call_id(
            rng
        ),
        "name": tool_name,
        "status": "error",
        "result": None,
        "error": {
            "code": error_code,
            "message": error_message,
        },
    }

    return {
        "category": (
            "failure_followup"
        ),
        "messages": [
            {
                "role": "system",
                "content": (
                    build_runtime_context(
                        rng,
                        {
                            tool_name
                        },
                    )
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
            {
                "role": "assistant",
                "content": tool_call(
                    tool_name,
                    arguments,
                ),
            },
            {
                "role": "user",
                "content": (
                    "TOOL_RESULT\n"
                    + json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            },
            {
                "role": "assistant",
                "content": final(
                    failure_response(
                        error_code
                    )
                ),
            },
        ],
    }


def conversation_fingerprint(
    messages: list[
        dict[str, str]
    ],
) -> str:
    return json.dumps(
        messages,
        sort_keys=True,
        ensure_ascii=False,
    )


def category_targets(
    count: int,
) -> dict[str, int]:
    positive = round(
        count * 0.45
    )

    hard_negative = round(
        count * 0.15
    )

    ambiguity = round(
        count * 0.15
    )

    success = round(
        count * 0.15
    )

    failure = (
        count
        - positive
        - hard_negative
        - ambiguity
        - success
    )

    return {
        "positive_tool_call": positive,
        "hard_negative": hard_negative,
        "ambiguity": ambiguity,
        "success_followup": success,
        "failure_followup": failure,
    }


def generate_split(
    count: int,
    rng: random.Random,
    split: str,
) -> list[
    dict[str, Any]
]:
    if count <= 0:
        raise ValueError(
            "count must be positive"
        )

    if split == "train":
        phrases = TRAIN_PHRASES
        negative_phrases = (
            HARD_NEGATIVE_TRAIN
        )
        ambiguity_phrases = (
            AMBIGUITY_TRAIN_V021
        )

    elif split == "eval":
        phrases = EVAL_PHRASES
        negative_phrases = (
            HARD_NEGATIVE_EVAL
        )
        ambiguity_phrases = (
            AMBIGUITY_EVAL_V021
        )

    else:
        raise ValueError(
            "Unknown split: "
            f"{split}"
        )

    generators = {
        "positive_tool_call": (
            lambda: make_positive_example(
                rng,
                phrases,
            )
        ),
        "hard_negative": (
            lambda: make_hard_negative_example(
                rng,
                negative_phrases,
            )
        ),
        "ambiguity": (
            lambda: make_ambiguity_example_v021(
                rng,
                ambiguity_phrases,
            )
        ),
        "success_followup": (
            lambda: make_success_followup_example(
                rng,
                phrases,
            )
        ),
        "failure_followup": (
            lambda: make_failure_followup_example(
                rng,
                phrases,
            )
        ),
    }

    examples: list[
        dict[str, Any]
    ] = []

    seen: set[str] = set()

    targets = category_targets(
        count
    )

    for (
        category,
        target,
    ) in targets.items():
        added = 0
        attempts = 0

        max_attempts = (
            count * 100
        )

        while added < target:
            attempts += 1

            if attempts > max_attempts:
                raise RuntimeError(
                    "Unable to generate enough "
                    f"unique {category} examples: "
                    f"{added}/{target}"
                )

            example = (
                generators[
                    category
                ]()
            )

            key = (
                conversation_fingerprint(
                    example[
                        "messages"
                    ]
                )
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            examples.append(
                example
            )

            added += 1

    rng.shuffle(
        examples
    )

    for (
        index,
        example,
    ) in enumerate(
        examples
    ):
        example[
            "id"
        ] = (
            f"atlas_tools_{split}_"
            f"{index:06d}"
        )

    return examples


def write_jsonl(
    path: Path,
    examples: list[
        dict[str, Any]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for example in examples:
            output_file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

            output_file.write(
                "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-count",
        type=int,
        default=12000,
    )

    parser.add_argument(
        "--eval-count",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260829,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    train_rng = random.Random(
        args.seed
    )

    eval_rng = random.Random(
        args.seed + 1
    )

    train = generate_split(
        count=args.train_count,
        rng=train_rng,
        split="train",
    )

    evaluation = generate_split(
        count=args.eval_count,
        rng=eval_rng,
        split="eval",
    )

    train_fingerprints = {
        conversation_fingerprint(
            example[
                "messages"
            ]
        )
        for example in train
    }

    eval_fingerprints = {
        conversation_fingerprint(
            example[
                "messages"
            ]
        )
        for example in evaluation
    }

    leakage = (
        train_fingerprints
        & eval_fingerprints
    )

    if leakage:
        raise RuntimeError(
            "Train/eval conversation "
            "leakage detected."
        )

    train_path = (
        args.output_dir
        / "tools_train.jsonl"
    )

    eval_path = (
        args.output_dir
        / "tools_eval.jsonl"
    )

    write_jsonl(
        train_path,
        train,
    )

    write_jsonl(
        eval_path,
        evaluation,
    )

    print(
        "Atlas Tools Dataset v0.2.1"
    )

    print()

    print(
        "Training examples: "
        f"{len(train)}"
    )

    print(
        "Evaluation examples: "
        f"{len(evaluation)}"
    )

    print()

    for label, examples in [
        (
            "Training distribution",
            train,
        ),
        (
            "Evaluation distribution",
            evaluation,
        ),
    ]:
        print(
            f"{label}:"
        )

        for category in [
            "positive_tool_call",
            "hard_negative",
            "ambiguity",
            "success_followup",
            "failure_followup",
        ]:
            category_count = sum(
                1
                for example
                in examples
                if example[
                    "category"
                ] == category
            )

            print(
                f"  {category}: "
                f"{category_count}"
            )

        print()

    print(
        "Train/eval leakage: "
        f"{len(leakage)}"
    )

    print()

    print(
        f"Training output: {train_path}"
    )

    print(
        f"Evaluation output: {eval_path}"
    )


if __name__ == "__main__":
    main()
