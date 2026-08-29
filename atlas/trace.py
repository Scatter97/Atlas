from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REDACTED = "[REDACTED]"

_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "token",
    "api_key",
    "apikey",
    "secret",
    "authorization",
    "credential",
)


def _contains_sensitive_key(
    key: str,
) -> bool:
    normalized = key.lower()

    return any(
        sensitive_part in normalized
        for sensitive_part in _SENSITIVE_KEY_PARTS
    )


def redact_sensitive(
    value: Any,
) -> Any:
    if isinstance(
        value,
        dict,
    ):
        redacted: dict[str, Any] = {}

        for key, item in value.items():
            key_text = str(key)

            if _contains_sensitive_key(
                key_text
            ):
                redacted[key_text] = _REDACTED
            else:
                redacted[key_text] = redact_sensitive(
                    item
                )

        return redacted

    if isinstance(
        value,
        list,
    ):
        return [
            redact_sensitive(item)
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return [
            redact_sensitive(item)
            for item in value
        ]

    return value


def default_log_path() -> Path:
    local_app_data = os.getenv(
        "LOCALAPPDATA"
    )

    if local_app_data:
        return (
            Path(local_app_data)
            / "Atlas"
            / "logs"
            / "atlas.jsonl"
        )

    return (
        Path.home()
        / ".atlas"
        / "logs"
        / "atlas.jsonl"
    )


class AtlasTracer:
    def __init__(
        self,
        terminal_enabled: bool = True,
        log_path: Path | None = None,
    ) -> None:
        self.terminal_enabled = terminal_enabled
        self.log_path = (
            log_path
            if log_path is not None
            else default_log_path()
        )

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def emit(
        self,
        event: str,
        data: dict[str, Any],
    ) -> None:
        safe_data = redact_sensitive(
            data
        )

        record = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "event": event,
            "data": safe_data,
        }

        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )
            log_file.write("\n")

        if self.terminal_enabled:
            self._print_event(
                event,
                safe_data,
            )

    def _print_event(
        self,
        event: str,
        data: dict[str, Any],
    ) -> None:
        if event == "model_action":
            action_type = data.get(
                "type",
                "unknown",
            )

            if action_type == "tool_call":
                print(
                    "[TRACE] MODEL -> TOOL_CALL "
                    f"{data.get('name')}"
                )
                print(
                    "[TRACE] arguments = "
                    + json.dumps(
                        data.get(
                            "arguments",
                            {},
                        ),
                        ensure_ascii=False,
                    )
                )
                return

            print(
                "[TRACE] MODEL -> "
                f"{str(action_type).upper()}"
            )
            return

        if event == "tool_call":
            print(
                "[TRACE] CORE -> EXECUTE "
                f"{data.get('name')} "
                f"({data.get('call_id')})"
            )
            return

        if event == "tool_result":
            print(
                "[TRACE] TOOL -> "
                f"{str(data.get('status')).upper()} "
                f"{data.get('name')} "
                f"({data.get('call_id')})"
            )

            if data.get(
                "status"
            ) == "success":
                print(
                    "[TRACE] result = "
                    + json.dumps(
                        data.get(
                            "result"
                        ),
                        ensure_ascii=False,
                    )
                )
            else:
                print(
                    "[TRACE] error = "
                    + json.dumps(
                        data.get(
                            "error"
                        ),
                        ensure_ascii=False,
                    )
                )

            return

        if event == "request_complete":
            print(
                "[TRACE] CORE -> COMPLETE "
                f"{str(data.get('type')).upper()}"
            )
            return

        if event == "action_limit_reached":
            print(
                "[TRACE] CORE -> ACTION LIMIT REACHED"
            )


