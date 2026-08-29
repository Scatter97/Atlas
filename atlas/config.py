from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    ollama_base_url: str
    model_timeout_seconds: float
    max_actions_per_request: int
    trace_enabled: bool
    model_profile: str | None

    @classmethod
    def from_env(cls) -> "Config":
        selected_profile = os.getenv(
            "ATLAS_MODEL_PROFILE"
        )

        if selected_profile is not None:
            selected_profile = (
                selected_profile.strip()
                or None
            )

        return cls(
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://127.0.0.1:11434",
            ).rstrip("/"),
            model_timeout_seconds=float(
                os.getenv(
                    "ATLAS_MODEL_TIMEOUT_SECONDS",
                    "120",
                )
            ),
            max_actions_per_request=int(
                os.getenv(
                    "ATLAS_MAX_ACTIONS",
                    "12",
                )
            ),
            trace_enabled=(
                os.getenv(
                    "ATLAS_TRACE",
                    "1",
                ).strip().lower()
                not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
            ),
            model_profile=selected_profile,
        )
