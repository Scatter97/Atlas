from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    ollama_base_url: str
    ollama_model: str
    model_timeout_seconds: float
    max_actions_per_request: int
    trace_enabled: bool


    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://127.0.0.1:11434",
            ).rstrip("/"),
            ollama_model=os.getenv(
                "OLLAMA_MODEL",
                "qwen3.5:4b",
            ),
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
        )
