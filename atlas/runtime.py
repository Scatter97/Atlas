from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from atlas.model import (
    ModelResponse,
    OllamaBackend,
)
from atlas.trace import AtlasTracer


ResidencyMode = Literal[
    "persistent",
    "cached",
    "on_demand",
]

MemoryMode = Literal[
    "auto",
    "vram",
    "ram",
    "hybrid",
]


class RuntimeConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class ModelSpec(RuntimeConfigModel):
    model: str = Field(
        min_length=1
    )

    residency: ResidencyMode

    memory: MemoryMode = "auto"

    cache_duration: str | None = None

    num_gpu_layers: int | None = Field(
        default=None,
        ge=1,
    )

    think: bool = False

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    @model_validator(
        mode="after"
    )
    def validate_policy(
        self,
    ) -> "ModelSpec":
        if (
            self.residency == "cached"
            and not self.cache_duration
        ):
            raise ValueError(
                "cached models require "
                "cache_duration"
            )

        if (
            self.memory == "hybrid"
            and self.num_gpu_layers is None
        ):
            raise ValueError(
                "hybrid memory mode requires "
                "num_gpu_layers"
            )

        if (
            self.memory != "hybrid"
            and self.num_gpu_layers is not None
        ):
            raise ValueError(
                "num_gpu_layers is only valid "
                "for hybrid memory mode"
            )

        return self

    def effective_keep_alive(
        self,
    ) -> str | int:
        if self.residency == "persistent":
            return -1

        if self.residency == "on_demand":
            return 0

        if self.cache_duration is None:
            raise ValueError(
                "cached model has no cache_duration"
            )

        return self.cache_duration

    def ollama_options(
        self,
    ) -> dict[str, Any]:
        if self.memory == "ram":
            return {
                "num_gpu": 0,
            }

        if self.memory == "vram":
            return {
                "num_gpu": -1,
            }

        if self.memory == "hybrid":
            if self.num_gpu_layers is None:
                raise ValueError(
                    "hybrid memory mode requires "
                    "num_gpu_layers"
                )

            return {
                "num_gpu": self.num_gpu_layers,
            }

        return {}


class ModelProfile(RuntimeConfigModel):
    controller: ModelSpec
    tools: ModelSpec
    general: ModelSpec


class ModelProfilesFile(RuntimeConfigModel):
    active_profile: str = Field(
        min_length=1
    )

    profiles: dict[
        str,
        ModelProfile,
    ]


class ModelRuntimeManager:
    def __init__(
        self,
        profile_name: str,
        profile: ModelProfile,
        ollama_base_url: str,
        timeout_seconds: float,
        tracer: AtlasTracer,
    ) -> None:
        self.profile_name = profile_name
        self.profile = profile
        self.tracer = tracer

        self._specs: dict[
            str,
            ModelSpec,
        ] = {
            "controller": profile.controller,
            "tools": profile.tools,
            "general": profile.general,
        }

        self._backends = {
            role: OllamaBackend(
                base_url=ollama_base_url,
                model=spec.model,
                timeout_seconds=timeout_seconds,
                keep_alive=spec.effective_keep_alive(),
                options=spec.ollama_options(),
                think=spec.think,
                temperature=spec.temperature,
            )
            for role, spec in self._specs.items()
        }

    @classmethod
    def from_file(
        cls,
        path: Path,
        ollama_base_url: str,
        timeout_seconds: float,
        tracer: AtlasTracer,
        selected_profile: str | None = None,
    ) -> "ModelRuntimeManager":
        raw = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        profile_file = (
            ModelProfilesFile.model_validate(
                raw
            )
        )

        profile_name = (
            selected_profile
            if selected_profile is not None
            else profile_file.active_profile
        )

        try:
            profile = profile_file.profiles[
                profile_name
            ]
        except KeyError as exc:
            available = ", ".join(
                sorted(
                    profile_file.profiles
                )
            )

            raise ValueError(
                "Unknown Atlas model profile "
                f"'{profile_name}'. "
                f"Available profiles: {available}"
            ) from exc

        return cls(
            profile_name=profile_name,
            profile=profile,
            ollama_base_url=ollama_base_url,
            timeout_seconds=timeout_seconds,
            tracer=tracer,
        )

    def spec(
        self,
        role: str,
    ) -> ModelSpec:
        try:
            return self._specs[
                role
            ]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model role: {role}"
            ) from exc

    def generate_structured(
        self,
        role: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        backend = self._backend(
            role
        )

        spec = self.spec(
            role
        )

        self._trace_request(
            role=role,
            request_id=request_id,
            kind="structured",
        )

        response = backend.generate_structured(
            messages=messages,
            response_schema=response_schema,
        )

        self._trace_response(
            role=role,
            request_id=request_id,
            response=response,
            spec=spec,
        )

        return response.content

    def generate_text(
        self,
        role: str,
        messages: list[dict[str, str]],
        request_id: str,
    ) -> str:
        backend = self._backend(
            role
        )

        spec = self.spec(
            role
        )

        self._trace_request(
            role=role,
            request_id=request_id,
            kind="text",
        )

        response = backend.generate_text(
            messages=messages,
        )

        self._trace_response(
            role=role,
            request_id=request_id,
            response=response,
            spec=spec,
        )

        return response.content

    def preload_persistent(
        self,
    ) -> None:
        for role, spec in self._specs.items():
            if spec.residency != "persistent":
                continue

            self.tracer.emit(
                "model_preload",
                {
                    "role": role,
                    "model": spec.model,
                    "residency": spec.residency,
                    "memory": spec.memory,
                },
            )

            self._backend(
                role
            ).preload()

    def unload(
        self,
        role: str,
    ) -> None:
        spec = self.spec(
            role
        )

        self.tracer.emit(
            "model_unload",
            {
                "role": role,
                "model": spec.model,
            },
        )

        self._backend(
            role
        ).unload()

    def summary(
        self,
    ) -> list[dict[str, Any]]:
        summary: list[
            dict[str, Any]
        ] = []

        for role in (
            "controller",
            "tools",
            "general",
        ):
            spec = self.spec(
                role
            )

            summary.append(
                {
                    "role": role,
                    "model": spec.model,
                    "residency": spec.residency,
                    "memory": spec.memory,
                    "cache_duration": spec.cache_duration,
                    "num_gpu_layers": spec.num_gpu_layers,
                }
            )

        return summary

    def _backend(
        self,
        role: str,
    ) -> OllamaBackend:
        try:
            return self._backends[
                role
            ]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model role: {role}"
            ) from exc

    def _trace_request(
        self,
        role: str,
        request_id: str,
        kind: str,
    ) -> None:
        spec = self.spec(
            role
        )

        self.tracer.emit(
            "model_request",
            {
                "request_id": request_id,
                "role": role,
                "model": spec.model,
                "kind": kind,
                "residency": spec.residency,
                "memory": spec.memory,
            },
        )

    def _trace_response(
        self,
        role: str,
        request_id: str,
        response: ModelResponse,
        spec: ModelSpec,
    ) -> None:
        load_duration_ms = None

        if response.load_duration_ns is not None:
            load_duration_ms = round(
                response.load_duration_ns
                / 1_000_000,
                2,
            )

        total_duration_ms = None

        if response.total_duration_ns is not None:
            total_duration_ms = round(
                response.total_duration_ns
                / 1_000_000,
                2,
            )

        self.tracer.emit(
            "model_response",
            {
                "request_id": request_id,
                "role": role,
                "model": spec.model,
                "load_duration_ms": load_duration_ms,
                "total_duration_ms": total_duration_ms,
                "prompt_eval_count": (
                    response.prompt_eval_count
                ),
                "eval_count": (
                    response.eval_count
                ),
            },
        )
