from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ModelBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelResponse:
    content: str
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


class ModelBackend(ABC):
    @abstractmethod
    def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def generate_text(
        self,
        messages: list[dict[str, str]],
    ) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def preload(
        self,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def unload(
        self,
    ) -> None:
        raise NotImplementedError


class OllamaBackend(ModelBackend):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        keep_alive: str | int = "5m",
        options: dict[str, Any] | None = None,
        think: bool = False,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self.options = dict(
            options or {}
        )
        self.think = think
        self.temperature = temperature

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> ModelResponse:
        payload = self._base_payload(
            messages=messages,
        )

        payload["format"] = response_schema

        body = self._request(
            payload
        )

        return self._parse_generation_response(
            body
        )

    def generate_text(
        self,
        messages: list[dict[str, str]],
    ) -> ModelResponse:
        payload = self._base_payload(
            messages=messages,
        )

        body = self._request(
            payload
        )

        return self._parse_generation_response(
            body
        )

    def preload(
        self,
    ) -> None:
        payload = {
            "model": self.model,
            "messages": [],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": self._request_options(),
        }

        self._request(
            payload
        )

    def unload(
        self,
    ) -> None:
        payload = {
            "model": self.model,
            "messages": [],
            "stream": False,
            "keep_alive": 0,
            "options": self._request_options(),
        }

        self._request(
            payload
        )

    def _base_payload(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": self._request_options(),
        }

    def _request_options(
        self,
    ) -> dict[str, Any]:
        options = {
            "temperature": self.temperature,
        }

        options.update(
            self.options
        )

        return options

    def _request(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(
                payload
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_text = (
                    response.read().decode(
                        "utf-8"
                    )
                )

                body = json.loads(
                    response_text
                )

        except urllib.error.HTTPError as exc:
            details = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise ModelBackendError(
                f"Ollama HTTP {exc.code}: {details}"
            ) from exc

        except urllib.error.URLError as exc:
            raise ModelBackendError(
                "Could not connect to Ollama at "
                f"{self.base_url}: {exc.reason}"
            ) from exc

        except (
            TimeoutError,
            socket.timeout,
        ) as exc:
            raise ModelBackendError(
                "Ollama request timed out."
            ) from exc

        except json.JSONDecodeError as exc:
            raise ModelBackendError(
                "Ollama returned invalid response JSON."
            ) from exc

        if not isinstance(
            body,
            dict,
        ):
            raise ModelBackendError(
                "Ollama returned an invalid response object."
            )

        return body

    def _parse_generation_response(
        self,
        body: dict[str, Any],
    ) -> ModelResponse:
        try:
            content = body[
                "message"
            ][
                "content"
            ]
        except (
            KeyError,
            TypeError,
        ) as exc:
            raise ModelBackendError(
                "Ollama response did not contain "
                "message.content."
            ) from exc

        if not isinstance(
            content,
            str,
        ):
            raise ModelBackendError(
                "Ollama message.content was not "
                "a string."
            )

        return ModelResponse(
            content=content,
            total_duration_ns=self._optional_int(
                body.get(
                    "total_duration"
                )
            ),
            load_duration_ns=self._optional_int(
                body.get(
                    "load_duration"
                )
            ),
            prompt_eval_count=self._optional_int(
                body.get(
                    "prompt_eval_count"
                )
            ),
            eval_count=self._optional_int(
                body.get(
                    "eval_count"
                )
            ),
        )

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> int | None:
        if isinstance(
            value,
            bool,
        ):
            return None

        if isinstance(
            value,
            int,
        ):
            return value

        return None
