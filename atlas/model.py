from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from atlas.protocol import Action, parse_action


class ModelBackendError(RuntimeError):
    pass


class ModelBackend(ABC):
    @abstractmethod
    def generate_action(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> Action:
        raise NotImplementedError


class OllamaBackend(ModelBackend):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate_action(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> Action:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": response_schema,
            "options": {
                "temperature": 0,
            },
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
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
                response_text = response.read().decode("utf-8")
                body = json.loads(response_text)

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
                f"Could not connect to Ollama at "
                f"{self.base_url}: {exc.reason}"
            ) from exc

        except (TimeoutError, socket.timeout) as exc:
            raise ModelBackendError(
                "Ollama request timed out."
            ) from exc

        except json.JSONDecodeError as exc:
            raise ModelBackendError(
                "Ollama returned invalid response JSON."
            ) from exc

        try:
            content = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ModelBackendError(
                "Ollama response did not contain message.content."
            ) from exc

        if not isinstance(content, str):
            raise ModelBackendError(
                "Ollama message.content was not a string."
            )

        try:
            return parse_action(content)

        except ValidationError as exc:
            raise ModelBackendError(
                f"Model returned an invalid Atlas action: {exc}"
            ) from exc
