from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DeviceRegistry:
    def __init__(
        self,
        devices: dict[str, dict[str, Any]],
    ) -> None:
        self._devices = devices

    @classmethod
    def from_file(
        cls,
        path: Path,
    ) -> "DeviceRegistry":
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        devices = data.get("devices")

        if not isinstance(devices, dict):
            raise ValueError(
                "Device registry must contain an object named 'devices'."
            )

        return cls(devices)

    def exists(
        self,
        device_id: str,
    ) -> bool:
        return device_id in self._devices

    def get(
        self,
        device_id: str,
    ) -> dict[str, Any] | None:
        device = self._devices.get(device_id)

        if device is None:
            return None

        return {
            "device_id": device_id,
            **device,
        }

    def list_public(
        self,
    ) -> list[dict[str, Any]]:
        return [
            {
                "device_id": device_id,
                **device,
            }
            for device_id, device in self._devices.items()
        ]


class ApplicationRegistry:
    def __init__(
        self,
        applications: dict[str, dict[str, Any]],
    ) -> None:
        self._applications = applications

    @classmethod
    def from_file(
        cls,
        path: Path,
    ) -> "ApplicationRegistry":
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        applications = data.get("applications")

        if not isinstance(applications, dict):
            raise ValueError(
                "Application registry must contain "
                "an object named 'applications'."
            )

        return cls(applications)

    def exists(
        self,
        application_id: str,
    ) -> bool:
        return application_id in self._applications

    def get(
        self,
        application_id: str,
    ) -> dict[str, Any] | None:
        application = self._applications.get(application_id)

        if application is None:
            return None

        return {
            "application_id": application_id,
            **application,
        }

    def list_public(
        self,
    ) -> list[dict[str, Any]]:
        return [
            {
                "application_id": application_id,
                **application,
            }
            for application_id, application in self._applications.items()
        ]
