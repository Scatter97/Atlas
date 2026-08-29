import json
import tempfile
import unittest
from pathlib import Path

from atlas.protocol import ToolCallAction, parse_action
from atlas.registries import ApplicationRegistry, DeviceRegistry
from atlas.tools import build_mock_tool_registry
from atlas.trace import AtlasTracer, redact_sensitive
from pydantic import ValidationError


class ProtocolTests(
    unittest.TestCase
):
    def test_valid_tool_call_parses(
        self,
    ) -> None:
        action = parse_action(
            '{"type":"tool_call",'
            '"name":"device_list",'
            '"arguments":{}}'
        )

        self.assertIsInstance(
            action,
            ToolCallAction,
        )

    def test_unknown_top_level_field_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ValidationError
        ):
            parse_action(
                '{"type":"final",'
                '"text":"Done.",'
                '"unexpected":true}'
            )


class ToolRegistryTests(
    unittest.TestCase
):
    def setUp(
        self,
    ) -> None:
        self.devices = DeviceRegistry(
            {
                "gaming_pc": {
                    "name": "Gaming PC",
                    "type": "computer",
                    "aliases": [
                        "desktop"
                    ],
                    "online": True,
                }
            }
        )

        self.applications = ApplicationRegistry(
            {
                "steam": {
                    "name": "Steam",
                    "aliases": [
                        "steam"
                    ],
                }
            }
        )

        self.tools = build_mock_tool_registry(
            devices=self.devices,
            applications=self.applications,
        )

    def test_unknown_tool_returns_structured_error(
        self,
    ) -> None:
        result = self.tools.execute(
            call_id="call_test",
            name="missing_tool",
            arguments={},
        )

        self.assertEqual(
            result.status,
            "error",
        )

        self.assertIsNotNone(
            result.error
        )

        self.assertEqual(
            result.error.code,
            "TOOL_NOT_FOUND",
        )

    def test_unknown_device_returns_structured_error(
        self,
    ) -> None:
        result = self.tools.execute(
            call_id="call_test",
            name="device_get_status",
            arguments={
                "device_id": "missing",
            },
        )

        self.assertEqual(
            result.status,
            "error",
        )

        self.assertIsNotNone(
            result.error
        )

        self.assertEqual(
            result.error.code,
            "DEVICE_NOT_FOUND",
        )

    def test_mock_application_launch_succeeds(
        self,
    ) -> None:
        result = self.tools.execute(
            call_id="call_test",
            name="computer_launch_application",
            arguments={
                "device_id": "gaming_pc",
                "application_id": "steam",
            },
        )

        self.assertEqual(
            result.status,
            "success",
        )

        self.assertEqual(
            result.result,
            {
                "launched": True,
                "already_running": False,
            },
        )

    def test_timer_ids_are_not_fixed(
        self,
    ) -> None:
        first = self.tools.execute(
            call_id="call_one",
            name="timer_create",
            arguments={
                "duration_seconds": 60,
                "label": "First",
            },
        )

        second = self.tools.execute(
            call_id="call_two",
            name="timer_create",
            arguments={
                "duration_seconds": 60,
                "label": "Second",
            },
        )

        self.assertEqual(
            first.status,
            "success",
        )

        self.assertEqual(
            second.status,
            "success",
        )

        self.assertNotEqual(
            first.result["timer_id"],
            second.result["timer_id"],
        )


class TraceTests(
    unittest.TestCase
):
    def test_sensitive_values_are_redacted(
        self,
    ) -> None:
        value = {
            "api_key": "secret-value",
            "nested": {
                "password": "password-value",
                "normal": "visible",
            },
            "items": [
                {
                    "access_token": "token-value",
                }
            ],
        }

        result = redact_sensitive(
            value
        )

        self.assertEqual(
            result["api_key"],
            "[REDACTED]",
        )

        self.assertEqual(
            result["nested"]["password"],
            "[REDACTED]",
        )

        self.assertEqual(
            result["nested"]["normal"],
            "visible",
        )

        self.assertEqual(
            result["items"][0]["access_token"],
            "[REDACTED]",
        )

    def test_tracer_writes_jsonl_record(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = (
                Path(temp_dir)
                / "atlas.jsonl"
            )

            tracer = AtlasTracer(
                terminal_enabled=False,
                log_path=log_path,
            )

            tracer.emit(
                "tool_call",
                {
                    "request_id": "req_test",
                    "call_id": "call_test",
                    "name": "device_list",
                    "arguments": {},
                },
            )

            lines = log_path.read_text(
                encoding="utf-8"
            ).splitlines()

            self.assertEqual(
                len(lines),
                1,
            )

            record = json.loads(
                lines[0]
            )

            self.assertEqual(
                record["event"],
                "tool_call",
            )

            self.assertEqual(
                record["data"]["call_id"],
                "call_test",
            )


if __name__ == "__main__":
    unittest.main()
