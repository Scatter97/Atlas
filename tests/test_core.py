import json
import tempfile
import unittest
from pathlib import Path

from atlas.protocol import (
    RouteAction,
    ToolCallAction,
    parse_action,
    parse_controller_action,
    parse_tool_specialist_action,
)
from atlas.registries import ApplicationRegistry, DeviceRegistry
from atlas.tools import build_mock_tool_registry
from atlas.trace import AtlasTracer, redact_sensitive
from atlas.runtime import ModelSpec
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


class MultiModelProtocolTests(
    unittest.TestCase
):
    def test_controller_route_parses(
        self,
    ) -> None:
        action = parse_controller_action(
            '{"type":"route",'
            '"target":"atlas_tools",'
            '"request":"Open Steam."}'
        )

        self.assertIsInstance(
            action,
            RouteAction,
        )

        self.assertEqual(
            action.target,
            "atlas_tools",
        )

    def test_tool_specialist_rejects_delegate(
        self,
    ) -> None:
        with self.assertRaises(
            ValidationError
        ):
            parse_tool_specialist_action(
                '{"type":"delegate",'
                '"target":"general_ai",'
                '"request":"Explain black holes."}'
            )

    def test_ram_mode_forces_cpu(
        self,
    ) -> None:
        spec = ModelSpec(
            model="test",
            residency="persistent",
            memory="ram",
        )

        self.assertEqual(
            spec.ollama_options(),
            {
                "num_gpu": 0,
            },
        )

    def test_vram_mode_requests_gpu_offload(
        self,
    ) -> None:
        spec = ModelSpec(
            model="test",
            residency="persistent",
            memory="vram",
        )

        self.assertEqual(
            spec.ollama_options(),
            {
                "num_gpu": -1,
            },
        )

    def test_on_demand_unloads_after_request(
        self,
        ) -> None:
        spec = ModelSpec(
            model="test",
            residency="on_demand",
            memory="auto",
        )

        self.assertEqual(
            spec.effective_keep_alive(),
            0,
        )

    def test_hybrid_requires_gpu_layer_count(
        self,
    ) -> None:
        with self.assertRaises(
            ValidationError
        ):
            ModelSpec(
                model="test",
                residency="persistent",
                memory="hybrid",
            )


class ExpandedToolTests(
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

        self.applications = (
            ApplicationRegistry(
                {
                    "steam": {
                        "name": "Steam",
                        "aliases": [
                            "steam"
                        ],
                    },
                    "firefox": {
                        "name": "Firefox",
                        "aliases": [
                            "firefox"
                        ],
                    },
                }
            )
        )

        self.tools = (
            build_mock_tool_registry(
                devices=self.devices,
                applications=(
                    self.applications
                ),
            )
        )

    def test_application_list(
        self,
    ) -> None:
        result = self.tools.execute(
            call_id="call_test",
            name="application_list",
            arguments={},
        )

        self.assertEqual(
            result.status,
            "success",
        )

        self.assertEqual(
            len(
                result.result[
                    "applications"
                ]
            ),
            2,
        )

    def test_launch_is_stateful(
        self,
    ) -> None:
        first = self.tools.execute(
            call_id="call_one",
            name=(
                "computer_launch_application"
            ),
            arguments={
                "device_id": "gaming_pc",
                "application_id": "steam",
            },
        )

        second = self.tools.execute(
            call_id="call_two",
            name=(
                "computer_launch_application"
            ),
            arguments={
                "device_id": "gaming_pc",
                "application_id": "steam",
            },
        )

        self.assertFalse(
            first.result[
                "already_running"
            ]
        )

        self.assertTrue(
            second.result[
                "already_running"
            ]
        )

    def test_close_application(
        self,
    ) -> None:
        self.tools.execute(
            call_id="call_launch",
            name=(
                "computer_launch_application"
            ),
            arguments={
                "device_id": "gaming_pc",
                "application_id": "steam",
            },
        )

        result = self.tools.execute(
            call_id="call_close",
            name=(
                "computer_close_application"
            ),
            arguments={
                "device_id": "gaming_pc",
                "application_id": "steam",
            },
        )

        self.assertEqual(
            result.status,
            "success",
        )

        self.assertTrue(
            result.result[
                "closed"
            ]
        )

    def test_running_application_list(
        self,
    ) -> None:
        self.tools.execute(
            call_id="call_launch",
            name=(
                "computer_launch_application"
            ),
            arguments={
                "device_id": "gaming_pc",
                "application_id": "firefox",
            },
        )

        result = self.tools.execute(
            call_id="call_list",
            name=(
                "computer_list_"
                "running_applications"
            ),
            arguments={
                "device_id": "gaming_pc",
            },
        )

        self.assertEqual(
            result.result[
                "application_ids"
            ],
            [
                "firefox"
            ],
        )

    def test_volume_is_stateful(
        self,
    ) -> None:
        set_result = (
            self.tools.execute(
                call_id="call_set",
                name=(
                    "computer_set_volume"
                ),
                arguments={
                    "device_id": (
                        "gaming_pc"
                    ),
                    "volume_percent": 35,
                },
            )
        )

        get_result = (
            self.tools.execute(
                call_id="call_get",
                name=(
                    "computer_get_volume"
                ),
                arguments={
                    "device_id": (
                        "gaming_pc"
                    ),
                },
            )
        )

        self.assertEqual(
            set_result.result[
                "volume_percent"
            ],
            35,
        )

        self.assertEqual(
            get_result.result[
                "volume_percent"
            ],
            35,
        )

    def test_timer_list_and_cancel(
        self,
    ) -> None:
        created = (
            self.tools.execute(
                call_id="call_create",
                name="timer_create",
                arguments={
                    "duration_seconds": 60,
                    "label": "Test",
                },
            )
        )

        timer_id = (
            created.result[
                "timer_id"
            ]
        )

        listed = self.tools.execute(
            call_id="call_list",
            name="timer_list",
            arguments={},
        )

        self.assertEqual(
            len(
                listed.result[
                    "timers"
                ]
            ),
            1,
        )

        cancelled = (
            self.tools.execute(
                call_id="call_cancel",
                name="timer_cancel",
                arguments={
                    "timer_id": timer_id,
                },
            )
        )

        self.assertTrue(
            cancelled.result[
                "cancelled"
            ]
        )

        listed_after = (
            self.tools.execute(
                call_id=(
                    "call_list_after"
                ),
                name="timer_list",
                arguments={},
            )
        )

        self.assertEqual(
            listed_after.result[
                "timers"
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main()
