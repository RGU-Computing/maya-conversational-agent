"""Deployment checks use real adapters with simulated transport, never patient data."""

import importlib.util
import io
import json
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from maya.cli import check_tools
from maya.framework.conversation.personalization import SessionConfig
from maya.tests.test_phyxio_sessions import configured_service, history_connection, history_reader, history_row


class ToolCheckTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(TemporaryDirectory()))
        self.session = SessionConfig(user_id="participant-001")
        self.session_path = self.folder / "session.json"
        self.session_path.write_text(self.session.model_dump_json())
        self.enterContext(patch.object(check_tools, "load_dotenv"))
        self.enterContext(patch.dict(os.environ, {
            "MAYA_SESSION_CONFIG": str(self.session_path), "MAYA_ENABLE_PHYXIO_IO": "1",
            "MAYA_EXERCISE_SESSION_CONFIG": str(self.folder / "exercise.json"),
            "AZURE_SPEECH_KEY": "synthetic-secret", "AZURE_SPEECH_REGION": "synthetic-region",
            "OPENAI_WHISPER_API_KEY": "synthetic-secret"}))
        configured_service(self.folder)

    def run_check(self, *args):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            code = check_tools.main(args)
        self.assertNotIn("synthetic-secret", output.getvalue())
        return code, json.loads(output.getvalue())

    def native(self, connected=True):
        from maya.framework.integrations.phyxio import bridge
        from maya.agents.phyxio_exercise_agent import phyxio_sessions

        reader = history_reader([history_row()])
        connection = history_connection(self.session, reader)
        connection._phyxio.ready = Mock()
        connection._phyxio.ready.wait.return_value = connected
        connection.stop = Mock()
        self.enterContext(patch.object(bridge, "get_phyxio_service", return_value=connection))
        self.enterContext(patch.object(phyxio_sessions, "get_phyxio_service", return_value=connection))
        self.enterContext(patch.object(check_tools.importlib, "import_module", return_value=SimpleNamespace(stringVersion=lambda: "3.7.11")))
        return connection, reader

    def test_configuration_check_does_not_connect_or_fetch(self):
        connection, reader = self.native()
        code, report = self.run_check("--only", "exercises")
        self.assertEqual(code, 0)
        self.assertEqual(report["mode"], "configuration_only")
        self.assertEqual(report["checks"]["exercises"]["status"], "configured")
        reader.getSessionList.assert_not_called()
        connection._phyxio.ready.wait.assert_not_called()

    def test_live_check_uses_actual_provider_and_closes_connection(self):
        connection, reader = self.native()
        code, report = self.run_check("--only", "exercises", "--fetch", "--wait-seconds", "3")
        self.assertEqual(code, 0)
        self.assertEqual(report["checks"]["exercises"]["status"], "ok")
        self.assertFalse(report["checks"]["exercises"]["synthetic"])
        self.assertEqual(report["user_id"], self.session.user_id)
        self.assertGreater(reader.getSessionList.call_count, 0)
        connection._phyxio.ready.wait.assert_called_once_with(3)
        connection.stop.assert_called_once()

    def test_timeout_and_identity_mismatch_do_not_fetch(self):
        connection, reader = self.native(connected=False)
        for connected, user_id in [(False, self.session.user_id), (True, "different-participant")]:
            connection._phyxio.ready.wait.return_value = connected
            connection.session = SessionConfig(user_id=user_id)
            code, report = self.run_check("--only", "exercises", "--fetch")
            self.assertEqual(code, 1)
            reader.getSessionList.assert_not_called()
        self.assertEqual(connection.stop.call_count, 2)

    def test_dummy_provider_is_not_a_live_configuration_pass(self):
        path = Path(__file__).resolve().parents[1] / "agents/phyxio_exercise_agent/session_config.example.json"
        with patch.dict(os.environ, {"MAYA_EXERCISE_SESSION_CONFIG": str(path)}):
            code, report = self.run_check("--only", "exercises")
        self.assertEqual(code, 1)
        self.assertEqual(report["checks"]["exercises"]["provider"], "dummy")

    def test_missing_identity_fails_before_any_tool(self):
        self.session_path.write_text('{}')
        with patch.object(check_tools, "check_exercises") as exercise, patch.object(check_tools, "check_biomarkers") as biomarker:
            code, report = self.run_check()
        self.assertEqual(code, 1)
        self.assertEqual(list(report["checks"]), ["session"])
        exercise.assert_not_called()
        biomarker.assert_not_called()

    def test_incompatible_ice_and_missing_modules_are_actionable(self):
        with patch.object(check_tools.importlib, "import_module", return_value=SimpleNamespace(stringVersion=lambda: "3.8.3")):
            code, report = self.run_check("--only", "exercises")
        self.assertEqual(code, 1)
        self.assertIn("3.7", report["checks"]["exercises"]["detail"])
        with patch.object(check_tools.importlib, "import_module", side_effect=ModuleNotFoundError("synthetic-secret", name="Ice")):
            code, report = self.run_check("--only", "exercises")
        self.assertEqual(report["checks"]["exercises"]["missing_module"], "Ice")

    def test_one_failed_provider_does_not_hide_other_result_or_leak_error(self):
        with patch.object(check_tools, "check_exercises", side_effect=RuntimeError("synthetic-secret")), \
                patch.object(check_tools, "check_biomarkers", return_value={"status": "configured"}):
            code, report = self.run_check()
        self.assertEqual(code, 1)
        self.assertEqual(report["checks"]["biomarkers"]["status"], "configured")

    @unittest.skipUnless(importlib.util.find_spec("libfhir"), "Requires UCLM libfhir.")
    def test_fhir_mapping_real_read_and_http_error_are_distinct(self):
        from maya.tests.test_fhir_biomarkers import configured_service as fhir_service, observation, response

        service = fhir_service(self.folder)
        self.addCleanup(service.provider.client._session.close)
        self.enterContext(patch.dict(os.environ, {"MAYA_BIOMARKER_CONFIG": str(self.folder / "biomarkers.json")}))
        self.enterContext(patch("maya.framework.integrations.biomarkers.load_biomarker_service", return_value=service))
        with patch.object(service.provider.client._session, "get", return_value=response([observation()])) as get:
            code, report = self.run_check("--only", "biomarkers")
            self.assertEqual(report["checks"]["biomarkers"]["status"], "configured")
            get.assert_not_called()
            get.side_effect = lambda *args, **kwargs: response([observation()] if kwargs["json"]["resource_type"] == "blood-pressure" else [])
            code, report = self.run_check("--only", "biomarkers", "--fetch")
            self.assertEqual(code, 0)
            self.assertEqual(report["checks"]["biomarkers"]["observation_count"], 1)
            self.assertNotIn("134", json.dumps(report))
            get.side_effect = None
            get.return_value = response(status=403)
            code, report = self.run_check("--only", "biomarkers", "--fetch")
            self.assertEqual(code, 1)
            self.assertEqual(report["checks"]["biomarkers"]["http_status"], 403)
            get.reset_mock()
            service.config.patient_map.clear()
            code, report = self.run_check("--only", "biomarkers", "--fetch")
            self.assertEqual(code, 1)
            self.assertIn("patient_map", report["checks"]["biomarkers"]["detail"])
            get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
