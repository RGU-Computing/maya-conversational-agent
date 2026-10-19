"""Native Phyxio history contract; synthetic records, no ICE or patient data."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from maya.agents.phyxio_exercise_agent import phyxio_sessions
from maya.agents.phyxio_exercise_agent.sessions import load_session_service
from maya.framework.conversation.personalization import SessionConfig, load_session


EXAMPLE = Path(__file__).resolve().parents[1] / "agents/phyxio_exercise_agent/session_config.phyxio.example.json"


def history_row(day=12, repetitions=20, **updates):
    return {"date": f"2026-10-{day:02d}T09:00:00+00:00", "routine_id": 48,
            "results": [{"exercise_id": 228, "status": "completed", "goal_iter": 20,
                         "done_iter": repetitions, "limit_time": 120, "elapsed_time": 69,
                         "report": "Synthetic source report."}], **updates}


def history_reader(rows=None, page_size=2, first_page=0):
    rows = rows if rows is not None else [history_row(day, count) for day, count in [(1, 10), (3, 12), (5, 14), (7, 16), (9, 18), (12, 20)]]
    reader = Mock(spec=["getRoutineList", "getSessionList"])
    reader.getRoutineList.return_value = json.dumps({"routines": {"48": {"name": "Assigned programme"}},
        "exercises": {"228": {"name": "Seated march"}}})
    reader.getSessionList.side_effect = lambda page: json.dumps({"page": page,
        "sessions": rows[(page-first_page)*page_size:(page-first_page+1)*page_size]})
    return reader


def partial_history_rows():
    """Synthetic sessions using the mirror's observed execution-status contract."""
    rows = [history_row(day) for day in (1, 3, 12)]
    for row, repetitions in zip(rows, (8, 0, 12)):
        row["results"][0].update(status="executed", goal_iter=20, done_iter=repetitions, limit_time=0)
        row["results"].append({"exercise_id": 229, "status": "not-executed", "goal_iter": 10,
                               "done_iter": 0, "limit_time": 60, "elapsed_time": 0, "report": ""})
    rows[0]["results"][1].update(status="executed", done_iter=6, elapsed_time=40)
    rows[1]["results"][0].update(status="skipped", elapsed_time=0, report="")
    return rows


def configured_service(folder, *, first_page=0):
    settings = json.loads(EXAMPLE.read_text())
    settings["first_page"] = first_page
    path = Path(folder) / "exercise.json"
    path.write_text(json.dumps(settings))
    with patch.dict(os.environ, {"MAYA_EXERCISE_SESSION_CONFIG": str(path)}):
        return load_session_service()


def history_connection(session, reader):
    """Actual participant/connection guard with only the ICE transport substituted."""
    name = "maya.agents.phyxio_exercise_agent._offline_history"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "framework/integrations/phyxio/service.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module, "Ice": Mock(),
            "phyxio": SimpleNamespace(connect=SimpleNamespace(PhyxioConnector=object))}):
        spec.loader.exec_module(module)
    service = object.__new__(module.PhyxioService)
    service.session = session
    proxy = Mock()
    proxy.ice_invocationTimeout.return_value = reader
    service._phyxio = SimpleNamespace(_admin_servant=SimpleNamespace(
        _clients={"one": SimpleNamespace(proxy=proxy)}))
    return service


class PhyxioSessionTests(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(TemporaryDirectory())
        self.service = configured_service(folder)
        self.enterContext(patch.dict(os.environ, {"MAYA_EXERCISE_SESSION_CONFIG": str(Path(folder) / "exercise.json")}))
        self.reader = history_reader()
        self.transport = Mock(spec=["session_history_reader"])
        self.transport.session_history_reader.return_value = self.reader
        self.connect = self.enterContext(patch.object(phyxio_sessions, "get_phyxio_service", return_value=self.transport))

    def latest(self):
        return self.service.latest("participant-001")

    def test_native_pages_are_sorted_and_five_comparisons_are_exact(self):
        result = self.latest()
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["synthetic"])
        self.assertEqual(len(result["previous"]), 5)
        self.assertEqual(result["comparison"]["228:done_iter"]["previous_mean"], 14)
        self.assertEqual(result["comparison"]["228:done_iter"]["difference"], 6)
        self.transport.session_history_reader.assert_called_once_with("participant-001", 5000)
        self.assertEqual(self.reader.getSessionList.call_args_list, [call(0), call(1), call(2), call(3)])

    def test_first_page_is_configurable_and_shared_with_history_listing(self):
        from maya.agents.phyxio_exercise_agent import agent

        with TemporaryDirectory() as folder:
            for first_page in (0, 1):
                with self.subTest(first_page=first_page):
                    service = configured_service(folder, first_page=first_page)
                    reader = history_reader([history_row()], first_page=first_page)
                    self.transport.session_history_reader.return_value = reader
                    self.assertEqual(service.latest("participant-001")["status"], "ok")
                    reader.getSessionList.assert_any_call(first_page)
                    with patch.dict(os.environ, {"MAYA_EXERCISE_SESSION_CONFIG": str(Path(folder) / "exercise.json")}):
                        for value in ("", None, "invalid", -1, first_page, str(first_page)):
                            self.assertEqual(agent._parse_page(value, service.config.first_page), first_page)
                        self.assertEqual(agent._parse_page(first_page + 2, service.config.first_page), first_page + 2)

    def test_partial_routines_preserve_statuses_without_zero_filled_comparisons(self):
        rows = partial_history_rows()
        self.transport.session_history_reader.return_value = history_reader(rows)
        result = self.latest()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["latest"]["exercise_results"], rows[-1]["results"])
        self.assertEqual(set(result["latest"]["metrics"]), {"228:done_iter", "228:elapsed_time"})
        self.assertEqual(result["comparison"]["228:done_iter"]["previous_count"], 1)
        self.assertEqual(result["comparison"]["228:done_iter"]["previous_mean"], 8)
        self.assertEqual(result["comparison"]["228:done_iter"]["difference"], 4)
        self.assertEqual(len(result["previous"]), 2)

    def test_latest_skipped_only_record_is_not_replaced_by_older_performance(self):
        rows = partial_history_rows()
        rows[1]["date"] = "2026-10-13T09:00:00Z"
        self.transport.session_history_reader.return_value = history_reader(rows)
        result = self.latest()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["latest"]["recorded_at"], rows[1]["date"])
        self.assertEqual(result["latest"]["metrics"], {})
        self.assertEqual(result["comparison"], {})

    def test_placeholder_targets_do_not_remove_real_zero_measurements(self):
        row = history_row(repetitions=0)
        row["results"][0].update(goal_iter=0, limit_time=0, elapsed_time=0, status="executed")
        self.transport.session_history_reader.return_value = history_reader([row])
        latest = self.latest()["latest"]
        for metric in latest["metrics"].values():
            self.assertEqual(metric["value"], 0)
            self.assertNotIn("target", metric)
            self.assertNotIn("target_kind", metric)
        self.assertEqual(latest["exercise_results"][0]["limit_time"], 0)
        self.service.provider.config.metrics["done_iter"].unset_target_values = []
        self.assertEqual(self.latest()["latest"]["metrics"]["228:done_iter"]["target"], 0)

    def test_status_vocabulary_comes_from_configuration(self):
        row = history_row()
        row["results"][0]["status"] = "partner-recorded"
        self.transport.session_history_reader.return_value = history_reader([row])
        self.assertEqual(self.latest()["status"], "no_data")
        self.service.config.accepted_statuses = ["partner-recorded"]
        self.service.provider.config.metric_statuses = ["partner-recorded"]
        self.assertEqual(self.latest()["latest"]["metrics"]["228:done_iter"]["value"], 20)

    def test_history_listing_uses_recorded_statuses_without_claiming_completion(self):
        from maya.agents.phyxio_exercise_agent import agent

        row = partial_history_rows()[-1]
        self.reader.getSessionList.side_effect = None
        self.reader.getSessionList.return_value = {"page": 0, "sessions": [row]}
        text = agent.get_exercise_history.invoke({}, config={"configurable": {"user_id": "participant-001"}})
        self.transport.session_history_reader.assert_called_once_with("participant-001", 5000)
        self.reader.getSessionList.assert_called_once_with(0)
        self.assertIn("1 executed, 1 not-executed", text)
        self.assertNotIn("completed", text)

    def test_source_fields_preserved_without_inventing_session_identity_or_finish_time(self):
        latest = self.latest()["latest"]
        self.assertEqual(latest["recorded_at"], "2026-10-12T09:00:00Z")
        self.assertEqual(latest["routine_name"], "Assigned programme")
        self.assertEqual(latest["exercise_results"][0]["report"], "Synthetic source report.")
        self.assertEqual(latest["metrics"]["228:elapsed_time"]["target_kind"], "maximum")
        for field in ("session_id", "ended_at", "user_id", "status"):
            self.assertNotIn(field, latest)

    def test_session_rejection_stops_both_history_tools_before_retrieval(self):
        from maya.agents.phyxio_exercise_agent import agent

        self.transport.session_history_reader.side_effect = ValueError("Participant mismatch")
        self.assertEqual(self.latest()["status"], "unavailable")
        response = agent.get_exercise_history.invoke({}, config={"configurable": {"user_id": "participant-001"}})
        self.assertIn("couldn't fetch", response)
        self.reader.getSessionList.assert_not_called()
        self.reader.getRoutineList.assert_not_called()
        self.assertNotIn("user_id", agent.get_exercise_history.args)

    def test_changed_targets_or_exercises_are_not_compared(self):
        rows = [history_row(1), history_row(3), history_row(12)]
        rows[0]["results"][0]["goal_iter"] = 30
        rows[1]["results"][0]["exercise_id"] = 229
        self.transport.session_history_reader.return_value = history_reader(rows)
        self.assertEqual(self.latest()["previous"], [])

    def test_missing_values_are_not_zero_filled_and_zero_is_preserved(self):
        row = history_row(repetitions=0)
        row["results"][0]["elapsed_time"] = None
        self.transport.session_history_reader.return_value = history_reader([row])
        result = self.latest()
        self.assertEqual(result["latest"]["metrics"]["228:done_iter"]["value"], 0)
        self.assertNotIn("228:elapsed_time", result["latest"]["metrics"])
        self.assertEqual(result["comparison"], {})

    def test_finished_statuses_and_future_dates_control_eligibility(self):
        rows = [history_row(1), history_row(3), history_row(date="2099-01-01T00:00:00Z")]
        rows[1]["results"][0]["status"] = "in_progress"
        self.transport.session_history_reader.return_value = history_reader(rows)
        self.assertEqual(self.latest()["latest"]["recorded_at"], "2026-10-01T09:00:00Z")
        rows[1]["results"][0]["status"] = "interrupted"
        self.assertEqual(self.latest()["latest"]["exercise_results"][0]["status"], "interrupted")

    def test_empty_history_and_missing_names(self):
        self.transport.session_history_reader.return_value = history_reader([])
        self.assertEqual(self.latest()["status"], "no_data")
        self.transport.session_history_reader.return_value = self.reader
        for names in (TimeoutError(), "not JSON", {"routines": None}, {"error": "Unavailable"}):
            with self.subTest(names=names):
                self.reader.getRoutineList.side_effect = names if isinstance(names, Exception) else None
                self.reader.getRoutineList.return_value = names
                result = self.latest()
                self.assertEqual(result["status"], "ok")
                self.assertNotIn("routine_name", result["latest"])

    def test_documented_end_error_is_not_confused_with_first_page_or_other_failure(self):
        end = {"error": self.service.provider.config.end_of_history_error}
        self.reader.getSessionList.side_effect = [json.dumps({"page": 0, "sessions": [history_row()]}), json.dumps(end)]
        self.assertEqual(self.latest()["status"], "ok")
        for error in (end, {"error": "Authentication expired"}):
            self.reader.getSessionList.side_effect = None
            self.reader.getSessionList.return_value = error
            self.assertEqual(self.latest()["status"], "unavailable")

    def test_timeout_or_page_limit_discards_partial_history(self):
        self.reader.getSessionList.side_effect = [{"page": 0, "sessions": [history_row()]}, TimeoutError()]
        self.assertEqual(self.latest()["status"], "unavailable")
        self.service.provider.config.max_pages = 1
        self.reader.getSessionList.side_effect = None
        self.reader.getSessionList.return_value = {"page": 0, "sessions": [history_row()]}
        result = self.latest()
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["latest"])

    def test_bad_pages_or_records_never_produce_partial_feedback(self):
        duplicate_exercise = history_row()
        duplicate_exercise["results"].append(copy.deepcopy(duplicate_exercise["results"][0]))
        bad_value = history_row()
        bad_value["results"][0]["done_iter"] = float("nan")
        cases = [None, "bad JSON", {"page": 2, "sessions": []}, {"page": 0, "sessions": None},
                 {"page": 0, "sessions": [history_row(), history_row()]},
                 {"page": 0, "sessions": [history_row(date="2026-10-12T09:00:00")]},
                 {"page": 0, "sessions": [duplicate_exercise]}, {"page": 0, "sessions": [bad_value]}]
        for data in cases:
            with self.subTest(data=data):
                self.reader.getSessionList.side_effect = None
                self.reader.getSessionList.return_value = data
                self.assertEqual(self.latest()["status"], "unavailable")

    def test_unavailable_connector_is_handled(self):
        self.connect.side_effect = ModuleNotFoundError("Ice")
        self.assertEqual(self.latest()["status"], "unavailable")

    def test_missing_native_method_and_invalid_metric_mapping_are_handled(self):
        self.transport.session_history_reader.return_value = object()
        self.assertEqual(self.latest()["status"], "unavailable")
        config = self.service.provider.config.model_copy(deep=True)
        config.metrics["misspelled_field"] = config.metrics["done_iter"]
        with self.assertRaises(ValueError):
            phyxio_sessions.PhyxioSessionProvider(config, self.service.config.first_page)

    def test_tied_latest_sessions_do_not_select_arbitrarily(self):
        self.transport.session_history_reader.return_value = history_reader([history_row(), history_row(routine_id=49)])
        self.assertEqual(self.latest()["status"], "ambiguous_latest")


class SessionHistoryConnectionTests(unittest.TestCase):
    def test_connection_count_and_timeout_are_enforced_without_changing_original_proxy(self):
        reader = Mock()
        service = history_connection(SessionConfig(user_id="participant-a"), reader)
        clients = service._phyxio._admin_servant._clients
        original = clients.pop("one").proxy
        with self.assertRaises(RuntimeError):
            service.session_history_reader("participant-a", 2000)
        clients["one"] = SimpleNamespace(proxy=original)
        for user_id in ("", "participant-b"):
            with self.assertRaises(ValueError):
                service.session_history_reader(user_id, 2000)
        original.ice_invocationTimeout.assert_not_called()
        self.assertIs(service.session_history_reader("participant-a", 2000), original.ice_invocationTimeout.return_value)
        original.ice_invocationTimeout.assert_called_once_with(2000)
        self.assertIs(clients["one"].proxy, original)
        clients["two"] = SimpleNamespace(proxy=Mock())
        with self.assertRaises(RuntimeError):
            service.session_history_reader("participant-a", 2000)

    def test_onboarding_file_is_the_only_participant_input_for_both_tools(self):
        from maya.agents.phyxio_exercise_agent import agent

        with TemporaryDirectory() as folder, patch.dict(os.environ, {
                "MAYA_EXERCISE_SESSION_CONFIG": str(Path(folder) / "exercise.json")}):
            configured_service(folder)
            settings = (Path(folder) / "exercise.json").read_text()
            self.assertNotIn("user_id", settings)
            threads = []
            for participant, repetitions in (("participant-a", 7), ("participant-b", 13)):
                payload = Path(folder) / "onboarding.json"
                payload.write_text(json.dumps({"user_id": participant, "preferences": {"personality_id": "supportive"}}))
                session = load_session(payload)
                config = session.graph_config()
                threads.append(config["configurable"]["thread_id"])
                reader = history_reader([history_row(repetitions=repetitions)])
                connection = history_connection(session, reader)
                with patch.object(phyxio_sessions, "get_phyxio_service", return_value=connection):
                    result = agent.get_latest_exercise_session.invoke({}, config=config)
                    self.assertEqual(result["latest"]["metrics"]["228:done_iter"]["value"], repetitions)
                    self.assertIn(f"{repetitions}/20 reps", agent.get_exercise_history.invoke({}, config=config))
                    reader.reset_mock()
                    wrong = SessionConfig(user_id="not-the-active-participant").graph_config()
                    self.assertEqual(agent.get_latest_exercise_session.invoke({}, config=wrong)["status"], "unavailable")
                    self.assertIn("couldn't fetch", agent.get_exercise_history.invoke({}, config=wrong))
                    self.assertIn("couldn't fetch", agent.get_exercise_history.invoke({}, config={}))
                    reader.getSessionList.assert_not_called()
                    reader.getRoutineList.assert_not_called()
                self.assertEqual((Path(folder) / "exercise.json").read_text(), settings)
            self.assertNotEqual(*threads)


if __name__ == "__main__":
    unittest.main()
