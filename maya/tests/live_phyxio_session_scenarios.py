"""Supervisor → real Phyxio adapter with simulated native history responses.

Run: python -m maya.tests.live_phyxio_session_scenarios
Real models, RAM memory, synthetic responses; no ICE, audio or participant writes.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from dotenv import load_dotenv


def run():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from maya.agents.phyxio_exercise_agent import agent as exercise, phyxio_sessions
    from maya.framework.conversation.personalization import SessionConfig, load_session
    from maya.framework.supervisor import factory
    from maya.tests.test_phyxio_offline_integration import make_service
    from maya.tests.test_phyxio_sessions import configured_service, history_reader, history_row, partial_history_rows, history_connection

    question = "Please review my latest recorded Phyxio exercise session and compare it with my previous sessions."
    cases = [
        ("contextual", ["Start my exercise routine.", "I have finished. How did I perform?"], "ok", 5),
        ("first_session", [question], "ok", 0),
        ("interrupted", [question], "ok", 5),
        ("no_records", [question], "no_data", 0),
        ("timeout", [question], "unavailable", 0),
        ("session_mismatch", [question], "unavailable", 0),
        ("second_participant", [question], "ok", 5),
        ("partial_history", [question], "unavailable", 0),
        ("partial_routine", [question], "ok", 2),
        ("skipped_only", [question], "ok", 2),
        ("history_listing", ["List my recorded Phyxio exercise sessions."], None, 0),
        ("history_session_mismatch", ["List my recorded Phyxio exercise sessions."], None, 0),
        ("other_person", ["How did my father perform in his latest Phyxio exercise session?"], None, 0),
        ("ambiguous", ["How did I perform?"], None, 0),
    ]
    report = []
    output = Path(__file__).resolve().parents[2] / "output/phyxio_provider_live.json"
    output.parent.mkdir(exist_ok=True)
    mirror = make_service()
    mirror.session_history_reader = Mock()
    with TemporaryDirectory() as folder, \
            patch.object(factory, "get_postgres_memory", return_value=(InMemoryStore(), InMemorySaver())), \
            patch.object(factory.OntologyMemoryService, "schedule", return_value=False), \
            patch.object(exercise, "get_phyxio_service", return_value=mirror), \
            patch.object(phyxio_sessions, "get_phyxio_service", return_value=mirror), \
            patch.dict(os.environ, {"MAYA_EXERCISE_SESSION_CONFIG": str(Path(folder) / "exercise.json")}):
        graph = factory.build_supervisor()
        for name, turns, status, count in cases:
            print("Running " + name, flush=True)
            configured_service(folder)
            session_path = Path(folder) / "onboarding.json"
            session_path.write_text(json.dumps({"user_id": "participant-002" if name == "second_participant" else "participant-001"}))
            session = load_session(session_path)
            rows = [history_row(day, value) for day, value in [(1, 10), (3, 12), (5, 14), (7, 16), (9, 18), (12, 20)]]
            if name == "first_session":
                rows = rows[-1:]
            elif name == "second_participant":
                rows[-1]["results"][0]["done_iter"] = 23
            elif name == "interrupted":
                rows[-1]["results"][0].update(status="interrupted", done_iter=8)
            elif name == "no_records":
                rows = []
            elif name in ("partial_routine", "skipped_only", "history_listing"):
                rows = partial_history_rows()
                if name == "skipped_only":
                    rows[-1]["results"][0].update(status="skipped", done_iter=0, elapsed_time=0, report="")
            elif name == "partial_history":
                path = Path(folder) / "exercise.json"
                settings = json.loads(path.read_text())
                settings["phyxio"]["max_pages"] = 1
                path.write_text(json.dumps(settings))
            reader = history_reader(rows)
            if name == "timeout":
                reader.getSessionList.side_effect = TimeoutError()
            active_session = SessionConfig(user_id="different-active-participant") if name in {"session_mismatch", "history_session_mismatch"} else session
            connection = history_connection(active_session, reader)
            mirror.session_history_reader.side_effect = connection.session_history_reader
            mirror.get_routine_list.return_value = json.loads(reader.getRoutineList.return_value)
            evidence = []
            config = session.graph_config()
            for text in turns:
                mirror.reset_mock()
                reader.reset_mock()
                result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
                message = result["messages"][-1]
                evidence.append({"question": text, "answer": message.content, "route": result.get("route_type"),
                    "metadata": message.additional_kwargs, "history_calls": reader.getSessionList.call_count,
                    "history_pages": [call.args[0] for call in reader.getSessionList.call_args_list],
                    "lookup_user_ids": [call.args[0] for call in mirror.session_history_reader.call_args_list],
                    "action_calls": mirror.run_routine.call_count + mirror.show_routine.call_count + mirror.calibrate_routine.call_count,
                    "interrupts": [i.value for i in graph.get_state(config).interrupts]})
            last = evidence[-1]
            data = last["metadata"].get("exercise_session", {})
            checks = {"no_action_on_review": last["action_calls"] == 0,
                      "completed_response": bool(last["answer"]) and not last["interrupts"]}
            if status is not None:
                checks.update(routed_to_exercise=last["route"] == "phyxio_exercise_agent",
                    selected_session_tool=last["metadata"].get("executed_tool") == "get_latest_exercise_session",
                    provider_status=data.get("status") == status,
                    comparison_count=len(data.get("previous", [])) == count)
                if status == "ok":
                    checks["native_date_preserved"] = data["latest"]["recorded_at"].startswith("2026-10-12")
                    checks["no_invented_completion_timestamp"] = "ended_at" not in data["latest"]
                    checks["starts_at_page_zero"] = last["history_pages"][0] == 0
            if last["lookup_user_ids"]:
                checks["identity_from_onboarding"] = all(value == session.user_id for value in last["lookup_user_ids"])
            if name in ("session_mismatch", "history_session_mismatch", "other_person", "ambiguous"):
                checks["no_history_access"] = last["history_calls"] == 0
            if name == "contextual":
                checks["one_start_before_review"] = evidence[0]["action_calls"] == 1
            if name == "interrupted":
                checks["negative_difference_preserved"] = data["comparison"]["228:done_iter"]["difference"] == -6
            if name == "second_participant":
                checks["second_participant_results"] = data["latest"]["metrics"]["228:done_iter"]["value"] == 23
            if name == "partial_routine":
                checks["skipped_measurements_excluded"] = data["comparison"]["228:done_iter"]["previous_count"] == 1
                checks["unperformed_exercise_excluded"] = "229:done_iter" not in data["latest"]["metrics"]
                checks["unset_time_limit"] = "target" not in data["latest"]["metrics"]["228:elapsed_time"]
                checks["unset_time_limit_kind"] = "target_kind" not in data["latest"]["metrics"]["228:elapsed_time"]
            if name == "skipped_only":
                checks["no_invented_performance"] = data["latest"]["metrics"] == {} and data["comparison"] == {}
            if name in {"history_listing", "history_session_mismatch"}:
                checks["selected_history_tool"] = last["metadata"].get("executed_tool") == "get_exercise_history"
            if name == "history_listing":
                checks["starts_at_page_zero"] = last["history_pages"] == [0]
            entry = {"case": name, "passed": all(checks.values()), "checks": checks, "turns": evidence}
            report.append(entry)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            print(json.dumps({key: entry[key] for key in ("case", "passed", "checks")}), flush=True)
    assert all(entry["passed"] for entry in report), f"Review {output}"


if __name__ == "__main__":
    run()
