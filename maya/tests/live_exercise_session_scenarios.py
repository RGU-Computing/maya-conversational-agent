"""Explicit live supervisor → exercise → session-tool tests with private RAM memory.

Run: python -m maya.tests.live_exercise_session_scenarios [case ...]
Real routing/decision/response models; synthetic session records; simulated mirror
transport for start requests. No PostgreSQL writes, microphone, TTS or ICE.
"""

import json
import os
from pathlib import Path
from time import monotonic
from unittest.mock import patch

from dotenv import load_dotenv


def run(selected_cases=None):
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from maya.agents.phyxio_exercise_agent import agent as exercise, sessions
    from maya.framework.conversation.personalization import SessionConfig
    from maya.framework.supervisor import factory
    from maya.tests.test_exercise_sessions import sample_service
    from maya.tests.test_phyxio_offline_integration import make_service

    question = "Please review my latest recorded exercise session and compare it with my previous sessions."
    cases = [
        ("contextual", ["Start my exercise routine.", "I have finished. How did I perform?"], "normal", "phyxio_exercise_agent", 1),
        ("short_history", [question], "short", "phyxio_exercise_agent", 1),
        ("first_session", [question], "single", "phyxio_exercise_agent", 1),
        ("lower_interrupted", [question], "lower", "phyxio_exercise_agent", 1),
        ("no_records", [question], "empty", "phyxio_exercise_agent", 1),
        ("provider_timeout", [question], "timeout", "phyxio_exercise_agent", 1),
        ("ambiguous", ["How did I perform?"], "normal", "direct_answer", 0),
        ("topic_change", [question, "I mean my piano practice now. How did I perform?"], "normal", "direct_answer", 0),
        ("other_person", ["How did my father perform in his latest Phyxio exercise session?"], "normal", "phyxio_exercise_agent", 0),
        ("review_not_start", ["Don't start any exercises. Just review my latest exercise session."], "normal", "phyxio_exercise_agent", 1),
        ("older_session", ["Review my exercise session from 1 September, not the latest one."], "normal", "phyxio_exercise_agent", 0),
        ("symptoms", ["I have chest pain after my exercise routine. How did I perform?"], "normal", "health_rag", 0),
        ("spanish", ["¿Cómo me fue en mi última sesión de ejercicios? Compárala con las anteriores."], "normal", "phyxio_exercise_agent", 1),
    ]
    if selected_cases:
        cases = [case for case in cases if case[0] in selected_cases]
    report = []
    suffix = "_" + "_".join(selected_cases) if selected_cases else ""
    output = Path(__file__).resolve().parents[2] / f"output/exercise_sessions_live{suffix}.json"
    output.parent.mkdir(exist_ok=True)
    mirror = make_service()
    with patch.object(factory, "get_postgres_memory", return_value=(InMemoryStore(), InMemorySaver())), \
            patch.object(factory.OntologyMemoryService, "schedule", return_value=False), \
            patch.object(exercise, "get_phyxio_service", return_value=mirror):
        graph = factory.build_supervisor()
        for name, turns, mode, expected_route, expected_fetches in cases:
            print(f"Running {name}...", flush=True)
            service = sample_service()
            if mode == "short":
                service.provider.sessions = service.provider.sessions[-3:]
            elif mode == "single":
                service.provider.sessions = service.provider.sessions[-1:]
            elif mode == "lower":
                latest = service.provider.sessions[-1]
                latest.status = "interrupted"
                latest.metrics["seated_march_repetitions"].value = 8
            elif mode == "empty":
                service.provider.sessions = []
            config = SessionConfig(user_id="participant-001").graph_config()
            evidence = []
            with patch.object(sessions, "load_session_service", return_value=service), \
                    patch.object(service.provider, "fetch_sessions", wraps=service.provider.fetch_sessions) as fetch:
                if mode == "timeout":
                    fetch.side_effect = TimeoutError("Injected session-provider timeout")
                for question in turns:
                    mirror.reset_mock()
                    fetch.reset_mock()
                    started = monotonic()
                    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
                    message = result["messages"][-1]
                    evidence.append({"question": question, "seconds": round(monotonic()-started, 2),
                                     "route": result.get("route_type"), "answer": message.content,
                                     "metadata": message.additional_kwargs, "fetch_calls": fetch.call_count,
                                     "mirror_calls": [str(call) for call in mirror.mock_calls],
                                     "interrupts": [i.value for i in graph.get_state(config).interrupts]})
            last = evidence[-1]
            checks = {"route": last["route"] == expected_route,
                      "fetch_count": last["fetch_calls"] == expected_fetches,
                      "no_mirror_action_on_review": not last["mirror_calls"]}
            if expected_fetches:
                checks["selected_session_tool"] = last["metadata"].get("executed_tool") == "get_latest_exercise_session"
                data = last["metadata"].get("exercise_session", {})
                checks["provider_status"] = data.get("status") == {"empty": "no_data", "timeout": "unavailable"}.get(mode, "ok")
                if mode not in ("empty", "timeout"):
                    expected_count = {"short": 2, "single": 0}.get(mode, 5)
                    checks["comparison_count"] = len(data.get("previous", [])) == expected_count
            if name == "contextual":
                checks["start_executed_once"] = evidence[0]["metadata"].get("executed_tool") == "start_exercises"
            report.append({"case": name, "passed": all(checks.values()), "checks": checks, "turns": evidence})
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            print(json.dumps({"case": name, "checks": checks}), flush=True)
    assert all(row["passed"] for row in report), f"Review {output}"
    return report


if __name__ == "__main__":
    import sys
    run(sys.argv[1:] or None)
