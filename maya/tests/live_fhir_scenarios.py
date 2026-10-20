"""Real supervisor/models/retrieval with the FHIR adapter and simulated HTTP responses.

Run: python -m maya.tests.live_fhir_scenarios [case ...]
Uses synthetic data and RAM memory; never contacts UCLM or writes participant data.
"""

import json
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dotenv import load_dotenv


def run(selected=()):
    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from langgraph.types import Command
    from requests import Session
    from requests.exceptions import Timeout
    from maya.framework.supervisor import factory
    from maya.framework.conversation.personalization import SessionConfig
    from maya.tests.live_health_config_scenarios import run as health_scenarios
    from maya.tests.test_fhir_biomarkers import configured_service, observation, response

    report = []
    original_get = Session.get

    def transport(result):
        def get(session, url, **kwargs):
            if not url.startswith("https://fhir.invalid/"):
                return original_get(session, url, **kwargs)
            if isinstance(result, Exception):
                raise result
            return result
        return get

    with TemporaryDirectory() as folder, \
            patch.object(factory.OntologyMemoryService, "schedule", return_value=False):
        service = configured_service(folder)
        health_cases = [name for name in ("biomarker_clarification", "unknown_test", "missing_observation")
                        if not selected or name in selected]
        if health_cases:
            with patch("requests.Session.get", autospec=True, side_effect=transport(response([observation()]))):
                report.extend(health_scenarios(health_cases, service=service))
        with patch.object(factory, "get_postgres_memory", return_value=(InMemoryStore(), InMemorySaver())):
            graph = factory.build_supervisor(biomarker_service=service)
        readout = "Please show my latest blood pressure measurement with its date, without interpreting it."
        weight = observation("body-weight", valueQuantity={"value": 150, "unit": "lb", "code": "[lb_av]"})
        del weight["component"]
        for name, http_result, expected, question in [
            ("readout", response([observation()]), "ok", readout),
            ("ambiguous_empty_result", response(), "unavailable", readout),
            ("timeout", Timeout(), "unavailable", readout),
            ("forbidden", response(status=403), "unavailable", readout),
            ("missing_scalar_unit", response([weight]), "unavailable",
             "Please show my latest recorded body weight and its date, without interpreting it."),
        ]:
            if selected and name not in selected:
                continue
            print("Running " + name, flush=True)
            config = {"configurable": {"user_id": "participant-001", "thread_id": "fhir-" + name}}
            with patch("requests.Session.get", autospec=True, side_effect=transport(http_result)) as get:
                result = graph.invoke({"messages": [HumanMessage(content=
                    question)]}, config)
            message = result["messages"][-1]
            checks = {
                "direct_readout": result["route_type"] == "biomarkers",
                "one_http_request": sum(call.args[1].startswith("https://fhir.invalid/") for call in get.call_args_list) == 1,
                "correct_outcome": result["biomarker_results"][0]["status"] == expected,
                "completed": bool(message.content) and not graph.get_state(config).interrupts,
                "no_credential_leak": "synthetic-secret" not in str(result),
            }
            if name == "readout":
                checks["reports_values"] = "134" in message.content and "82" in message.content
                checks["matches_question_language"] = "blood pressure" in message.content.lower()
            if name == "missing_scalar_unit":
                answer = message.content.lower()
                checks["withholds_unvalidated_value"] = "150" not in answer
                checks["does_not_invent_unit"] = not any(
                    word in answer for word in ("kg", "kilogram", "pound", " lb"))
            if expected == "unavailable":
                answer = message.content.lower()
                checks["explains_access_failure"] = any(word in answer for word in ("access", "retrieve", "retriev", "verify"))
                checks["does_not_claim_absent_records"] = not any(phrase in answer for phrase in (
                    "no records", "not available in your records", "couldn't find"))
            entry = {"case": name, "passed": all(checks.values()), "checks": checks, "answer": message.content}
            report.append(entry)
            print(json.dumps({key: entry[key] for key in ("case", "passed", "checks")}), flush=True)

        if not selected or "missing_unit_clarification" in selected:
            print("Running missing_unit_clarification", flush=True)
            config = SessionConfig(user_id="participant-001").graph_config()
            with patch("requests.Session.get", autospec=True, side_effect=transport(response([weight]))) as http:
                result = graph.invoke({"messages": [HumanMessage(content=
                    "I am 1.80 metres tall. Is my latest recorded body weight healthy for me?")]}, config)
                questions = [str(item.value) for item in graph.get_state(config).interrupts]
                first_calls = sum(call.args[1].startswith("https://fhir.invalid/") for call in http.call_args_list)
                resumed = graph.invoke(Command(resume="My weight is 150 pounds."), config) if questions else result
            message = resumed["messages"][-1]
            metadata = message.additional_kwargs
            query = " ".join(item.get("retrieval_query", "") for item in metadata.get("workflow_trace", []))
            checks = {
                "health_route": result["route_type"] == "health_rag",
                "fetched_before_clarification": first_calls == 1,
                "rejects_incomplete_measurement": result["biomarker_results"][0]["status"] == "unavailable"
                    and not result["biomarker_results"][0]["observations"],
                "asks_for_measurement": bool(questions) and any(
                    word in " ".join(questions).lower() for word in ("weight", "measurement", "reading")),
                "does_not_refetch_on_resume": sum(
                    call.args[1].startswith("https://fhir.invalid/") for call in http.call_args_list) == 1,
                "completed_with_references": bool(metadata.get("presentation", {}).get("references"))
                    and not graph.get_state(config).interrupts,
                "query_uses_confirmed_unit": "150" in query and any(
                    word in query.lower() for word in ("pound", "lb")),
            }
            entry = {"case": "missing_unit_clarification", "passed": all(checks.values()),
                     "checks": checks, "questions": questions, "answer": message.content,
                     "retrieval_query": query, "metadata": metadata}
            report.append(entry)
            print(json.dumps({key: entry[key] for key in ("case", "passed", "checks")}), flush=True)

        if not selected or "main_runner" in selected:
            # Exercise the real supported runner with a real supervisor and SDK.
            # Only terminal input, native audio and FHIR HTTP are simulated here.
            from maya.framework.supervisor import run_with_vad as runner

            (Path(folder) / "token.txt").write_text("synthetic-file-token\n")
            (Path(folder) / "client.json").write_text(json.dumps({
                "url": "https://fhir.invalid/proxy", "token": None, "token_file": "token.txt"}))
            service.provider = type(service.provider)(service.config.fhir, Path(folder), service.config.accepted_statuses)
            output = io.StringIO()
            with patch.object(runner, "build_supervisor", return_value=graph), \
                    patch.object(runner, "init_phyxio_io", return_value=None), \
                    patch.object(runner, "init_asr", return_value=(None, None)), \
                    patch.object(runner, "init_tts", return_value=None), \
                    patch.object(runner, "read_text", side_effect=[readout, None]), \
                    patch.object(runner, "speak_output") as speech, \
                    patch.dict(os.environ, {"MAYA_RUN_LOCK": str(Path(folder) / "runner.lock"), "MAYA_RUN_THREAD_ID": ""}), \
                    patch("requests.Session.get", autospec=True, side_effect=transport(response([observation()]))) as http, \
                    redirect_stdout(output):
                runner.main(SessionConfig(user_id="participant-001"))
            spoken = str(speech.call_args)
            checks = {
                "one_fhir_request": sum(call.args[1].startswith("https://fhir.invalid/") for call in http.call_args_list) == 1,
                "displayed_values": "134" in output.getvalue() and "82" in output.getvalue(),
                "speech_values": "134" in spoken and "82" in spoken,
                "no_unsolicited_greeting": speech.call_count == 1,
                "token_file_used": any(call.kwargs.get("json", {}).get("token") == "synthetic-file-token"
                                       for call in http.call_args_list),
                "no_credential_leak": all(token not in output.getvalue() + spoken
                                         for token in ("synthetic-secret", "synthetic-file-token")),
            }
            entry = {"case": "main_runner", "passed": all(checks.values()), "checks": checks}
            report.append(entry)
            print(json.dumps(entry), flush=True)
    suffix = "_" + "_".join(selected) if selected else ""
    output = Path(__file__).resolve().parents[2] / f"output/fhir_live_results{suffix}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str))
    assert all(entry["passed"] for entry in report), f"Review {output}"


if __name__ == "__main__":
    import sys
    run(sys.argv[1:])
