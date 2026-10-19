"""Real supervisor/models/retrieval with the FHIR adapter and simulated HTTP responses.

Run: python -m maya.tests.live_fhir_scenarios
Uses synthetic data and RAM memory; never contacts UCLM or writes participant data.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dotenv import load_dotenv


def run():
    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from requests.exceptions import Timeout
    from maya.framework.supervisor import factory
    from maya.tests.live_health_config_scenarios import run as health_scenarios
    from maya.tests.test_fhir_biomarkers import configured_service, observation, response

    report = []
    with TemporaryDirectory() as folder:
        service = configured_service(folder)
        session = service.provider.client._session
        try:
            with patch.object(session, "get", side_effect=lambda *args, **kwargs: response([observation()])):
                report.extend(health_scenarios(
                    ["biomarker_clarification", "unknown_test", "missing_observation"], service=service))
            with patch.object(factory, "get_postgres_memory", return_value=(InMemoryStore(), InMemorySaver())), \
                    patch.object(factory.OntologyMemoryService, "schedule", return_value=False):
                graph = factory.build_supervisor(biomarker_service=service)
            for name, http_result, expected in [
                ("readout", response([observation()]), "ok"),
                ("no_records", response(), "no_data"),
                ("timeout", Timeout(), "unavailable"),
                ("forbidden", response(status=403), "unavailable"),
            ]:
                print("Running " + name, flush=True)
                outcome = {"side_effect": http_result} if isinstance(http_result, Exception) else {"return_value": http_result}
                config = {"configurable": {"user_id": "participant-001", "thread_id": "fhir-" + name}}
                with patch.object(session, "get", **outcome) as get:
                    result = graph.invoke({"messages": [HumanMessage(content=
                        "Please show my latest blood pressure measurement with its date, without interpreting it.")]}, config)
                message = result["messages"][-1]
                checks = {
                    "direct_readout": result["route_type"] == "biomarkers",
                    "one_http_request": get.call_count == 1,
                    "correct_outcome": result["biomarker_results"][0]["status"] == expected,
                    "completed": bool(message.content) and not graph.get_state(config).interrupts,
                    "no_credential_leak": "synthetic-secret" not in str(result),
                }
                if name == "readout":
                    checks["reports_values"] = "134" in message.content and "82" in message.content
                    checks["matches_question_language"] = "blood pressure" in message.content.lower()
                if expected == "unavailable":
                    answer = message.content.lower()
                    checks["explains_access_failure"] = any(word in answer for word in ("access", "retrieve", "retriev", "verify"))
                    checks["does_not_claim_absent_records"] = not any(phrase in answer for phrase in (
                        "no records", "not available in your records", "couldn't find"))
                entry = {"case": name, "passed": all(checks.values()), "checks": checks, "answer": message.content}
                report.append(entry)
                print(json.dumps({key: entry[key] for key in ("case", "passed", "checks")}), flush=True)
        finally:
            session.close()
    output = Path(__file__).resolve().parents[2] / "output/fhir_live_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str))
    assert all(entry["passed"] for entry in report), f"Review {output}"


if __name__ == "__main__":
    run()
