"""Explicit live model/retrieval smoke test with synthetic observations and RAM memory.

Run: python -m maya.tests.live_biomarker_scenarios
No participant database writes, ICE or audio playback.
"""

import json
import os
from pathlib import Path
from time import monotonic
from unittest.mock import patch

from dotenv import load_dotenv


def run(selected_cases=None):
    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from maya.framework.supervisor import factory
    from maya.tests.test_biomarkers import demo_service

    service = demo_service()
    store = InMemoryStore()
    with patch.object(factory, "get_postgres_memory", return_value=(store, InMemorySaver())), \
            patch.object(factory.OntologyMemoryService, "schedule", return_value=False):
        graph = factory.build_supervisor(biomarker_service=service)

    cases = [
        ("readout", "Please show my latest blood pressure and blood glucose measurements, with dates.", True),
        ("interpretation", "What does my latest blood pressure measurement mean? Please use my recorded result.", True),
        ("general", "In general, what is blood pressure?", False),
        ("other_person", "What does my friend's blood pressure reading mean?", False),
        ("health_initiated", "Please explain my latest fasting blood glucose using the recorded measurement.", True),
    ]
    if selected_cases:
        cases = [case for case in cases if case[0] in selected_cases]
    report = []
    suffix = "_" + "_".join(selected_cases) if selected_cases else ""
    output = Path(__file__).resolve().parents[2] / f"output/biomarker_live_results{suffix}.json"
    output.parent.mkdir(exist_ok=True)
    for name, question, expected_lookup in cases:
        print(f"Running {name}...", flush=True)
        start = monotonic()
        config = {"configurable": {"user_id": "participant-001", "thread_id": f"biomarker-live-{name}"}}
        with patch.object(service.provider, "fetch_latest", wraps=service.provider.fetch_latest) as fetch:
            if name == "health_initiated":
                # Enter HealthRAG with no supervisor lookup to exercise its own decision.
                graph.update_state(config, {"messages": [HumanMessage(content=question)],
                    "route_type": "health_rag", "question_subject": "current_user", "selected_ontology_models": ["measurement_biomarker"],
                    "biomarker_results": []}, as_node="supervisor")
                result = graph.invoke(None, config)
            else:
                result = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
        snapshot = graph.get_state(config)
        answer = result["messages"][-1]
        entry = {"case": name, "seconds": round(monotonic()-start, 2), "route": result["route_type"],
                 "lookup_calls": fetch.call_count, "answer": answer.content,
                 "interrupts": [i.value for i in snapshot.interrupts],
                 "metadata": answer.additional_kwargs}
        entry["passed"] = bool(fetch.call_count) == expected_lookup
        if expected_lookup:
            entry["passed"] &= fetch.call_count == 1 and not snapshot.interrupts
            entry["passed"] &= service.config.synthetic_notice in answer.content
        if name == "readout":
            entry["passed"] &= result["route_type"] == "biomarkers"
        if name == "interpretation":
            entry["passed"] &= bool(answer.additional_kwargs.get("presentation", {}).get("references"))
        report.append(entry)
        output.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps({key: entry[key] for key in ("case", "seconds", "lookup_calls", "passed")}), flush=True)
    assert all(entry["passed"] for entry in report), f"Review {output}"
    return report


if __name__ == "__main__":
    import sys
    run(sys.argv[1:] or None)
