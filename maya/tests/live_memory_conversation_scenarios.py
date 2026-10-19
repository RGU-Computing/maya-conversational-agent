"""Live clarification → ontology memory → new-conversation checks.

Run explicitly: python -m maya.tests.live_memory_conversation_scenarios [successful_save|timeout_recovery]
Uses real models/retrieval, synthetic biomarkers and an isolated RAM store.
Rebuilds graphs and memory workers; does not exercise PostgreSQL, audio or ICE.
"""

import json
import os
from pathlib import Path
from threading import Event
from time import monotonic
from unittest.mock import Mock, patch

from dotenv import load_dotenv


QUESTION = (
    "Please interpret my latest blood pressure measurement and explain whether "
    "it is safe given the cancer treatment I received."
)
REPLY = "I finished doxorubicin chemotherapy six months ago."


def run(selected=()):
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from langgraph.types import Command
    from maya.framework.memory.ontology import OntologyMemoryService
    from maya.framework.conversation.personalization import SessionConfig
    from maya.framework.supervisor import factory
    from maya.tests.test_biomarkers import demo_service

    report = []
    suffix = '_' + '_'.join(selected) if selected else ''
    output = Path(__file__).resolve().parents[2] / f"output/memory_conversation_live{suffix}.json"
    output.parent.mkdir(exist_ok=True)

    def save(name, checks, **evidence):
        entry = {"case": name, "passed": all(checks.values()), "checks": checks, **evidence}
        report.append(entry)
        output.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps({"case": name, "passed": entry["passed"], "checks": checks}), flush=True)

    def drain(memory):
        with memory._condition:
            assert memory._condition.wait_for(lambda: not memory._scheduled_jobs, timeout=120), \
                "Background memory worker did not finish within the test deadline"

    def turn(graph, config, value):
        start = monotonic()
        result = graph.invoke(value, config)
        message = result["messages"][-1]
        return {
            "seconds": round(monotonic() - start, 2),
            "interrupts": [i.value for i in graph.get_state(config).interrupts],
            "answer": message.content if isinstance(message, AIMessage) else "",
            "metadata": message.additional_kwargs,
            "selected_models": result.get("selected_ontology_models", []),
            "memory_update_status": result.get("memory_update_status"),
        }

    def ask(graph, user, question=QUESTION):
        config = SessionConfig(user_id=user).graph_config()
        return config, turn(graph, config, {"messages": [HumanMessage(content=question)]})

    for fail_save in (False, True):
        name = "timeout_recovery" if fail_save else "successful_save"
        if selected and name not in selected:
            continue
        print(f"Running {name}...", flush=True)
        store = InMemoryStore()
        biomarkers = demo_service()
        user = f"live-memory-{name}"
        other = f"{user}-other"
        biomarkers.config.patient_map.update({user: "synthetic-patient", other: "synthetic-patient"})
        services = []
        release = Event()

        def build(*, fail=False, block=False):
            def memory_factory(**kwargs):
                memory = OntologyMemoryService(**kwargs, register_shutdown=False)
                original_factory = memory.extractor_factory
                failed = False

                def extractor_factory(model):
                    real = original_factory(model)

                    def invoke(*args, **kw):
                        nonlocal failed
                        if fail and not failed:
                            failed = True
                            raise TimeoutError("Injected first-save timeout")
                        if block:
                            assert release.wait(120), "Test recovery gate was not released"
                        return real.invoke(*args, **kw)

                    return Mock(invoke=invoke)

                memory.extractor_factory = extractor_factory
                services.append(memory)
                return memory

            with patch.object(factory, "get_postgres_memory", return_value=(store, InMemorySaver())), \
                    patch.object(factory, "OntologyMemoryService", side_effect=memory_factory):
                graph = factory.build_supervisor(biomarker_service=biomarkers)
            return graph, services[-1]

        try:
            graph, memory = build(fail=fail_save)
            with patch.object(biomarkers.provider, "fetch_latest", wraps=biomarkers.provider.fetch_latest) as fetch:
                config, first = ask(graph, user)
                assert first["interrupts"], f"Expected treatment clarification: {first}"
                answered = turn(graph, config, Command(resume=REPLY))
                lookup_count = fetch.call_count
            drain(memory)
            before = memory.load_records(user, ["cancer_history"])
            pending = store.search(memory.pending_namespace(user))
            save(f"{name}_first_conversation", {
                "asks_for_treatment": "treatment" in str(first["interrupts"]).lower(),
                "fetches_once_across_resume": lookup_count == 1,
                "answers_with_references": bool(answered["metadata"].get("presentation", {}).get("references")),
                "queues_resolved_clarification": answered["memory_update_status"] == "scheduled",
                "expected_save_state": (not before and bool(pending)) if fail_save else (
                    bool(before) and "doxorubicin" in json.dumps(before).lower() and not pending),
            }, first=first, answered=answered, records=before,
                 pending_payloads=[item.value for item in pending])

            memory.close()
            graph, memory = build(block=fail_save)
            if fail_save:
                _, waiting = ask(graph, user, QUESTION + " I have hypertension.")
                save("recovery_waits_for_verification", {
                    "reports_pending": waiting["metadata"].get("memory_status") == "pending",
                    "does_not_repeat_clarification": not waiting["interrupts"],
                    "does_not_retrieve": not waiting["metadata"].get("workflow_trace"),
                    "preserves_current_turn_for_memory": waiting["memory_update_status"] == "scheduled",
                }, waiting=waiting)
                release.set()
                drain(memory)

            _, recalled = ask(graph, user)
            drain(memory)
            records = memory.load_records(user, ["cancer_history", "clinical_profile"])
            save(f"{name}_new_conversation", {
                "does_not_ask_treatment_again": not recalled["interrupts"],
                "uses_saved_treatment": "doxorubicin" in json.dumps(recalled["metadata"]).lower(),
                "answers_with_references": bool(recalled["metadata"].get("presentation", {}).get("references")),
                "verified_treatment_saved": any(r["binding_path"] == "cancer_history.treatment_records"
                    and "doxorubicin" in json.dumps(r["properties"]).lower() for r in records),
                "pending_queue_drained": not store.search(memory.pending_namespace(user)),
                "new_assertion_retained": not fail_save or any(r["binding_path"] == "clinical_profile.condition_records"
                    and "hypertension" in json.dumps(r["properties"]).lower() for r in records),
            }, recalled=recalled, records=records)

            _, isolated = ask(graph, other)
            save(f"{name}_different_user", {
                "asks_for_own_treatment": bool(isolated["interrupts"])
                    and "treatment" in str(isolated["interrupts"]).lower(),
                "no_inherited_records": not memory.load_records(other, ["cancer_history", "clinical_profile"]),
            }, result=isolated)
        finally:
            release.set()
            for memory in services:
                memory.close()

    assert all(entry["passed"] for entry in report), f"Review {output}"
    return report


if __name__ == "__main__":
    import sys
    run(sys.argv[1:])
