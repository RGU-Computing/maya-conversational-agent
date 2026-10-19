"""Explicit live smoke test: Azure + configured literature retrieval + supervisor.

Run with python -m maya.tests.live_personalization_scenarios. Participant memory
and checkpoints stay in RAM; background memory extraction is outside this test.
Speech and ICE are not exercised. Full replies are saved for human comparison.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv


def run(
    personalities: tuple[str | None, ...] = ("default", "supportive", "concise"),
    output_name: str = "personalization_live_results.json",
) -> list[dict]:
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from langgraph.types import Command

    from maya.framework.conversation.personalization import Preferences, SessionConfig
    from maya.framework.supervisor import factory
    from maya.framework.rag.vectorstore import load_rag_vector_config

    scenarios = [
        ("definition", "What is cardiotoxicity after cancer treatment?", None),
        (
            "worried_question",
            "I'm worried about heart problems after cancer treatment. Can you explain, in general terms, "
            "why follow-up heart monitoring is used?",
            None,
        ),
        (
            "clarification",
            "Is this result bad?",
            "I mean my left ventricular ejection fraction: it is 55%, measured on an echocardiogram "
            "during routine follow-up after chemotherapy. What does that generally mean?",
        ),
    ]
    output = Path(__file__).resolve().parents[2] / "output" / output_name
    output.parent.mkdir(exist_ok=True)
    results: list[dict] = []
    vector_config = load_rag_vector_config()
    metadata = {
        "rag_backend": vector_config.backend,
        "embedding_backend": vector_config.embedding_backend,
        "memory": "isolated in-memory store/checkpoints; background extraction disabled",
        "scope": "live text graph; no speech or ICE",
        "automated_checks": "routing, retrieval, interrupt/resume, nonempty reply; content requires human review",
    }

    with patch.object(factory, "get_postgres_memory", return_value=(InMemoryStore(), InMemorySaver())), \
            patch.object(factory.OntologyMemoryService, "schedule", return_value=False):
        graph = factory.build_supervisor()
        print("Built supervisor with live models and configured literature retrieval.", flush=True)
        for name, question, clarification in scenarios:
            for personality in personalities:
                preferences = Preferences() if personality is None else Preferences(personality_id=personality)
                config = SessionConfig(
                    user_id=f"live-personalization-{name}-{personality}",
                    preferences=preferences,
                ).graph_config()
                started = time.monotonic()
                record = {
                    "scenario": name, "personality_input": personality,
                    "personality": preferences.personality_id, "question": question,
                }
                print(f"Running {name}/{personality}...", flush=True)
                try:
                    state = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
                    interrupts = state.get("__interrupt__", ())
                    record["clarification_questions"] = [str(item.value) for item in interrupts]
                    if interrupts and clarification:
                        state = graph.invoke(Command(resume=clarification), config)
                        record["clarification_answer"] = clarification
                    record["route"] = state.get("route_type")
                    record["completed"] = not bool(state.get("__interrupt__"))
                    last = state["messages"][-1]
                    record["answer"] = last.content if last.type == "ai" else ""
                    record["trace"] = last.additional_kwargs.get("workflow_trace", [])
                    record["flow_passed"] = (
                        record["route"] == "health_rag"
                        and record["completed"]
                        and bool(record["answer"])
                        and bool(interrupts) == bool(clarification)
                        and any(trace.get("retrieval_used") for trace in record["trace"])
                    )
                except Exception as error:
                    record.update(flow_passed=False, error_type=type(error).__name__)
                record["seconds"] = round(time.monotonic() - started, 2)
                results.append(record)
                output.write_text(json.dumps({"metadata": metadata, "results": results}, ensure_ascii=False, indent=2))
                print(json.dumps(record, ensure_ascii=False), flush=True)

    print(f"Saved {len(results)} comparisons to {output}", flush=True)
    if any(not record["flow_passed"] for record in results):
        raise AssertionError("Some live scenarios did not meet the expected flow; inspect the saved results.")
    return results


if __name__ == "__main__":
    run()
