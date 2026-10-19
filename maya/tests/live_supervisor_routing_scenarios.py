"""Live Azure checks for supervisor routing and ontology-model selection.

Run explicitly with outbound access:

    python -m maya.tests.live_supervisor_routing_scenarios
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from maya.framework.integrations.models import get_chat_model
from maya.framework.ontology.registry import OntologyRegistry
from maya.framework.supervisor.factory import SupervisorRouteDecision
from maya.framework.supervisor.factory import HealthRouteGuardDecision
from maya.framework.supervisor.prompts import (
    SUPERVISOR_SYSTEM_MESSAGE,
    build_health_route_guard_prompt,
)


@dataclass(frozen=True)
class RoutingScenario:
    name: str
    text: str
    expected_route: str
    expected_models: tuple[str, ...]


SCENARIOS = (
    RoutingScenario(
        "fact_only_medication",
        "Desde ayer tomo Bisoprolol 2,5 mg.",
        "direct_answer",
        ("clinical_profile",),
    ),
    RoutingScenario(
        "medication_question",
        "Is Bisoprolol safe for me to take with asthma?",
        "health_rag",
        ("clinical_profile",),
    ),
    RoutingScenario(
        "fact_only_treatment",
        "My chemotherapy included Doxorubicin in 2019.",
        "direct_answer",
        ("cancer_history",),
    ),
    RoutingScenario(
        "mixed_fact_and_question",
        "I take Bisoprolol 2.5 mg daily. Is it safe for me to exercise?",
        "health_rag",
        ("clinical_profile",),
    ),
    RoutingScenario(
        "profile_update",
        "My name is Sam and I live in Manchester.",
        "user_profile",
        (),
    ),
    RoutingScenario(
        "general_memory_update",
        "I felt worried before my appointment today.",
        "general_memory",
        (),
    ),
    RoutingScenario(
        "biomarker_question",
        "What is a cardiac biomarker?",
        "health_rag",
        (),
    ),
    RoutingScenario(
        "exercise_request",
        "Show my exercise routine on the mirror.",
        "phyxio_exercise_agent",
        ("exercise",),
    ),
    RoutingScenario(
        "ordinary_conversation",
        "Tell me a short joke.",
        "direct_answer",
        (),
    ),
)


def run() -> list[dict[str, object]]:
    registry = OntologyRegistry()
    system_message = "\n\n".join(
        [
            SUPERVISOR_SYSTEM_MESSAGE.format(
                user_profile=None,
                general_memory="No general memories.",
                time="2026-07-16T12:00:00+01:00",
            ),
            registry.catalog_for_agent("supervisor"),
        ]
    )
    model = get_chat_model()
    router = model.with_structured_output(SupervisorRouteDecision, strict=True)
    health_guard = model.with_structured_output(HealthRouteGuardDecision, strict=True)
    results: list[dict[str, object]] = []
    failures: list[str] = []

    for scenario in SCENARIOS:
        decision = router.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content=scenario.text),
            ]
        )
        selected = set(registry.resolve_model_ids(decision.selected_ontology_models))
        route_type = decision.route_type
        if decision.fact_only_ontology_update and registry.has_active_bindings(selected):
            route_type = "direct_answer"
        if route_type == "direct_answer":
            guarded = health_guard.invoke(
                [
                    SystemMessage(content=build_health_route_guard_prompt()),
                    HumanMessage(content=scenario.text),
                ]
            )
            if guarded.route_type != "direct_answer":
                route_type = guarded.route_type
        agent_id = route_type if route_type in {"health_rag", "phyxio_exercise_agent"} else "supervisor"
        selected = set(
            registry.resolve_model_ids(decision.selected_ontology_models, agent_id=agent_id)
        )
        expected_models = set(scenario.expected_models)
        passed = route_type == scenario.expected_route and expected_models <= selected
        if not passed:
            failures.append(
                f"{scenario.name}: expected route={scenario.expected_route}, models>={sorted(expected_models)}; "
                f"got route={route_type}, models={sorted(selected)}"
            )
        results.append(
            {
                "name": scenario.name,
                "passed": passed,
                "route": route_type,
                "selected_models": sorted(selected),
            }
        )

    print(json.dumps(results, indent=2))
    if failures:
        raise AssertionError("\n".join(failures))
    return results


if __name__ == "__main__":
    run()
