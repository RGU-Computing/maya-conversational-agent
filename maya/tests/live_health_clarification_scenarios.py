"""Live Azure checks for the HealthRAG clarification gate.

Run explicitly with outbound access:

    python -m maya.tests.live_health_clarification_scenarios
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from maya.agents.health_rag.agent import HealthRagHandle
from maya.framework.integrations.models import get_chat_model
from maya.framework.conversation.clarification import load_disambiguation_policy, render_disambiguation_block
from maya.framework.ontology import get_ontology_registry


@dataclass(frozen=True)
class ClarificationScenario:
    name: str
    question: str
    expected_clarification: bool
    conversation_context: str = ""
    treatment_memory: tuple[dict, ...] = ()
    expected_field_terms: tuple[str, ...] = ()


SCENARIOS = (
    ClarificationScenario(
        "reported_medication_missing_treatment",
        "I'm taking bisoprolol for my blood pressure. Is it okay because of my treatments?",
        True,
        expected_field_terms=("treatment", "exposure", "intervention"),
    ),
    ClarificationScenario(
        "reported_medication_with_historical_treatment_memory",
        "I'm taking bisoprolol for my blood pressure. Is it okay because of my treatments?",
        False,
        treatment_memory=({"model_id": "cancer_history", "binding_path": "cancer_history.treatment_records",
                           "properties": {"treatment_name": "Doxorubicin", "treatment_period": "2019"},
                           "source_type": "user_reported"},),
    ),
    ClarificationScenario(
        "new_treatment_cannot_be_resolved_from_old_exposure",
        "I've started a new cancer treatment this week. Is my bisoprolol okay alongside it?",
        True,
        treatment_memory=({"model_id": "cancer_history", "binding_path": "cancer_history.treatment_records",
                           "properties": {"treatment_name": "Doxorubicin", "treatment_period": "2019"},
                           "source_type": "user_reported"},),
        expected_field_terms=("treatment", "exposure", "intervention"),
    ),
    ClarificationScenario(
        "other_medication_missing_treatment",
        "I take amlodipine for blood pressure. Is it suitable alongside my cancer treatments?",
        True,
        expected_field_terms=("treatment", "exposure", "intervention"),
    ),
    ClarificationScenario(
        "friend_cannot_inherit_users_treatment_memory",
        "My friend takes bisoprolol. Is it okay with her cancer treatments?",
        True,
        treatment_memory=({"model_id": "cancer_history", "binding_path": "cancer_history.treatment_records",
                           "properties": {"treatment_name": "Doxorubicin", "treatment_period": "2019"},
                           "source_type": "user_reported"},),
        expected_field_terms=("treatment", "exposure", "intervention"),
    ),
    ClarificationScenario(
        "medication_general_education",
        "What is bisoprolol generally used for?",
        False,
    ),
    ClarificationScenario(
        "reported_clear_survivorship_question",
        "How can I manage my cardiovascular health as a cancer survivor six months after finishing chemotherapy?",
        False,
    ),
    ClarificationScenario(
        "clear_blood_pressure_management",
        "How can I manage my blood pressure?",
        False,
    ),
    ClarificationScenario(
        "clear_cardiotoxicity_definition",
        "Tell me about cardiotoxicity after cancer treatment.",
        False,
    ),
    ClarificationScenario(
        "clear_surveillance_education",
        "What heart monitoring is generally used after anthracycline chemotherapy?",
        False,
    ),
    ClarificationScenario(
        "clear_lvef_interpretation",
        "My LVEF is 45% six months after doxorubicin. What does that generally mean?",
        False,
    ),
    ClarificationScenario(
        "clear_spanish_question",
        "¿Cómo puedo cuidar mi salud cardiovascular después de terminar la quimioterapia?",
        False,
    ),
    ClarificationScenario(
        "missing_treatment_exposure",
        "Am I at high risk of cardiotoxicity from the cancer treatment I received?",
        True,
    ),
    ClarificationScenario(
        "unnamed_echocardiogram_change",
        "My echocardiogram changed after chemotherapy. Does this mean I have cardiotoxicity?",
        True,
    ),
    ClarificationScenario(
        "unnamed_cardiac_biomarker",
        "My cardiac blood test is elevated after doxorubicin. Does this mean heart damage?",
        True,
    ),
    ClarificationScenario(
        "vague_post_treatment_symptom",
        "I have felt something unusual since chemotherapy. Could it be cardiotoxicity?",
        True,
    ),
    ClarificationScenario(
        "treatment_resolved_from_context",
        "What heart monitoring may be relevant for that treatment?",
        False,
        "The user previously said they received doxorubicin in 2019.",
    ),
    ClarificationScenario(
        "urgent_symptoms",
        "I have chest pain and shortness of breath right now. What should I do?",
        False,
    ),
)


def run(scenarios=SCENARIOS, output_name="health_clarification_live_results.json") -> list[dict[str, object]]:
    ontology = load_disambiguation_policy()
    handle = HealthRagHandle(
        name="health_rag",
        agent=None,
        ontology_block=render_disambiguation_block(
            ontology,
            limit=None,
            rules_per_entity=None,
        ),
        ontology_entity_ids=frozenset(entity["concept_id"] for entity in ontology["entities"]),
        ontology_rule_counts={entity["concept_id"]: len(entity["prompt_rules"]) for entity in ontology["entities"]},
    )
    registry = get_ontology_registry()
    model = get_chat_model()
    results: list[dict[str, object]] = []
    failures: list[str] = []

    for scenario in scenarios:
        print(f"Checking {scenario.name}...", flush=True)
        decision = handle.assess_clarification(
            scenario.question,
            model=model,
            conversation_context=scenario.conversation_context,
            ontology_context=registry.render_context(
                "health_rag", ["clinical_profile", "cancer_history", "measurement_biomarker"],
                scenario.treatment_memory,
            ),
        )
        passed = decision.needs_clarification == scenario.expected_clarification
        if scenario.expected_field_terms:
            passed = passed and any(term in decision.clarification_field.lower() for term in scenario.expected_field_terms)
        if not passed:
            failures.append(
                f"{scenario.name}: expected clarification={scenario.expected_clarification}, "
                f"got {decision.model_dump(mode='json')}"
            )
        results.append(
            {
                "name": scenario.name,
                "passed": passed,
                "expected_clarification": scenario.expected_clarification,
                "question": scenario.question,
                "decision": decision.model_dump(mode="json"),
            }
        )

    output = Path(__file__).resolve().parents[2] / "output" / output_name
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Passed {len(results) - len(failures)}/{len(results)} scenarios; saved {output}")
    if failures:
        raise AssertionError("\n".join(failures))
    return results


if __name__ == "__main__":
    run()
