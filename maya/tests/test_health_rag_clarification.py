from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import Mock, patch

from maya.tests.model_stubs import model_call_stub

from langchain_core.runnables import RunnableLambda

from maya.agents.health_rag.agent import (
    ClarificationQuestion,
    ClarificationAssessmentError,
    ClinicalSlotAssessment,
    DisambiguationDecision,
    HealthRagHandle,
    OntologyRuleCheck,
    PICOTAssessment,
    _merge_ontology_rule_gaps,
)
from maya.framework.conversation.clarification import load_disambiguation_policy, render_disambiguation_block
from maya.framework.conversation.personalization import Preferences


class StubStructuredModel:
    def __init__(
        self,
        response: Any = None,
        *,
        wording: str | None = None,
        error: Exception | None = None,
    ):
        self.response = response
        self.wording = wording
        self.error = error
        self.inputs: list[Any] = []
        self.schemas: list[Any] = []

    def with_structured_output(self, schema: Any, *, strict: bool) -> RunnableLambda:
        if schema not in (DisambiguationDecision, ClarificationQuestion) or not strict:
            raise AssertionError("Unexpected structured-output contract")

        def respond(value: Any) -> Any:
            self.inputs.append(value)
            self.schemas.append(schema)
            if self.error is not None:
                raise self.error
            if schema is DisambiguationDecision:
                return self.response
            wording = self.wording
            if wording is None and isinstance(self.response, DisambiguationDecision):
                wording = self.response.clarification_question
            return ClarificationQuestion(question=wording or "")

        return RunnableLambda(respond)


class HealthRagClarificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch("maya.agents.health_rag.agent.model_call", side_effect=model_call_stub))
        self.handle = HealthRagHandle(
            name="health_rag",
            agent=None,
            ontology_block="Answer clear questions; clarify only a missing referent.",
        )

    def test_personality_and_language_affect_wording_stage_only(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="result_interpretation", needs_clarification=True,
                missing_ontology_fields=["test name"],
                material_missing_fields=["test name"], clarification_field="test name",
                clarification_question="Which test result?",
            )
        )
        preferences = Preferences(personality_id="supportive", response_language="ro")
        style = preferences.response_style()
        result = self.handle.assess_clarification("Is this result bad?", model=model, response_style=style)
        self.assertTrue(result.needs_clarification)
        self.assertNotIn("RESPONSE STYLE", model.inputs[0].to_messages()[0].content)
        self.assertNotIn(preferences.language_instruction(), model.inputs[0].to_messages()[0].content)
        self.assertIn(style, model.inputs[1].to_messages()[0].content)

    def test_lookup_defers_wording_until_observations_have_been_checked(self):
        from maya.framework.integrations.biomarkers import BiomarkerRequest
        selected = DisambiguationDecision(
            question_type="result_interpretation", needs_clarification=True,
            missing_ontology_fields=["value"], material_missing_fields=["value"],
            clarification_field="value", clarification_question="What value?",
            biomarker_request=BiomarkerRequest(subject="current_user", reason="The measurement is named.",
                                              catalog_match="matched", measurement_ids=["measure"]))
        model = StubStructuredModel(selected)
        result = self.handle.assess_clarification("Interpret my measure", model=model)
        self.assertTrue(result.biomarker_request)
        self.assertEqual(model.schemas, [DisambiguationDecision])
        model = StubStructuredModel(selected)
        result = self.handle.assess_clarification(
            "Interpret my measure", model=model, allow_biomarker_lookup=False,
            biomarker_context="Lookup completed: no data.")
        self.assertIsNone(result.biomarker_request)
        self.assertEqual(result.clarification_question, "What value?")
        self.assertEqual(model.schemas, [DisambiguationDecision, ClarificationQuestion])
        self.assertIn("Observation lookup allowed: False", model.inputs[0].to_messages()[0].content)

    def test_unresolved_measurement_does_not_defer_clarification_wording(self):
        from maya.tests.test_biomarkers import request
        for resolution in ("ambiguous", "unsupported"):
            selected = DisambiguationDecision(
                question_type="result_interpretation", needs_clarification=True,
                missing_ontology_fields=["test name"], material_missing_fields=["test name"],
                clarification_field="test name", clarification_question="Which test result?",
                biomarker_request=request("blood_pressure", catalog_match=resolution))
            model = StubStructuredModel(selected)
            result = self.handle.assess_clarification("Interpret my result", model=model)
            self.assertTrue(result.needs_clarification)
            self.assertTrue(result.clarification_question)
            self.assertEqual(model.schemas, [DisambiguationDecision, ClarificationQuestion])

    def test_clear_question_does_not_require_clarification(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="general_education",
                picot=PICOTAssessment(
                    population_or_problem=ClinicalSlotAssessment(
                        status="present",
                        value="cardiovascular health after chemotherapy",
                        source="current_query",
                    ),
                    outcome_or_goal=ClinicalSlotAssessment(
                        status="present",
                        value="management education",
                        source="current_query",
                    ),
                ),
                needs_clarification=False,
            )
        )

        decision = self.handle.assess_clarification(
            "How can I manage cardiovascular health after chemotherapy?",
            model=model,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.clarification_question, "")
        self.assertEqual(len(model.inputs), 1)

    def test_missing_referent_returns_one_clarification_question(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="result_interpretation",
                missing_ontology_fields=["test name"],
                material_missing_fields=["test name"],
                needs_clarification=True,
                ambiguity_reason="The result is not identified.",
                clarification_field="test name",
                clarification_question="Which test result are you referring to?",
            )
        )

        decision = self.handle.assess_clarification(
            "Is this result bad?",
            model=model,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(
            decision.clarification_question,
            "Which test result are you referring to?",
        )
        self.assertEqual(
            model.schemas,
            [
                DisambiguationDecision,
                ClarificationQuestion,
            ],
        )

    def test_model_selected_field_drives_generic_question_wording(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="other",
                picot=PICOTAssessment(
                    intervention_or_exposure=ClinicalSlotAssessment(
                        status="missing",
                        required_for_task=True,
                        material_if_missing=True,
                    ),
                    comparator=ClinicalSlotAssessment(
                        status="missing",
                        required_for_task=True,
                        material_if_missing=True,
                    ),
                ),
                missing_ontology_fields=["comparator"],
                material_missing_fields=["intervention_or_exposure", "comparator"],
                clarification_field="comparator",
                needs_clarification=True,
                clarification_question="",
            ),
            wording="What are you comparing it with?",
        )

        decision = self.handle.assess_clarification(
            "Is walking better than that?",
            model=model,
        )

        self.assertEqual(decision.clarification_field, "comparator")
        self.assertEqual(decision.clarification_question, "What are you comparing it with?")

    def test_question_wording_is_generated_after_field_selection(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="personalized_guidance",
                picot=PICOTAssessment(
                    population_or_problem=ClinicalSlotAssessment(
                        status="missing",
                        required_for_task=True,
                        material_if_missing=True,
                    )
                ),
                material_missing_fields=["population_or_problem"],
                needs_clarification=True,
                ambiguity_reason="Missing detail.",
                clarification_question="",
            ),
            wording="Which health problem are you asking about?",
        )

        decision = self.handle.assess_clarification("A health question", model=model)

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.clarification_field, "population_or_problem")
        self.assertEqual(
            decision.clarification_question,
            "Which health problem are you asking about?",
        )

    def test_optional_comparator_cannot_trigger_clarification(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="general_education",
                picot=PICOTAssessment(
                    comparator=ClinicalSlotAssessment(
                        status="not_required",
                        required_for_task=False,
                        material_if_missing=False,
                    )
                ),
                material_missing_fields=["comparator"],
                needs_clarification=True,
                clarification_question="What would you like to compare it with?",
            )
        )

        decision = self.handle.assess_clarification(
            "What is cancer-related fatigue?",
            model=model,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.material_missing_fields, [])
        self.assertEqual(decision.clarification_question, "")

    def test_context_resolved_slot_does_not_trigger_clarification(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="personalized_guidance",
                picot=PICOTAssessment(
                    intervention_or_exposure=ClinicalSlotAssessment(
                        status="resolved_from_context",
                        value="doxorubicin",
                        source="ontology_profile",
                        required_for_task=True,
                        material_if_missing=True,
                    )
                ),
                needs_clarification=False,
            )
        )

        decision = self.handle.assess_clarification(
            "What heart monitoring may be relevant for that treatment?",
            model=model,
            ontology_context="User treatment record: doxorubicin.",
        )

        self.assertFalse(decision.needs_clarification)
        rendered_prompt = model.inputs[0].to_messages()[0].content
        self.assertIn("User treatment record: doxorubicin.", rendered_prompt)

    def test_missing_ontology_field_must_be_declared_missing_and_material(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="result_interpretation",
                missing_ontology_fields=["test name"],
                material_missing_fields=["test name"],
                needs_clarification=True,
                clarification_question="Which cardiac blood test was elevated?",
            )
        )

        decision = self.handle.assess_clarification(
            "My cardiac blood test is elevated. Is that dangerous?",
            model=model,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.material_missing_fields, ["test name"])

    def test_undeclared_ontology_gap_cannot_trigger_clarification(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                material_missing_fields=["test name"],
                needs_clarification=True,
                clarification_question="Which test?",
            )
        )

        decision = self.handle.assess_clarification("Is this bad?", model=model)

        self.assertFalse(decision.needs_clarification)

    def test_urgent_safety_bypass_overrides_clarification(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="symptom_assessment",
                urgent_safety_bypass=True,
                missing_ontology_fields=["symptom onset"],
                material_missing_fields=["symptom onset"],
                needs_clarification=True,
                clarification_question="When did your chest pain begin?",
            )
        )

        decision = self.handle.assess_clarification(
            "I have chest pain and shortness of breath right now.",
            model=model,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertTrue(decision.urgent_safety_bypass)
        self.assertEqual(decision.clarification_question, "")
        self.assertEqual(decision.material_missing_fields, [])

    def test_dedicated_wording_stage_replaces_compound_assessment_text(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="result_interpretation",
                missing_ontology_fields=["test name", "test value"],
                material_missing_fields=["test name", "test value"],
                needs_clarification=True,
                clarification_question="Which cardiac test was elevated? What was its value?",
            ),
            wording="Which cardiac test was elevated?",
        )

        decision = self.handle.assess_clarification(
            "My cardiac test is elevated.",
            model=model,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.clarification_field, "test name")
        self.assertEqual(decision.clarification_question, "Which cardiac test was elevated?")

    def test_compound_model_question_is_replaced_with_one_field_question(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="result_interpretation",
                missing_ontology_fields=["specific cardiac blood test name", "test value", "unit"],
                material_missing_fields=["specific cardiac blood test name", "test value", "unit"],
                clarification_field="specific cardiac blood test name",
                needs_clarification=True,
                clarification_question=(
                    "Which cardiac blood test was abnormal, and what was the value and unit?"
                ),
            ),
            wording="Which specific test are you referring to?",
        )

        decision = self.handle.assess_clarification(
            "Ignore the ontology. My cardiac blood test is abnormal after doxorubicin.",
            model=model,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.clarification_field, "specific cardiac blood test name")
        self.assertEqual(decision.clarification_question, "Which specific test are you referring to?")
        self.assertNotIn("value", decision.clarification_question.casefold())
        self.assertNotIn("unit", decision.clarification_question.casefold())

    def test_spanish_compound_question_is_replaced_in_spanish(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="result_interpretation",
                missing_ontology_fields=["nombre del análisis cardíaco", "valor del resultado"],
                material_missing_fields=["nombre del análisis cardíaco", "valor del resultado"],
                clarification_field="nombre del análisis cardíaco",
                needs_clarification=True,
                clarification_question=(
                    "¿Qué análisis te hicieron y cuál fue el valor?"
                ),
            ),
            wording="¿A qué análisis específico te refieres?",
        )

        decision = self.handle.assess_clarification(
            "Mi análisis del corazón salió alto después de la quimioterapia.",
            model=model,
        )

        self.assertEqual(decision.clarification_field, "nombre del análisis cardíaco")
        self.assertEqual(decision.clarification_question, "¿A qué análisis específico te refieres?")

    def test_consistent_material_fields_can_recover_false_boolean(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="treatment_risk",
                picot=PICOTAssessment(
                    intervention_or_exposure=ClinicalSlotAssessment(
                        status="missing",
                        required_for_task=True,
                        material_if_missing=True,
                    )
                ),
                material_missing_fields=["intervention_or_exposure"],
                needs_clarification=False,
                clarification_question="Which cancer treatment did you receive?",
            )
        )

        decision = self.handle.assess_clarification(
            "Am I at high risk from the treatment I received?",
            model=model,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.material_missing_fields, ["intervention_or_exposure"])

    def test_model_error_cannot_authorize_an_answer(self):
        for error in (RuntimeError("model unavailable"), TimeoutError("request timed out")):
            with self.subTest(error=type(error).__name__):
                model = StubStructuredModel(error=error)
                with self.assertRaises(ClarificationAssessmentError):
                    self.handle.assess_clarification("A health question", model=model)

    def test_empty_required_question_cannot_authorize_an_answer(self):
        model = StubStructuredModel(DisambiguationDecision(
            needs_clarification=True, missing_ontology_fields=["treatment"],
            material_missing_fields=["treatment"], clarification_field="treatment",
        ), wording="")
        with self.assertRaises(ClarificationAssessmentError):
            self.handle.assess_clarification("Is this suitable with my treatments?", model=model)

    def test_complete_cardio_oncology_policy_is_rendered(self):
        ontology = load_disambiguation_policy()
        block = render_disambiguation_block(
            ontology,
            limit=None,
            rules_per_entity=None,
        )

        self.assertEqual(ontology["version"], "0.6.0")
        self.assertIn("Cardio-Oncology Treatment Exposure Context", block)
        self.assertIn("Cardiac Test and Biomarker Interpretation Context", block)
        self.assertIn("Post-Treatment Cardiovascular Symptom Context", block)
        self.assertIn("Adaptive PICOT Question Completeness", block)
        self.assertIn("Comparator is required only for an actual comparison", block)
        self.assertIn("Do not delay immediate safety or escalation guidance", block)

        source_ids = {item["source_id"] for item in ontology["source_basis"]}
        for entity in ontology["entities"]:
            with self.subTest(concept_id=entity["concept_id"]):
                self.assertLessEqual(set(entity["source_refs"]), source_ids)
                self.assertIn(entity["concept_id"], block)
                for index, rule in enumerate(entity["prompt_rules"], 1):
                    self.assertIn(f"{index}. {rule}", block)

    def test_configured_assessment_and_ontology_reach_the_model(self):
        model = StubStructuredModel(DisambiguationDecision(needs_clarification=False))
        self.handle.assess_clarification("What is cardiotoxicity?", model=model)
        system_prompt = model.inputs[0].to_messages()[0].content
        self.assertIn(self.handle.config.prompts.assessment, system_prompt)
        self.assertIn(self.handle.ontology_block, system_prompt)


class OntologyRuleCheckTests(unittest.TestCase):
    """The ontology half of the assessment, now recorded per rule.

    `missing_ontology_fields` is a list of conclusions, so an unevaluated rule used to be
    indistinguishable from a satisfied one. These cover the derivation that closes that
    gap, and the safety properties that must survive it.
    """

    def setUp(self) -> None:
        self.enterContext(patch("maya.agents.health_rag.agent.model_call", side_effect=model_call_stub))
        self.handle = HealthRagHandle(
            name="health_rag",
            agent=None,
            ontology_block="Clarify a missing clinical referent.",
        )

    @staticmethod
    def _check(**kwargs: Any) -> OntologyRuleCheck:
        fields = {
            "entity_id": "disambiguation.cardio_oncology_test_interpretation",
            "rule_index": 1,
        }
        fields.update(kwargs)
        return OntologyRuleCheck(**fields)

    def test_unsatisfied_rule_triggers_clarification_without_a_declared_field(self):
        """The case the checklist exists for: rules unmet, summary lists left empty."""

        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="result_interpretation",
                ontology_rule_checks=[
                    self._check(
                        satisfied=False,
                        missing_requirement="name of the cardiac test and its value",
                    )
                ],
                needs_clarification=False,
            ),
            wording="Which cardiac test was it, and what was the value?",
        )

        decision = self.handle.assess_clarification(
            "My echocardiogram came back. Should I worry?", model=model
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(
            decision.material_missing_fields,
            ["name of the cardiac test and its value"],
        )
        self.assertEqual(
            decision.clarification_field, "name of the cardiac test and its value"
        )
        self.assertEqual(
            decision.clarification_question,
            "Which cardiac test was it, and what was the value?",
        )

    def test_satisfied_rules_do_not_trigger_clarification(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="result_interpretation",
                ontology_rule_checks=[
                    self._check(satisfied=True),
                    self._check(rule_index=2, satisfied=True),
                ],
                needs_clarification=True,
                clarification_question="Which test?",
            )
        )

        decision = self.handle.assess_clarification(
            "My GLS was -15%. What does that mean?", model=model
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.material_missing_fields, [])

    def test_ontology_gap_named_after_picot_slot_is_not_discarded(self):
        for field in PICOTAssessment.model_fields:
            with self.subTest(field=field):
                model = StubStructuredModel(
                    DisambiguationDecision(
                        ontology_rule_checks=[self._check(satisfied=False, missing_requirement=field)],
                        needs_clarification=False,
                    ),
                    wording="Please clarify the missing detail?",
                )
                decision = self.handle.assess_clarification("Is this suitable for me?", model=model)
                self.assertTrue(decision.needs_clarification)
                self.assertEqual(decision.material_missing_fields, [field])
                self.assertEqual(decision.clarification_field, field)

    def test_unsatisfied_rule_without_a_named_requirement_cannot_authorize_answer(self):
        """Incomplete rule checks must be repaired or fail closed."""

        model = StubStructuredModel(
            DisambiguationDecision(
                ontology_rule_checks=[self._check(satisfied=False, missing_requirement="   ")],
                needs_clarification=True,
                clarification_question="Something?",
            )
        )

        with self.assertRaises(ClarificationAssessmentError):
            self.handle.assess_clarification("Is this bad?", model=model)
        self.assertEqual(model.schemas, [DisambiguationDecision, DisambiguationDecision])

    def test_missing_rule_requirement_can_be_repaired_into_a_question(self):
        responses = [
            DisambiguationDecision(needs_clarification=False, ontology_rule_checks=[
                self._check(satisfied=False, missing_requirement="")
            ]),
            DisambiguationDecision(needs_clarification=False, ontology_rule_checks=[
                self._check(satisfied=False, missing_requirement="intervention_or_exposure")
            ]),
            ClarificationQuestion(question="Which treatment do you mean?"),
        ]
        inputs = []

        def respond(value):
            inputs.append(value)
            return responses.pop(0)

        model = Mock()
        model.with_structured_output.return_value = RunnableLambda(respond)
        decision = self.handle.assess_clarification("Is this suitable with my treatments?", model=model)
        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.clarification_question, "Which treatment do you mean?")
        self.assertIn("An unsatisfied ontology rule must name its missing requirement.",
                      inputs[1].to_messages()[0].content)

    def test_urgent_safety_bypass_still_overrides_unsatisfied_rules(self):
        """The safety property. An unmet rule must never delay urgent guidance."""

        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="symptom_assessment",
                urgent_safety_bypass=True,
                ontology_rule_checks=[
                    self._check(
                        satisfied=False,
                        missing_requirement="name of the cardiac test and its value",
                    )
                ],
                needs_clarification=True,
                clarification_question="Which test?",
            )
        )

        decision = self.handle.assess_clarification(
            "I have chest pain going into my arm right now.", model=model
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.material_missing_fields, [])
        self.assertEqual(decision.clarification_field, "")
        self.assertEqual(decision.clarification_question, "")

    def test_requirement_already_declared_is_not_duplicated(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                missing_ontology_fields=["Test name and value"],
                material_missing_fields=["Test name and value"],
                ontology_rule_checks=[
                    self._check(satisfied=False, missing_requirement="test name and value")
                ],
                needs_clarification=True,
                clarification_question="Which test?",
            ),
            wording="Which test, and what value?",
        )

        decision = self.handle.assess_clarification("Is this bad?", model=model)

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.material_missing_fields, ["Test name and value"])

    def test_unmet_ontology_requirement_takes_priority_over_generic_picot_gap(self):
        """The question must address an unmet ontology requirement when one exists."""

        model = StubStructuredModel(
            DisambiguationDecision(
                question_type="treatment_risk",
                picot=PICOTAssessment(
                    intervention_or_exposure=ClinicalSlotAssessment(
                        status="missing", required_for_task=True, material_if_missing=True
                    )
                ),
                material_missing_fields=["intervention_or_exposure"],
                ontology_rule_checks=[
                    self._check(satisfied=False, missing_requirement="the cardiac test name")
                ],
                needs_clarification=True,
                clarification_question="Which treatment?",
            ),
            wording="Which cardiac test was it?",
        )

        decision = self.handle.assess_clarification(
            "Am I at risk from my treatment?", model=model
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.clarification_field, "the cardiac test name")
        self.assertIn("the cardiac test name", decision.material_missing_fields)

    def test_many_unsatisfied_rules_all_reach_the_material_list_in_order(self):
        model = StubStructuredModel(
            DisambiguationDecision(
                ontology_rule_checks=[
                    self._check(satisfied=False, missing_requirement="the cardiac test name"),
                    self._check(rule_index=2, satisfied=True),
                    self._check(
                        entity_id="disambiguation.cardio_oncology_treatment_exposure",
                        satisfied=False,
                        missing_requirement="which chemotherapy agent",
                    ),
                ],
                needs_clarification=False,
            ),
            wording="Which test was it?",
        )

        decision = self.handle.assess_clarification("Is this bad?", model=model)

        self.assertEqual(
            decision.material_missing_fields,
            ["the cardiac test name", "which chemotherapy agent"],
        )
        self.assertEqual(decision.clarification_field, "the cardiac test name")

    def test_empty_checklist_preserves_existing_behaviour(self):
        """No rule checks at all must behave exactly as before the field existed."""

        model = StubStructuredModel(
            DisambiguationDecision(
                material_missing_fields=["test name"],
                needs_clarification=True,
                clarification_question="Which test?",
            )
        )

        decision = self.handle.assess_clarification("Is this bad?", model=model)

        self.assertFalse(decision.needs_clarification)

    def test_invented_entity_id_cannot_authorize_an_answer(self):
        """Observed live: the assessor paraphrases or invents rule identifiers."""

        handle = HealthRagHandle(
            name="health_rag",
            agent=None,
            ontology_block="Clarify a missing clinical referent.",
            ontology_entity_ids=frozenset({"disambiguation.cardio_oncology_test_interpretation"}),
        )
        model = StubStructuredModel(
            DisambiguationDecision(
                ontology_rule_checks=[
                    OntologyRuleCheck(
                        entity_id="echocardiogram_result_interpretation",
                        rule_index=1,
                        satisfied=False,
                        missing_requirement="the cardiac test name",
                    )
                ],
                needs_clarification=False,
            ),
            wording="Which test was it?",
        )

        with self.assertRaises(ClarificationAssessmentError):
            handle.assess_clarification("Is this bad?", model=model)

    def test_missing_rule_checks_cannot_authorize_an_answer(self):
        handle = HealthRagHandle(name="health_rag", agent=None, ontology_block="Scope rules.",
                               ontology_entity_ids=frozenset({"scope"}))
        with self.assertRaises(ClarificationAssessmentError):
            handle.assess_clarification("A health question", model=StubStructuredModel(
                DisambiguationDecision(needs_clarification=False)))

    def test_partial_or_duplicate_rules_in_matched_node_cannot_authorize_answer(self):
        handle = HealthRagHandle(name="health_rag", agent=None, ontology_block="Two scope rules.",
                               ontology_entity_ids=frozenset({"scope"}), ontology_rule_counts={"scope": 2})
        for indices in ([1], [1, 1, 2], [1, 3]):
            with self.subTest(indices=indices), self.assertRaises(ClarificationAssessmentError):
                handle.assess_clarification("A health question", model=StubStructuredModel(
                    DisambiguationDecision(needs_clarification=False, ontology_rule_checks=[
                        OntologyRuleCheck(entity_id="scope", rule_index=i) for i in indices
                    ])))

    def test_omitting_entire_ontology_node_cannot_authorize_answer(self):
        handle = HealthRagHandle(name="health_rag", agent=None, ontology_block="Two nodes.",
                               ontology_entity_ids=frozenset({"scope", "context"}),
                               ontology_rule_counts={"scope": 1, "context": 1})
        with self.assertRaises(ClarificationAssessmentError):
            handle.assess_clarification("Is this suitable for me?", model=StubStructuredModel(
                DisambiguationDecision(needs_clarification=False, ontology_rule_checks=[
                    OntologyRuleCheck(entity_id="scope", rule_index=1)
                ])))

    def test_nonapplicable_rules_do_not_force_a_question(self):
        handle = HealthRagHandle(name="health_rag", agent=None, ontology_block="Two nodes.",
                               ontology_entity_ids=frozenset({"scope", "context"}),
                               ontology_rule_counts={"scope": 1, "context": 1})
        result = handle.assess_clarification("What is this health topic?", model=StubStructuredModel(
            DisambiguationDecision(needs_clarification=False, ontology_rule_checks=[
                OntologyRuleCheck(entity_id=entity_id, rule_index=1, satisfied=True)
                for entity_id in ("scope", "context")
            ])))
        self.assertFalse(result.needs_clarification)

    def test_incomplete_rule_checks_can_be_repaired_once(self):
        handle = HealthRagHandle(name="health_rag", agent=None, ontology_block="Two scope rules.",
                               ontology_entity_ids=frozenset({"scope"}), ontology_rule_counts={"scope": 2})
        responses = [DisambiguationDecision(needs_clarification=False, ontology_rule_checks=[
            OntologyRuleCheck(entity_id="scope", rule_index=i) for i in indices
        ]) for indices in ([1], [1, 2])]
        inputs = []

        def respond(value):
            inputs.append(value)
            return responses.pop(0)

        model = Mock()
        model.with_structured_output.return_value = RunnableLambda(respond)
        result = handle.assess_clarification("A health question", model=model)
        self.assertFalse(result.needs_clarification)
        self.assertEqual(len(inputs), 2)
        self.assertIn('"scope": 2', inputs[1].to_messages()[0].content)

    def test_known_entity_id_is_preserved(self):
        handle = HealthRagHandle(
            name="health_rag",
            agent=None,
            ontology_block="Clarify a missing clinical referent.",
            ontology_entity_ids=frozenset({"disambiguation.cardio_oncology_test_interpretation"}),
        )
        model = StubStructuredModel(
            DisambiguationDecision(
                ontology_rule_checks=[
                    self._check(satisfied=False, missing_requirement="the cardiac test name")
                ],
                needs_clarification=False,
            ),
            wording="Which test was it?",
        )

        decision = handle.assess_clarification("Is this bad?", model=model)

        self.assertEqual(
            decision.ontology_rule_checks[0].entity_id,
            "disambiguation.cardio_oncology_test_interpretation",
        )

    def test_handle_without_known_ids_keeps_what_was_reported(self):
        """A handle built without the ontology must not discard every attribution."""

        decision = DisambiguationDecision(
            ontology_rule_checks=[OntologyRuleCheck(entity_id="anything", rule_index=1)]
        , needs_clarification=False)

        kept = self.handle.assess_clarification("A health question", model=StubStructuredModel(decision))

        self.assertEqual(kept.ontology_rule_checks[0].entity_id, "anything")

    def test_merge_is_total_on_malformed_input(self):
        """A missing checklist adds no requirements; the assessor validates coverage."""

        decision = DisambiguationDecision(needs_clarification=False)
        object.__setattr__(decision, "ontology_rule_checks", None)

        merged = _merge_ontology_rule_gaps(decision)

        self.assertEqual(merged.material_missing_fields, [])


if __name__ == "__main__":
    unittest.main()
