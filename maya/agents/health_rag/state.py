from __future__ import annotations

from typing import List, NotRequired, TypedDict, Literal

from pydantic import BaseModel, Field
from maya.framework.integrations.biomarkers import BiomarkerRequest

from langchain.agents import AgentState as BaseAgentState


class LocalCitation(TypedDict, total=False):
    """Structured metadata describing a retrieved local-literature snippet."""

    title: str
    excerpt: str
    source_path: NotRequired[str]
    page_number: NotRequired[int]


class HealthRagState(BaseAgentState):
    """State extension for the local-literature health agent."""

    citations: NotRequired[List[LocalCitation]]
    conversation_context: NotRequired[str]
    ontology_context: NotRequired[str]
    ontology_memory_context: NotRequired[str]
    biomarker_context: NotRequired[str]
    standalone_question: NotRequired[str]
    needs_query_rewrite: NotRequired[bool]
    urgent_safety_bypass: NotRequired[bool]
    retrieval_used: NotRequired[bool]
    response_style: NotRequired[str]
    # The text actually searched against the literature. Recorded so a workflow trace can
    # show what was retrieved on, which differs from standalone_question once a
    # multi-turn question has been rewritten.
    retrieval_query: NotRequired[str]


SlotStatus = Literal["present", "resolved_from_context", "missing", "not_required"]
SlotSource = Literal["current_query", "conversation", "ontology_profile", "observation_tool", "none"]
QuestionType = Literal[
    "general_education",
    "personalized_guidance",
    "treatment_risk",
    "result_interpretation",
    "symptom_assessment",
    "other",
]


class ClinicalSlotAssessment(BaseModel):
    """Completeness assessment for one adaptive PICOT component."""

    status: SlotStatus = Field(
        default="not_required",
        description="Whether the component is present, resolved from context, materially missing, or not needed.",
    )
    value: str = Field(
        default="",
        description="The resolved component value; empty only when missing or not required.",
    )
    source: SlotSource = Field(
        default="none",
        description="Where the value was found; none when missing or not required.",
    )
    required_for_task: bool = Field(
        default=False,
        description="Whether this component is necessary for the specific question type and requested task.",
    )
    material_if_missing: bool = Field(
        default=False,
        description="Whether different values could materially change interpretation, retrieval, safety, or the answer.",
    )
    reason: str = Field(
        default="",
        description="Brief reason for the status and materiality assessment.",
    )


class PICOTAssessment(BaseModel):
    """Adaptive PICO/PICOT representation of the user's question."""

    population_or_problem: ClinicalSlotAssessment = Field(default_factory=ClinicalSlotAssessment)
    intervention_or_exposure: ClinicalSlotAssessment = Field(default_factory=ClinicalSlotAssessment)
    comparator: ClinicalSlotAssessment = Field(default_factory=ClinicalSlotAssessment)
    outcome_or_goal: ClinicalSlotAssessment = Field(default_factory=ClinicalSlotAssessment)
    timeframe: ClinicalSlotAssessment = Field(default_factory=ClinicalSlotAssessment)


class OntologyRuleCheck(BaseModel):
    """Completeness assessment for one ontology prompt rule.

    The ontology-side counterpart of `ClinicalSlotAssessment`: one record per item
    under evaluation. PICOT has five fixed components that can be named as schema
    fields; the ontology's rules are loaded at runtime and vary in number, so they are
    carried as a list instead. Without it the assessor is asked to evaluate every rule
    but given nowhere to record the result, and an unevaluated rule is indistinguishable
    from a satisfied one.

    Every supplied rule is recorded, so omitting an entire node cannot silently
    authorize an answer. Conditional rules that do not apply are satisfied.
    """

    entity_id: str = Field(
        default="",
        description=(
            "The concept_id of the ontology entity whose rule was evaluated, copied "
            "verbatim from the supplied ontology, for example "
            "'disambiguation.cardio_oncology_test_interpretation'. Never invent or "
            "paraphrase an identifier."
        ),
    )
    rule_index: int = Field(
        default=0,
        description="1-based index of the prompt rule within that entity.",
    )
    satisfied: bool = Field(
        default=True,
        description=(
            "Whether the rule's requirement is already met by the current query, the "
            "conversation, or the ontology/profile context, or its condition does not apply. False when the rule demands "
            "something the available context does not supply."
        ),
    )
    missing_requirement: str = Field(
        default="",
        description=(
            "When satisfied is false: the specific item the rule requires and the "
            "context does not supply, named concretely, for example 'name of the cardiac "
            "test and its value'. Empty when satisfied."
        ),
    )


class DisambiguationDecision(BaseModel):
    biomarker_request: BiomarkerRequest | None = None
    question_type: QuestionType = Field(
        default="other",
        description="The question type used to decide which PICOT components are required.",
    )
    urgent_safety_bypass: bool = Field(
        default=False,
        description="True when immediate safety guidance must not be delayed by clarification.",
    )
    picot: PICOTAssessment = Field(default_factory=PICOTAssessment)
    ontology_rule_checks: list[OntologyRuleCheck] = Field(
        default_factory=list,
        description=(
            "One entry for every numbered rule in every supplied disambiguation ontology node. "
            "Conditional rules whose condition does not apply are satisfied. Do not omit nodes."
        ),
    )
    missing_ontology_fields: list[str] = Field(
        default_factory=list,
        description="Ontology-required fields that remain unavailable after checking all supplied context.",
    )
    material_missing_fields: list[str] = Field(
        default_factory=list,
        description="Only missing required fields whose value could materially change the response.",
    )
    materiality_reason: str = Field(
        default="",
        description="Why the missing information could or could not change the response.",
    )
    needs_clarification: bool = Field(
        description=(
            "True only when a specific unresolved ambiguity prevents a useful, safe "
            "literature-grounded answer; otherwise false."
        )
    )
    ambiguity_reason: str = Field(
        default="",
        description="The specific missing referent, intent, or safety-critical detail when clarification is required.",
    )
    clarification_field: str = Field(
        default="",
        description=(
            "Exactly one highest-priority entry from material_missing_fields to request from the user; "
            "otherwise empty."
        ),
    )
    clarification_question: str = Field(
        default="",
        description=(
            "The final user-facing clarification question. The assessment stage leaves this empty; "
            "a dedicated wording stage populates it after one field has been selected."
        ),
    )


class ClarificationQuestion(BaseModel):
    """User-facing wording for one already-selected missing field."""

    question: str = Field(
        description="One concise question requesting only the supplied missing field.",
    )



class RewrittenQuery(BaseModel):
    """One self-contained literature query built from a multi-turn consultation."""

    query: str = Field(
        default="",
        description="A single self-contained question for clinical literature search.",
    )
