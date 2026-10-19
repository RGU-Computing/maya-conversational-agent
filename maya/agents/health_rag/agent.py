"""Ontology-guided clarification and local-literature answers for MAYA."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, dynamic_prompt, hook_config
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from maya.agents.health_rag.state import (
    HealthRagState, ClinicalSlotAssessment, PICOTAssessment, OntologyRuleCheck,
    DisambiguationDecision, ClarificationQuestion, RewrittenQuery,
)
from maya.agents.health_rag.config import HealthRagConfig, load_health_rag_config
from maya.framework.integrations.models import get_chat_model
from maya.framework.conversation.responses import load_response_policy
from maya.framework.conversation.clarification import load_disambiguation_policy, render_disambiguation_block
from maya.framework.rag.corpus import (
    chunk_snapshot_path,
    corpus_root,
    load_chunk_snapshot,
    load_corpus_documents,
    split_documents,
)
from maya.framework.rag.vectorstore import build_embeddings, build_vector_store, load_rag_vector_config


AGENT_ID = "health_rag"


class ClarificationAssessmentError(RuntimeError):
    """The completeness check failed; this is not permission to answer."""



def _field_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _material_field_match(candidate: str, material_fields: list[str]) -> str:
    candidate_key = _field_key(candidate)
    for field in material_fields:
        if candidate_key == _field_key(field):
            return field
    return ""


def _merge_ontology_rule_gaps(
    decision: DisambiguationDecision,
) -> DisambiguationDecision:
    """Fold unsatisfied ontology rules into the declared missing-field lists.

    Runs BEFORE `_normalise_disambiguation_decision`, so every consistency rule there,
    including the urgent-safety bypass, applies to the merged result. This function only
    adds candidates; it decides nothing about whether to clarify.

    A requirement must land in BOTH lists. The normaliser promotes an ontology field to
    material only when it also appears in `missing_ontology_fields`, so appending to one
    list alone would be silently discarded.

    Pure: no I/O. Missing checklists add no candidates here; the assessor validates
    their presence and attribution before this function is called.
    """

    requirements: list[str] = []
    for check in decision.ontology_rule_checks or []:
        if check.satisfied:
            continue
        requirement = (check.missing_requirement or "").strip()
        if requirement:
            requirements.append(requirement)
    if not requirements:
        return decision

    # An unmet ontology requirement takes priority over a generic PICOT gap.
    def _extend(existing: list[str]) -> list[str]:
        seen = {_field_key(item) for item in existing if item.strip()}
        merged = list(existing)
        for requirement in requirements:
            key = _field_key(requirement)
            if key and key not in seen:
                seen.add(key)
                merged.append(requirement)
        return merged

    return decision.model_copy(
        update={
            "missing_ontology_fields": _extend(decision.missing_ontology_fields),
            "material_missing_fields": _extend(decision.material_missing_fields),
            "clarification_field": _material_field_match(decision.clarification_field, requirements) or requirements[0],
        }
    )


def _normalise_disambiguation_decision(
    decision: DisambiguationDecision,
) -> DisambiguationDecision:
    """Validate field consistency without interpreting clinical language."""

    if decision.urgent_safety_bypass:
        return decision.model_copy(
            update={
                "needs_clarification": False,
                "clarification_field": "",
                "clarification_question": "",
                "material_missing_fields": [],
            }
        )

    material_fields: list[str] = []
    for slot_name in PICOTAssessment.model_fields:
        slot = getattr(decision.picot, slot_name)
        if (
            slot.status == "missing"
            and slot.required_for_task
            and slot.material_if_missing
        ):
            material_fields.append(slot_name)

    missing_ontology_fields = {
        _field_key(field): field.strip()
        for field in decision.missing_ontology_fields
        if field.strip()
    }
    explicit_material_fields: list[str] = []
    for field in decision.material_missing_fields:
        field = field.strip()
        if not field:
            continue
        key = _field_key(field)
        # An ontology requirement remains required even when its name is a PICOT slot.
        if key in missing_ontology_fields:
            material_fields.append(missing_ontology_fields[key])
            explicit_material_fields.append(missing_ontology_fields[key])
        elif key in PICOTAssessment.model_fields:
            slot_name = key
            slot = getattr(decision.picot, slot_name)
            if (
                slot.status == "missing"
                and slot.required_for_task
                and slot.material_if_missing
            ):
                material_fields.append(slot_name)
                explicit_material_fields.append(slot_name)

    material_fields = list(dict.fromkeys(material_fields))
    explicit_material_fields = list(dict.fromkeys(explicit_material_fields))
    clarification_field = _material_field_match(
        decision.clarification_field,
        material_fields,
    )
    if not clarification_field:
        clarification_field = (
            explicit_material_fields[0]
            if explicit_material_fields
            else (material_fields[0] if material_fields else "")
        )
    needs_clarification = bool(material_fields and clarification_field)
    return decision.model_copy(
        update={
            "needs_clarification": needs_clarification,
            "clarification_field": clarification_field if needs_clarification else "",
            "clarification_question": "",
            "material_missing_fields": material_fields,
        }
    )


class RetrieveLocalLiterature(AgentMiddleware[HealthRagState, Any]):
    state_schema = HealthRagState

    def __init__(self, retriever: Any, model: Any = None, settings: HealthRagConfig | None = None):
        self.settings = settings or load_health_rag_config()
        self.retriever = retriever
        self.model = model

    def _retrieval_query(self, state: HealthRagState, question: str) -> str:
        """Choose the text to search with.

        Search a self-contained question directly. With conversation, memory or observations,
        rewrite once; on failure retain that context in the fallback search text.
        """

        # before_model runs ahead of every model call in the agent loop. Reuse a query
        # already built for this invocation rather than paying for it again.
        existing = str(state.get("retrieval_query") or "").strip()
        if existing:
            return existing

        conversation_context = str(state.get("conversation_context") or "").strip()
        biomarker_context = str(state.get("biomarker_context") or "").strip()
        memory_context = str(state.get("ontology_memory_context") or "").strip()
        fallback = "\n\n".join(part for part in (conversation_context, question, biomarker_context, memory_context) if part)
        if not (conversation_context or biomarker_context or memory_context or state.get("needs_query_rewrite")) or self.model is None:
            return fallback
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "{instructions}\n\n{ontology}"),
                    (
                        "human",
                        "Recent health conversation context:\n{conversation_context}\n\n"
                        "Observation tool context:\n{biomarker_context}\n\n"
                        "Current user question:\n{question}",
                    ),
                ]
            )
            chain = prompt | self.model.with_structured_output(RewrittenQuery)
            rewritten = chain.invoke(
                {
                    "instructions": self.settings.prompts.query_rewrite,
                    "ontology": str(state.get("ontology_context") or "").strip(),
                    "conversation_context": conversation_context,
                    "biomarker_context": biomarker_context,
                    "question": question,
                }
            )
            if not isinstance(rewritten, RewrittenQuery):
                rewritten = RewrittenQuery.model_validate(rewritten)
            return rewritten.query.strip() or fallback
        except Exception:
            return fallback

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: HealthRagState, runtime: Any) -> dict[str, Any] | None:
        if state.get("urgent_safety_bypass"):
            return {"citations": [], "retrieval_query": "", "retrieval_used": False}
        question = str(state.get("standalone_question") or "").strip()
        if not question:
            for message in reversed(state.get("messages", [])):
                if isinstance(message, HumanMessage):
                    question = message.content.strip()
                    break
        if not question:
            return {"citations": []}

        retrieval_query = self._retrieval_query(state, question)
        documents = self.retriever.invoke(retrieval_query)
        citations: list[dict[str, Any]] = []
        for document in documents[:self.settings.retrieval.citation_limit]:
            excerpt = re.sub(r"\s+", " ", document.page_content).strip()
            if len(excerpt) > self.settings.retrieval.excerpt_chars:
                excerpt = excerpt[:self.settings.retrieval.excerpt_chars - 3].rsplit(" ", 1)[0] + "..."
            citations.append(
                {
                    "title": document.metadata.get("title", ""),
                    "source_path": document.metadata.get("source_path", ""),
                    "page_number": document.metadata.get("page_number", 0),
                    "excerpt": excerpt,
                }
            )
        update = {"citations": citations, "retrieval_query": retrieval_query, "retrieval_used": True}
        if not citations:
            update.update(messages=[AIMessage(content=self.settings.messages.no_evidence)], jump_to="end")
        return update


def health_rag_prompt(request: ModelRequest, settings: HealthRagConfig | None = None) -> str:
    settings = settings or load_health_rag_config()
    citations = request.state.get("citations") or []
    conversation_context = str(request.state.get("conversation_context") or "").strip()
    ontology_context = str(request.state.get("ontology_context") or "").strip()

    sections = [
        settings.prompts.urgent_answer if request.state.get("urgent_safety_bypass") else settings.prompts.answer,
        load_response_policy().health_output_instruction,
    ]
    if response_style := request.state.get("response_style"):
        sections.append(response_style)

    if conversation_context:
        sections.append(f"Recent health conversation context:\n{conversation_context}")
    if ontology_context:
        sections.append(ontology_context)
    if biomarker_context := request.state.get("biomarker_context"):
        sections.append(biomarker_context)

    context = "\n\n".join(
        f"{item.get('title', 'Local literature')} (page {int(item.get('page_number', 0)) + 1})\n{item.get('excerpt', '')}"
        for item in citations
    )
    sections.append(f"Retrieved local literature:\n{context}")
    return "\n\n".join(sections)


@dataclass(frozen=True)
class HealthRagHandle:
    name: str
    agent: Any
    ontology_block: str
    # concept_ids the ontology defines, used to reject invented rule attributions.
    ontology_entity_ids: frozenset[str] = frozenset()
    ontology_rule_counts: dict[str, int] = field(default_factory=dict)
    config: HealthRagConfig = field(default_factory=load_health_rag_config)

    def cache_scope(self, response_style: str, ontology_context: str, ontology_memory_context: str = "") -> str:
        """Reuse answers only under the same instructions, style and ontology records."""
        content = [self.config.model_dump(), self.ontology_block, ontology_context,
                   response_style, load_response_policy().health_output_instruction]
        if ontology_memory_context:
            # Exclude summaries created before records were supplied to retrieval.
            content.append(ontology_memory_context)
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def assess_clarification(
        self,
        question: str,
        *,
        model: Any,
        conversation_context: str = "",
        ontology_context: str = "",
        response_style: str = "",
        biomarker_context: str = "",
        biomarker_catalog: str = "",
        allow_biomarker_lookup: bool = True,
    ) -> DisambiguationDecision:
        if not question.strip() or model is None:
            raise ClarificationAssessmentError("A question and assessment model are required.")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{instructions}\n\nObservation lookup allowed: {allow_biomarker_lookup}\n\n{ontology}",
                ),
                (
                    "human",
                    "Recent health conversation context:\n{conversation_context}\n\n"
                    "Current user question:\n{question}",
                ),
            ]
        )
        combined_ontology = "\n\n".join(
            part for part in [self.ontology_block, ontology_context.strip(), biomarker_catalog, biomarker_context] if part
        )
        try:
            chain = prompt | model.with_structured_output(DisambiguationDecision, strict=True)
            inputs = {"instructions": self.config.prompts.assessment, "question": question,
                      "conversation_context": conversation_context, "ontology": combined_ontology,
                      "allow_biomarker_lookup": allow_biomarker_lookup}
            # One structural repair is allowed; a failed repair never authorizes an answer.
            for repair in (False, True):
                decision = DisambiguationDecision.model_validate(chain.invoke(inputs))
                try:
                    if not decision.urgent_safety_bypass and any(
                        not check.satisfied and not check.missing_requirement.strip()
                        for check in decision.ontology_rule_checks
                    ):
                        raise ValueError("An unsatisfied ontology rule must name its missing requirement.")
                    if self.ontology_entity_ids and not decision.urgent_safety_bypass:
                        if {check.entity_id for check in decision.ontology_rule_checks} != self.ontology_entity_ids or any(
                            check.rule_index < 1 for check in decision.ontology_rule_checks
                        ):
                            raise ValueError("Assessment must check every supplied ontology node using its exact ID.")
                        for entity_id in {check.entity_id for check in decision.ontology_rule_checks}:
                            if count := self.ontology_rule_counts.get(entity_id):
                                indices = [check.rule_index for check in decision.ontology_rule_checks if check.entity_id == entity_id]
                                if sorted(indices) != list(range(1, count + 1)):
                                    raise ValueError("Assessment skipped or duplicated rules in a supplied ontology node.")
                    break
                except ValueError as error:
                    if repair:
                        raise
                    inputs["ontology"] = "\n\n".join([
                        combined_ontology, self.config.prompts.assessment_repair, str(error),
                        json.dumps(self.ontology_rule_counts),
                    ])
            decision = _merge_ontology_rule_gaps(decision)
            decision = _normalise_disambiguation_decision(decision)
            if not allow_biomarker_lookup or decision.urgent_safety_bypass:
                decision = decision.model_copy(update={"biomarker_request": None})
            if (not decision.needs_clarification or
                    (decision.biomarker_request and decision.biomarker_request.subject == "current_user"
                     and decision.biomarker_request.measurement_ids and not decision.urgent_safety_bypass)):
                return decision

            wording_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "{instructions}\n\n{response_style}",
                    ),
                    (
                        "human",
                        "Selected missing field:\n{clarification_field}\n\n"
                        "Why clarification may matter:\n{reason}\n\n"
                        "Recent health conversation context:\n{conversation_context}\n\n"
                        "Current user question:\n{question}",
                    ),
                ]
            )
            wording_chain = wording_prompt | model.with_structured_output(
                ClarificationQuestion,
                strict=True,
            )
            wording = wording_chain.invoke(
                {
                    "instructions": self.config.prompts.clarification,
                    "response_style": response_style,
                    "clarification_field": decision.clarification_field,
                    "reason": next((check.missing_requirement for check in decision.ontology_rule_checks
                                    if not check.satisfied and _field_key(check.missing_requirement) == _field_key(decision.clarification_field)),
                                   decision.materiality_reason or decision.ambiguity_reason),
                    "conversation_context": conversation_context,
                    "question": question,
                }
            )
            if not isinstance(wording, ClarificationQuestion):
                wording = ClarificationQuestion.model_validate(wording)
            clarification_question = wording.question.strip()
            if not clarification_question:
                raise ValueError("Required clarification question is empty.")
            return decision.model_copy(
                update={"clarification_question": clarification_question}
            )
        except Exception as error:
            raise ClarificationAssessmentError("Could not complete the clarification check.") from error


def build(spec: dict[str, Any] | None = None) -> HealthRagHandle:
    spec = spec or {}
    unknown = set(spec) - {"config_path", "description", "disabled"}
    if unknown:
        raise ValueError(f"Unsupported HealthRAG agent fields: {sorted(unknown)}. Edit its config_path file instead.")
    settings = load_health_rag_config(spec.get("config_path"))
    root_dir = Path(__file__).resolve().parents[2]
    vector_config = load_rag_vector_config()
    documents = load_chunk_snapshot(chunk_snapshot_path(root_dir))
    if not documents and vector_config.backend != "pgvector":
        source_documents = load_corpus_documents(root_dir)
        documents = split_documents(source_documents) if source_documents else []
    if not documents and vector_config.backend != "pgvector":
        raise RuntimeError(f"No local health literature found under {corpus_root(root_dir)}.")

    embeddings = build_embeddings(vector_config)
    vector_store = build_vector_store(vector_config, embeddings, pre_delete=False)
    if vector_config.backend == "in_memory" and documents:
        vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold" if vector_config.backend == "pgvector" else "similarity",
        search_kwargs={"k": settings.retrieval.k, "score_threshold": settings.retrieval.score_threshold}
        if vector_config.backend == "pgvector" else {"k": settings.retrieval.k},
    )
    llm = get_chat_model()

    @dynamic_prompt
    def configured_prompt(request: ModelRequest) -> str:
        return health_rag_prompt(request, settings)

    agent = create_agent(
        llm,
        tools=[],
        middleware=[RetrieveLocalLiterature(retriever, llm, settings), configured_prompt],
        state_schema=HealthRagState,
        name=AGENT_ID,
    )
    ontology = load_disambiguation_policy()
    return HealthRagHandle(
        name=AGENT_ID,
        agent=agent,
        config=settings,
        ontology_entity_ids=frozenset(
            str(entity.get("concept_id") or "")
            for entity in (ontology.get("entities") or [])
            if entity.get("concept_id")
        ),
        ontology_rule_counts={entity["concept_id"]: len(entity.get("prompt_rules", []))
                              for entity in ontology.get("entities", [])},
        ontology_block=render_disambiguation_block(
            ontology,
            limit=None,
            rules_per_entity=None,
        ),
    )
