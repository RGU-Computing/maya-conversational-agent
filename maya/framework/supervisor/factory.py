import uuid
import json
import hashlib
import logging
import re
from datetime import datetime
from typing import Optional, Literal, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
import importlib

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from langgraph.graph import StateGraph, END, START
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from psycopg import Error as PostgresError

from maya.agents.config import load_agent_specs
from maya.agents.health_rag.agent import ClarificationAssessmentError, DisambiguationDecision
from maya.agents.health_rag.config import EMERGENCY_CONTACT_MARKER
from maya.framework.integrations.biomarkers import BiomarkerRequest, load_biomarker_service
from maya.framework.memory import DeterministicMemoryEmbeddings, get_postgres_memory
from maya.framework.memory.ontology import OntologyMemoryService, OntologyMemoryNotReady
from maya.framework.integrations.models import get_chat_model, model_call
from maya.framework.integrations.privacy import privacy_config
from maya.framework.conversation.personalization import preferences_from_config
from maya.framework.conversation.history import recent_dialogue
from maya.framework.conversation.responses import health_presentation, localize_messages, localize_text, response_policy
from maya.framework.ontology import get_ontology_registry
from maya.framework.supervisor.state import SupervisorState
from dotenv import load_dotenv
from .prompts import (
    PROFILE_UPDATE_INSTRUCTION,
    MEMORY_UPDATE_INSTRUCTION,
    MEMORY_VERIFICATION_INSTRUCTION,
    SUPERVISOR_SYSTEM_MESSAGE,
    build_health_route_guard_prompt,
    system,
)
from langgraph.types import interrupt
from langgraph.config import get_stream_writer
from langgraph.func import task

load_dotenv()

logger = logging.getLogger(__name__)
model = get_chat_model()


def _load_agents() -> Tuple[List[Any], Dict[str, str]]:
    specs = load_agent_specs()
    agents: List[Any] = []
    descriptions: Dict[str, str] = {}
    for agent_id, spec in specs.items():
        if spec.get("disabled"):
            continue
        module = importlib.import_module(f"maya.agents.{agent_id}.agent")
        agent = module.build(spec)
        name = str(getattr(agent, "name"))
        if name in descriptions:
            raise RuntimeError(f"Duplicate agent name '{name}' detected.")
        agents.append(agent)
        descriptions[name] = (spec.get("description") or "").strip()
    if not agents:
        raise RuntimeError("No enabled agents configured.")
    return agents, descriptions


class SupervisorRouteDecision(BaseModel):
    """Validated supervisor route decision for the current turn."""

    reason: str = Field(default="", description=(
        "Identify the user's actual request or assertion and the applicable routing rule before choosing a route. "
        "Distinguish recall, profile/experience updates, clinical fact-only updates, recorded-value readout and health advice."
    ))
    question_subject: Literal["current_user", "other", "general", "unknown"] = Field(
        default="unknown",
        description=("Whose health or measurements the question concerns: current_user for the signed-in user, "
                     "other for another person, general for educational questions about no particular person, "
                     "unknown when unresolved or not a health request. Identify the subject from the question "
                     "and dialogue, never from the existence of stored user records."),
    )
    route_type: Literal[
        "direct_answer",
        "user_profile",
        "general_memory",
        "health_rag",
        "biomarkers",
        "phyxio_exercise_agent",
    ] = Field(
        description=(
            "Choose exactly one route. direct_answer also covers messages that only assert a clinical fact "
            "supported by a selected writable ontology binding; the memory finalizer stores it."
        )
    )
    biomarker_request: BiomarkerRequest | None = None
    selected_ontology_models: list[str] = Field(
        description="Every relevant model_id from the supplied ontology catalog; use an empty list when none apply."
    )


class HealthRouteGuardDecision(BaseModel):
    """Validation pass to prevent specialist requests from falling through to direct answers."""

    route_type: Literal["direct_answer", "health_rag", "phyxio_exercise_agent"] = Field(
        description="Choose a specialist route for health or exercise requests; otherwise direct_answer."
    )
    exercise_reference: str = Field(
        default="", description="For an exercise override, quote exact dialogue text explicitly naming an exercise, routine or exercise operation. Performance alone does not name an activity. Return an empty string if absent or for other routes."
    )

class Memory(BaseModel):
    content: str = Field(
        description="A new or corrected personal fact explicitly asserted by the current user; preserve its meaning and uncertainty."
    )
    memory_id: str | None = Field(
        default=None,
        description="Existing memory ID to correct, or null for a new memory. Never invent an existing ID.",
    )


class MemoryUpdate(BaseModel):
    memories: list[Memory] = Field(
        default_factory=list,
        description="Only new or corrected user facts; empty for recall, questions, speculation or requests not to save.",
    )


memory_extractor = model_call(model, MemoryUpdate, strict=True)

class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interests that the user has", 
        default_factory=list
    )


class ProfileUpdate(BaseModel):
    profile: Profile | None = Field(
        default=None,
        description="Complete updated profile, preserving existing facts; null when no new user assertion supports a change.",
    )


profile_extractor = model_call(model, ProfileUpdate, strict=True)


class MemoryVerification(BaseModel):
    reason: str = Field(default="", description="Brief explanation of which user assertions support the proposed changes, or why they must be rejected.")
    supported: bool = Field(
        description="True only when the latest user message explicitly supports the entire proposed change."
    )



def build_supervisor(*, biomarker_service=None):
    biomarkers = biomarker_service or load_biomarker_service()

    def clinical_context(user_id, agent_id, model_ids, subject):
        if agent_id == "health_rag" and subject != "current_user":
            return ontology_registry.render_context(agent_id, model_ids), ""
        return ontology_memory.load_context(user_id=user_id, agent_id=agent_id, model_ids=model_ids)

    def memory_unavailable(selected_models, config):
        return {
            "messages": [AIMessage(
                content=localize_text(ontology_registry.runtime["memory_policy"]["recovery_unavailable_text"],
                                      preferences_from_config(config).response_language),
                additional_kwargs={"source": "ontology_memory", "memory_status": "pending",
                                   "selected_ontology_models": selected_models},
            )],
            "route_type": "direct_answer", "selected_ontology_models": selected_models,
            "resolved_user_input": "", "biomarker_results": [],
        }
    def _latest_user_message(messages: list[Any]) -> HumanMessage | None:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message
        return None

    def _recent_health_context(messages: list[Any]) -> str:
        prior_user = ""
        prior_health_answer = ""

        last_user_index = max(
            (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
            default=-1,
        )
        if last_user_index <= 0:
            return ""

        for message in reversed(messages[:last_user_index]):
            if not prior_health_answer and isinstance(message, AIMessage):
                extra = getattr(message, "additional_kwargs", {}) or {}
                trace = extra.get("workflow_trace") or []
                if any(
                    isinstance(item, dict) and item.get("stage") == "health_rag"
                    for item in trace
                ):
                    text = getattr(message, "content", "").strip()
                    if text:
                        prior_health_answer = text
                        continue

            if not prior_user and isinstance(message, HumanMessage):
                text = getattr(message, "content", "").strip()
                if text:
                    prior_user = text

            if prior_user and prior_health_answer:
                break

        parts: list[str] = []
        if prior_user:
            parts.append(f"Previous user health question: {prior_user}")
        if prior_health_answer:
            parts.append(f"Previous health answer: {prior_health_answer}")
        return "\n".join(parts)

    def supervisor(state: SupervisorState, config: RunnableConfig, store: BaseStore):

        """Load memories from the store and use them to personalize the chatbot's response."""
        
        # Get the user ID from the config
        last_message = state["messages"][-1] if state["messages"] else None
        if isinstance(last_message, ToolMessage) and (last_message.additional_kwargs or {}).get("source") == "health_rag":
            extra = last_message.additional_kwargs or {}
            workflow_trace = extra.get("workflow_trace")
            kwargs = {"workflow_trace": workflow_trace} if isinstance(workflow_trace, list) else {}
            return {"messages": [AIMessage(content=last_message.content, additional_kwargs=kwargs)]}

        user_id = config["configurable"]["user_id"]

        # Retrieve profile memory from the store
        namespace = ("profile", user_id)
        memories = store.search(namespace)
        if memories:
            user_profile = memories[0].value
        else:
            user_profile = None
        identity = {key: (user_profile or {}).get(key) for key in ("name", "location")}
        identity["patient_id"] = biomarkers.config.patient_map.get(user_id)
        config = privacy_config(config, identity)
        
        # Retrieve general memory from the store
        namespace = ("memory", user_id)
        memories = store.search(namespace)
        if memories:
            general_memory = "\n".join(json.dumps(mem.value, ensure_ascii=False) for mem in memories)
        else:
            general_memory = "No general memories."

        system_msg = "\n\n".join(
            [
                SUPERVISOR_SYSTEM_MESSAGE.format(
                    user_profile=json.dumps(user_profile, ensure_ascii=False),
                    profile_schema=json.dumps(Profile.model_json_schema()),
                    general_memory=general_memory,
                    time=datetime.now().isoformat(),
                ),
                ontology_registry.catalog_for_agent("supervisor"),
                biomarkers.catalog_context(),
            ]
        )

        latest_user = _latest_user_message(state["messages"])
        dialogue = recent_dialogue(state["messages"])
        routing_messages = [SystemMessage(content=system_msg), *dialogue]

        decision = model_call(model, SupervisorRouteDecision, strict=True).invoke(
            routing_messages, config=config
        )

        if (
            isinstance(decision, SupervisorRouteDecision)
            and decision.route_type == "direct_answer"
            and latest_user is not None
        ):
            health_guard = model_call(model, HealthRouteGuardDecision, strict=True).invoke(
                [
                    SystemMessage(content=build_health_route_guard_prompt(
                        ontology_registry.catalog_for_agent("supervisor"))),
                    *dialogue,
                ], config=config,
            )
            if health_guard.route_type == "phyxio_exercise_agent" and not (
                health_guard.exercise_reference.strip()
                and any(health_guard.exercise_reference in str(message.content) for message in dialogue)
            ):
                health_guard = health_guard.model_copy(update={"route_type": "direct_answer"})
            if isinstance(health_guard, HealthRouteGuardDecision) and health_guard.route_type != "direct_answer":
                decision = decision.model_copy(update={"route_type": health_guard.route_type})

        route_type = decision.route_type
        agent_id = route_type if route_type in {"health_rag", "phyxio_exercise_agent"} else "supervisor"
        selected_models = ontology_registry.resolve_model_ids(
            decision.selected_ontology_models,
            agent_id=agent_id,
        )
        try:
            ontology_context, _ = clinical_context(user_id, agent_id, selected_models, decision.question_subject)
        except OntologyMemoryNotReady:
            return memory_unavailable(selected_models, config)
        resolved_user_input = latest_user.content.strip() if latest_user is not None else ""
        turn_update = {
            "privacy_context": identity,
            "route_type": route_type,
            "selected_ontology_models": selected_models,
            "ontology_context": ontology_context,
            "question_subject": decision.question_subject,
            "resolved_user_input": resolved_user_input,
            # Each user turn starts with a fresh lookup; never inherit another turn's values.
            "biomarker_results": biomarkers.fetch(user_id, decision.biomarker_request, [])
            if route_type in {"health_rag", "biomarkers"} and decision.question_subject == "current_user" else [],
        }

        if route_type != "direct_answer":
            return turn_update

        answer_messages = [
            SystemMessage(
                content=system(agent_descriptions=agent_descriptions).format(
                    memory=general_memory,
                    profile=json.dumps(user_profile, ensure_ascii=False) if user_profile else "No saved profile.",
                    personal_instructions=ontology_context,
                )
                + preferences_from_config(config).response_style()
            )
        ]
        answer_messages.extend(dialogue)
        response = model_call(model).invoke(answer_messages, config=config)
        return {"messages": [response], **turn_update}

    def update_memory(state: SupervisorState, config: RunnableConfig, store: BaseStore):
        """Save supported assertions from this turn, then answer using the actual result."""
        config = privacy_config(config, state.get("privacy_context"))
        namespace = ("memory", config["configurable"]["user_id"])
        existing = {item.key: item.value for item in store.search(namespace)}
        latest_user = _latest_user_message(state["messages"])
        source_text = str(latest_user.content) if latest_user is not None else ""
        saved = []
        if source_text.strip():
            result = memory_extractor.invoke([
                SystemMessage(content=MEMORY_UPDATE_INSTRUCTION.format(time=datetime.now().isoformat())),
                HumanMessage(content=json.dumps({"latest_user_message": source_text, "existing_memories": existing}, ensure_ascii=False)),
            ], config=config)
            approved = []
            for memory in result.memories:
                if not memory.content.strip() or (memory.memory_id is not None and memory.memory_id not in existing):
                    continue
                verification = model_call(model, MemoryVerification, strict=True).invoke([
                    SystemMessage(content=MEMORY_VERIFICATION_INSTRUCTION),
                    HumanMessage(content=json.dumps({"latest_user_message": source_text,
                                                    "existing_records": existing,
                                                    "proposed_update": memory.model_dump()}, ensure_ascii=False)),
                ], config=config)
                if verification.supported:
                    approved.append(memory)
            for memory in approved:
                if any(item.get("content") == memory.content for item in existing.values()):
                    continue
                key = memory.memory_id or str(uuid.uuid4())
                record = {"content": memory.content, "source_message_id": latest_user.id}
                store.put(namespace, key, record)
                existing[key] = record
                saved.append(record)
        response = model_call(model).invoke([
            SystemMessage(content="Answer the user's latest message using only their stored memories below. "
                          "Acknowledge only the facts listed in saved_this_turn as saved. If that list is empty, "
                          "do not claim you saved anything. For recall, answer from stored_memories; if absent, say you do not know."
                          + preferences_from_config(config).response_style()),
            HumanMessage(content=json.dumps({"latest_user_message": source_text, "stored_memories": list(existing.values()),
                                            "saved_this_turn": saved}, ensure_ascii=False)),
        ], config=config)
        return {"messages": [response]}


    def update_profile(state: SupervisorState, config: RunnableConfig, store: BaseStore):
        """Update this user's profile only from explicit assertions in the latest message."""
        config = privacy_config(config, state.get("privacy_context"))
        namespace = ("profile", config["configurable"]["user_id"])
        existing = store.search(namespace)
        profile = existing[0].value if existing else None
        latest_user = _latest_user_message(state["messages"])
        source_text = str(latest_user.content) if latest_user is not None else ""
        saved = False
        if source_text.strip():
            result = profile_extractor.invoke([
                SystemMessage(content=PROFILE_UPDATE_INSTRUCTION.format(time=datetime.now().isoformat())),
                HumanMessage(content=json.dumps({"latest_user_message": source_text, "existing_profile": profile}, ensure_ascii=False)),
            ], config=config)
            if result.profile is not None:
                updated = result.profile.model_dump(mode="json")
                previous = profile or {}
                # Verify factual changes, not the full schema's unknown defaults.
                # Clearing an existing value is a change and still needs evidence.
                changes = {key: value for key, value in updated.items()
                           if value != previous.get(key)
                           and (value not in (None, []) or previous.get(key) not in (None, []))}
                if changes:
                    verification = model_call(model, MemoryVerification, strict=True).invoke([
                        SystemMessage(content=MEMORY_VERIFICATION_INSTRUCTION),
                        HumanMessage(content=json.dumps({"latest_user_message": source_text,
                                                        "existing_records": profile,
                                                        "proposed_update": changes}, ensure_ascii=False)),
                    ], config=config)
                    if verification.supported:
                        store.put(namespace, existing[0].key if existing else "profile", updated)
                        profile, saved = updated, True
        response = model_call(model).invoke([
            SystemMessage(content="Answer the user's latest message from their stored profile below. "
                          "Confirm an update only when saved_this_turn is true. Otherwise do not claim to save anything. "
                          "For recall, answer from stored_profile; if absent, say you do not know."
                          + preferences_from_config(config).response_style()),
            HumanMessage(content=json.dumps({"latest_user_message": source_text, "stored_profile": profile,
                                            "saved_this_turn": saved}, ensure_ascii=False)),
        ], config=config)
        return {"messages": [response], "privacy_context": {
            **(state.get("privacy_context") or {}),
            **{key: (profile or {}).get(key) for key in ("name", "location")},
        }}

    def biomarker_readout(state: SupervisorState, config: RunnableConfig):
        config = privacy_config(config, state.get("privacy_context"))
        results = state.get("biomarker_results") or []
        response = model_call(model).invoke([
            SystemMessage(content=biomarkers.config.readout_instruction
                          + "\n\n" + biomarkers.config.context_instruction
                          + preferences_from_config(config).response_style()),
            HumanMessage(content=json.dumps({"question": state.get("resolved_user_input"),
                                            "lookup_results": biomarkers.model_results(results)})),
        ], config=config)
        return {"messages": [AIMessage(
            content=biomarkers.label_answer(response.content, results, preferences_from_config(config).response_language),
            additional_kwargs={"source": "biomarkers", "biomarker_results": results},
        )]}

    @task
    def assess_health_clarification(question: str, conversation_context: str,
                                  ontology_context: str, response_style: str,
                                  user_id: str, results: list[dict], subject: str,
                                  identity: dict) -> dict[str, Any]:
        """Checkpoint assessment and optional lookup together across clarification resume."""
        def assess(allow_lookup=True):
            return agent_by_name["health_rag"].assess_clarification(
                question, model=model, conversation_context=conversation_context,
                ontology_context=ontology_context, response_style=response_style,
                biomarker_catalog=biomarkers.catalog_context(results) if subject == "current_user" else "",
                biomarker_context=biomarkers.render_context(results),
                allow_biomarker_lookup=allow_lookup and subject == "current_user",
                config=privacy_config({"configurable": {"user_id": user_id}}, identity),
            )
        decision = assess()
        if (subject == "current_user" and decision.biomarker_request and decision.biomarker_request.subject == "current_user"
                and decision.biomarker_request.catalog_match == "matched"
                and decision.biomarker_request.measurement_ids and not decision.urgent_safety_bypass):
            results = biomarkers.fetch(user_id, decision.biomarker_request, results)
            decision = assess(allow_lookup=False)
        return {"decision": decision.model_dump(mode="json"), "biomarkers": results}

    def health_rag_node(
        state: SupervisorState,
        config: RunnableConfig,
        store: BaseStore,
    ) -> dict[str, Any]:
        """Answer medical questions from the local literature corpus."""
        config = privacy_config(config, state.get("privacy_context"))
        messages = state["messages"]
        selected_models = ontology_registry.resolve_model_ids(
            state.get("selected_ontology_models") or [],
            agent_id="health_rag",
        )
        clarification_trace: dict[str, Any] = {}
        subject = state.get("question_subject", "unknown")
        biomarker_results = (state.get("biomarker_results") or []) if subject == "current_user" else []

        def send(
            text: str,
            trace: dict[str, Any] | None = None,
            *,
            resolved_input: str = "",
        ) -> dict[str, Any]:
            text = (text or "").strip() or health_messages.empty_response
            text = biomarkers.label_answer(text, biomarker_results, preferences.response_language)
            extra = {"source": "health_rag", "biomarker_results": biomarker_results}
            if clarification_trace:
                extra["clarification"] = clarification_trace
            if trace:
                extra["workflow_trace"] = [trace]
                extra["presentation"] = health_presentation(text, trace.get("citations", []), preferences.response_language)
            update: dict[str, Any] = {
                "messages": [AIMessage(content=text, additional_kwargs=extra)],
                "selected_ontology_models": selected_models,
                "biomarker_results": biomarker_results,
            }
            if resolved_input:
                update["resolved_user_input"] = resolved_input
            return update

        # Latest user question
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        question = human_messages[-1].content.strip() if human_messages else ""
        if not question:
            question = "Please answer the user's latest health question."
        health_context = _recent_health_context(messages)
        preferences = preferences_from_config(config)
        # Prepare error wording before assessment so its failure needs no further model call.
        policy = response_policy(preferences.response_language)
        user_id = config["configurable"]["user_id"]
        try:
            ontology_context, ontology_memory_context = clinical_context(user_id, "health_rag", selected_models, subject)
        except OntologyMemoryNotReady:
            return memory_unavailable(selected_models, config)
        agent = agent_by_name.get("health_rag")
        if agent is None:
            return send(policy.clarification_unavailable_text, resolved_input=question)

        health_messages = localize_messages(agent.config.messages, preferences.response_language)

        get_stream_writer()({"event": "health_processing_started"})
        try:
            assessed = assess_health_clarification(
                question, health_context, ontology_context, preferences.response_style(),
                user_id, biomarker_results, subject, state.get("privacy_context") or {},
            ).result()
            biomarker_results = assessed["biomarkers"]
            clarification_decision = DisambiguationDecision.model_validate(assessed["decision"])
        except ClarificationAssessmentError as error:
            clarification_trace.update(status="error", selected_ontology_models=selected_models,
                                       error_type=type(error.__cause__ or error).__name__)
            return send(policy.clarification_unavailable_text)
        clarification_trace.update(
            status="assessed", selected_ontology_models=selected_models, question_subject=subject,
            **clarification_decision.model_dump(mode="json"),
        )
        if clarification_decision.needs_clarification:
            prompt = clarification_decision.clarification_question.strip()
            clarification = interrupt(biomarkers.label_answer(prompt, biomarker_results, preferences.response_language))
            if isinstance(clarification, list):
                clarification = clarification[0] if clarification else ""
            clarification_text = str(clarification or "").strip()
            if not clarification_text:
                return send(prompt, resolved_input=question)
            clarification_trace["status"] = "clarified"
            question = f"Original question: {question}\nClarification asked: {prompt}\nUser clarification: {clarification_text}"

        standalone_question = (
            f"{health_context}\n\nCurrent user question: {question}"
            if health_context
            else question
        )

        namespace = ("health_rag", user_id, "qa_history")
        style_key = agent.cache_scope(preferences.response_style(), ontology_context, ontology_memory_context)
        question_hash = hashlib.sha256(json.dumps([health_context, question]).encode("utf-8")).hexdigest()
        # The new key keeps replies independent of legacy records that stored dialogue.
        summary_key = f"answer:{style_key}:{question_hash}"

        allow_cache = not biomarker_results and not clarification_decision.urgent_safety_bypass
        exact_item = answer_cache.get(namespace, summary_key) if allow_cache else None
        if allow_cache and exact_item is None:
            try:
                exact_item = store.get(namespace, summary_key)
            except PostgresError:
                logger.warning("Could not read a saved HealthRAG reply.")
        if exact_item is not None:
            value = getattr(exact_item, "value", {})
            if isinstance(value, dict) and value.get("summary"):
                return send(
                    f"{health_messages.cached_answer_prefix} {value['summary']}",
                    {
                        "stage": "health_rag",
                        "tool_name": "health_rag_tool",
                        "memory_hit": True,
                        "memory_lookup": "exact",
                        "memory_namespace": list(getattr(exact_item, "namespace", namespace)),
                        "memory_key": getattr(exact_item, "key", summary_key),
                        "retrieval_used": False,
                        "summary_stored": False,
                        "query": standalone_question,
                        "citations": value.get("citations", []),
                    },
                    resolved_input=question,
                )

        semantic_items = [] if not allow_cache else answer_cache.search(
            namespace, query=question,
            filter={"type": "health_rag_summary", "response_style": style_key}, limit=1,
        )
        if semantic_items:
            semantic_item = semantic_items[0]
            semantic_value = getattr(semantic_item, "value", {})
            semantic_score = float(getattr(semantic_item, "score", 0.0) or 0.0)
            normalized_user_question = " ".join(re.sub(r"[^a-z0-9]+", " ", question.lower()).split())
            stored_user_question = str(semantic_value.get("user_question") or "").strip()
            normalized_stored_user_question = " ".join(
                re.sub(r"[^a-z0-9]+", " ", stored_user_question.lower()).split()
            )
            current_tokens = set(normalized_user_question.split())
            stored_tokens = set(normalized_stored_user_question.split())
            token_overlap = (
                len(current_tokens & stored_tokens) / max(len(current_tokens), len(stored_tokens), 1)
            )
            if (
                isinstance(semantic_value, dict)
                and semantic_value.get("summary")
                and semantic_score >= agent.config.cache.semantic_score_threshold
                and token_overlap >= agent.config.cache.token_overlap_threshold
            ):
                return send(
                    f"{health_messages.cached_answer_prefix} {semantic_value['summary']}",
                    {
                        "stage": "health_rag",
                        "tool_name": "health_rag_tool",
                        "memory_hit": True,
                        "memory_lookup": "semantic",
                        "memory_namespace": list(getattr(semantic_item, "namespace", namespace)),
                        "memory_key": getattr(semantic_item, "key", ""),
                        "retrieval_used": False,
                        "summary_stored": False,
                        "query": standalone_question,
                        "citations": semantic_value.get("citations", []),
                    },
                    resolved_input=question,
                )

        result = agent.agent.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "conversation_context": health_context,
                "ontology_context": ontology_context,
                "ontology_memory_context": ontology_memory_context,
                "biomarker_context": biomarkers.render_context(biomarker_results),
                "standalone_question": question,
                "needs_query_rewrite": clarification_trace.get("status") == "clarified",
                "urgent_safety_bypass": clarification_decision.urgent_safety_bypass,
                "has_emergency_contact": preferences.emergency_contact is not None,
                "response_style": preferences.response_style(),
                "response_language": preferences.response_language,
            },
            config={**config, "recursion_limit": 10},
        )
        answer_message = result["messages"][-1] if isinstance(result, dict) else None
        answer_text = getattr(answer_message, "content", "") if answer_message is not None else ""
        citations = result.get("citations", []) if isinstance(result, dict) else []

        summary_text = (
            str(answer_text or "").strip()
            or health_messages.no_evidence
        )
        if clarification_decision.urgent_safety_bypass and preferences.emergency_contact:
            contact = preferences.emergency_contact
            if EMERGENCY_CONTACT_MARKER in summary_text:
                summary_text = summary_text.replace(EMERGENCY_CONTACT_MARKER, contact)
            else:
                summary_text = f"{summary_text}\n\n{contact}"
        summary_stored = False
        if citations and answer_text and allow_cache:
            saved_answer = {"summary": summary_text, "citations": citations}
            try:
                store.put(namespace, summary_key, saved_answer, index=False)
            except PostgresError:
                logger.warning("Could not save a HealthRAG reply.")
            answer_cache.put(
                namespace,
                summary_key,
                {
                    **saved_answer,
                    "user_question": question,
                    "type": "health_rag_summary",
                    "response_style": style_key,
                },
            )
            summary_stored = True

        return send(
            summary_text,
            {
                "stage": "health_rag",
                "tool_name": "health_rag_tool",
                "memory_hit": False,
                "memory_lookup": "miss",
                "memory_namespace": list(namespace),
                "memory_key": summary_key,
                "retrieval_used": result.get("retrieval_used", True) if isinstance(result, dict) else False,
                "summary_stored": summary_stored,
                "query": standalone_question,
                "retrieval_query": result.get("retrieval_query", "") if isinstance(result, dict) else "",
                "citations": citations,
            },
            resolved_input=question,
        )

    def schedule_ontology_memory_update(
        state: SupervisorState,
        config: RunnableConfig,
    ) -> dict[str, str]:
        """Queue ontology extraction without adding latency or a user-visible message."""

        latest_user = _latest_user_message(state.get("messages") or [])
        source_text = str(state.get("resolved_user_input") or "").strip()
        if not source_text and latest_user is not None:
            source_text = str(latest_user.content or "").strip()
        configurable = config.get("configurable") or {}
        user_id = str(configurable.get("user_id") or "")
        thread_id = str(configurable.get("thread_id") or "")
        if latest_user is not None and getattr(latest_user, "id", None):
            source_message_id = str(latest_user.id)
        else:
            digest = hashlib.sha256(f"{thread_id}|{source_text}".encode("utf-8")).hexdigest()[:20]
            source_message_id = f"message:{digest}"

        scheduled = bool(user_id and thread_id) and ontology_memory.schedule(
            user_id=user_id,
            thread_id=thread_id,
            source_message_id=source_message_id,
            source_text=source_text,
            model_ids=state.get("selected_ontology_models") or [],
            privacy_context=state.get("privacy_context") or {},
        )
        return {"memory_update_status": "scheduled" if scheduled else "skipped"}

    def route_message(
        state: SupervisorState,
    ) -> Literal[
        "schedule_ontology_memory_update",
        "update_profile",
        "update_memory",
        "health_rag",
        "biomarker_readout",
        "phyxio_exercise_agent",
    ]:

        """
        Use this tool to decide where to route the user’s request:
        - 'health_rag' for ANY medical/health-related question
        - 'user_profile' for stable personal facts
        - 'general_memory' for experiences / events / feelings
        - 'phyxio_exercise_agent' when the user asks to show, open, list, calibrate, start, or review exercise history/progress.
        """

        route_type = state.get("route_type")
        if route_type == "direct_answer" or not route_type:
            return "schedule_ontology_memory_update"
        if route_type == "user_profile":
            return "update_profile"
        if route_type == "general_memory":
            return "update_memory"
        if route_type == "health_rag":
            return "health_rag"
        if route_type == "biomarkers":
            return "biomarker_readout"
        if route_type == "phyxio_exercise_agent":
            return "phyxio_exercise_agent"
        raise ValueError(f"Unknown supervisor route_type: {route_type}")
            



    store, checkpointer = get_postgres_memory()
    answer_cache = InMemoryStore(index={
        "dims": 1536, "embed": DeterministicMemoryEmbeddings(), "fields": ["user_question"],
    })
    ontology_registry = get_ontology_registry()
    ontology_memory = OntologyMemoryService(model=model, store=store, registry=ontology_registry)
    agents, agent_descriptions = _load_agents()
    agent_by_name = {a.name: a for a in agents}
    # Create the graph + all nodes
    builder = StateGraph(SupervisorState)

    # Define the flow of the memory extraction process
    builder.add_node(supervisor)
    builder.add_node(update_profile)
    builder.add_node(update_memory)

    for agent in agents:
        if agent.name != "health_rag":
            builder.add_node(agent.name, agent)

    builder.add_node("health_rag", health_rag_node)
    builder.add_node("biomarker_readout", biomarker_readout)
    builder.add_node("schedule_ontology_memory_update", schedule_ontology_memory_update)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_message)
    builder.add_edge("update_profile", "schedule_ontology_memory_update")
    builder.add_edge("update_memory", "schedule_ontology_memory_update")
    builder.add_edge("health_rag", "schedule_ontology_memory_update")
    builder.add_edge("biomarker_readout", "schedule_ontology_memory_update")
    builder.add_edge("phyxio_exercise_agent", "schedule_ontology_memory_update")
    builder.add_edge("schedule_ontology_memory_update", END)


    # We compile the graph with the checkpointer and store
    graph = builder.compile(checkpointer=checkpointer, store=store)
    return graph
