from __future__ import annotations
from typing_extensions import Literal, TypedDict, NotRequired, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps

class SupervisorState(TypedDict):
    # REQUIRED by prebuilt agents/supervisor in LangGraph v1
    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: RemainingSteps

    # Optional
    active_agent: NotRequired[str]
    selected_ontology_models: NotRequired[list[str]]
    ontology_context: NotRequired[str]
    question_subject: NotRequired[Literal["current_user", "other", "general", "unknown"]]
    biomarker_results: NotRequired[list[dict]]
    resolved_user_input: NotRequired[str]
    memory_update_status: NotRequired[str]
    route_type: NotRequired[
        Literal[
            "direct_answer",
            "user_profile",
            "general_memory",
            "health_rag",
            "biomarkers",
            "phyxio_exercise_agent",
        ]
    ]
