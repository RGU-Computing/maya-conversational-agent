from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, Literal, Mapping, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from maya.framework.conversation.personalization import preferences_from_config
from maya.framework.conversation.history import recent_dialogue
from langchain_core.tools import BaseTool, tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from maya.framework.integrations.phyxio.bridge import get_phyxio_service
from .sessions import get_latest_exercise_session, load_session_service
from .phyxio_sessions import PhyxioSessionProvider
from maya.framework.integrations.models import get_chat_model

load_dotenv()

AGENT_ID = "phyxio_exercise_agent"
DEFAULT_TOOLS: Sequence[str] = (
    "show_exercises",
    "calibrate_exercises",
    "start_exercises",
    "list_exercises",
    "get_exercise_history",
    "get_latest_exercise_session",
)
ROUTINE_ID = os.getenv("MAYA_PHYXIO_ROUTINE_ID", "").strip()

ExerciseToolName = Literal[
    "show_exercises",
    "calibrate_exercises",
    "start_exercises",
    "list_exercises",
    "get_exercise_history",
    "get_latest_exercise_session",
    "none",
]


class ExerciseActionDecision(BaseModel):
    """Validated semantic decision made before any Phyxio action is executed."""

    action: ExerciseToolName = Field(
        description="The single Phyxio tool to execute, or none when no action should run.",
    )
    session_subject: Literal["current_user", "other", "unknown"] = Field(
        default="unknown", description="Whose session is being reviewed; resolve from current dialogue, never assume another person's session belongs to the participant."
    )
    safety_hold: bool = Field(
        description="True when the user's current context makes executing an exercise action unsafe.",
    )
    needs_clarification: bool = Field(
        description="True when one concise follow-up is required before selecting an action.",
    )
    reason: str = Field(
        description="A brief internal explanation based on the complete meaning of the request.",
    )
    user_response: str = Field(
        description=(
            "A concise response in the user's language for a safety hold, clarification, or out-of-scope request. "
            "Do not claim that a tool ran."
        ),
    )


class ExerciseAgentResponse(BaseModel):
    """User-facing response grounded in the actual Phyxio tool result."""

    response: str = Field(
        description="A concise, truthful response in the user's language based on the supplied tool result.",
    )


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _normalize_routine_payload(data: Any) -> dict[str, Any]:
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    if isinstance(data, str):
        return json.loads(data)
    if isinstance(data, dict):
        return data
    raise TypeError(f"Unexpected routine list type: {type(data).__name__}")


def _routine_candidates(routine_id: str | None = None) -> list[str]:
    candidates: list[str] = []
    for value in (routine_id, ROUTINE_ID):
        text = str(value or "").strip()
        if text and text not in candidates:
            candidates.append(text)
    return candidates


def _resolve_routine_id(routine_id: str | None = None, data: dict[str, Any] | None = None) -> str:
    if data is None:
        data = _get_routine_list()

    error = data.get("error")
    if error:
        raise RuntimeError(str(error))

    routines = data.get("routines") or {}
    if not isinstance(routines, dict) or not routines:
        raise RuntimeError("No assigned Phyxio routines were returned.")

    for candidate in _routine_candidates(routine_id):
        if candidate in routines:
            return candidate
        try:
            numeric_candidate = str(int(candidate))
        except (TypeError, ValueError):
            numeric_candidate = ""
        if numeric_candidate and numeric_candidate in routines:
            return numeric_candidate

    return str(next(iter(routines.keys())))


def _call_service(action: str, routine_id: str | None = None) -> str:
    try:
        service = get_phyxio_service()
    except Exception as exc:
        return f"Phyxio service unavailable: {exc}"

    try:
        resolved_routine_id = _resolve_routine_id(routine_id)
        if action == "show":
            return _to_text(service.show_routine(resolved_routine_id))
        if action == "calibrate":
            return _to_text(service.calibrate_routine(resolved_routine_id))
        if action == "start":
            return _to_text(service.run_routine(resolved_routine_id))
    except Exception as exc:
        return f"Phyxio action '{action}' failed: {exc}"

    return f"Unknown Phyxio action '{action}'."


def _get_routine_list() -> dict[str, Any]:
    service = get_phyxio_service()
    if hasattr(service, "get_routine_list"):
        data = service.get_routine_list()
    else:
        data = service._phyxio.routine.get_list()

    return _normalize_routine_payload(data)


def _format_goal(value: Any, unit: str) -> str:
    text = str(value or "").strip()
    if not text or text in {"0", "None", "null"}:
        return ""
    return f"{text} {unit}"


def _brief_description(value: Any, max_chars: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    first_sentence = text.split(". ", 1)[0].rstrip(".")
    if len(first_sentence) <= max_chars:
        return first_sentence
    return first_sentence[: max_chars - 1].rstrip() + "."


def _parse_page(value: Any, first_page: int) -> int:
    text = str(value).strip() if value is not None else ""
    if not text:
        return first_page
    try:
        page = int(text)
    except ValueError:
        return first_page
    return max(page, first_page)


def _get_routine(data: dict[str, Any], routine_id: str | None = None) -> tuple[str, dict[str, Any]]:
    resolved_routine_id = _resolve_routine_id(routine_id, data)
    routines = data.get("routines") or {}
    routine = routines.get(resolved_routine_id)
    if not isinstance(routine, dict):
        raise RuntimeError(f"Routine {resolved_routine_id} is not available.")
    return resolved_routine_id, routine


def _format_spoken_routine_summary(data: dict[str, Any], routine_id: str | None = None) -> str:
    error = data.get("error")
    if error:
        return f"I couldn't fetch your exercise list: {error}"

    exercises = data.get("exercises") or {}
    try:
        resolved_routine_id, routine = _get_routine(data, routine_id)
    except Exception as exc:
        return f"I couldn't fetch your exercise list: {exc}"

    routine_name = str(routine.get("name") or f"Routine {resolved_routine_id}").strip()
    routine_exercises = routine.get("exercises") or []
    if not routine_exercises:
        return f"{routine_name} does not currently list any exercises."

    count = len(routine_exercises)
    plural = "s" if count != 1 else ""
    lines = [f"Your current routine is {routine_name}, with {count} exercise{plural}."]
    for index, item in enumerate(routine_exercises, start=1):
        exercise_id = str(item.get("id") or "").strip()
        exercise = exercises.get(exercise_id) or exercises.get(item.get("id")) or {}
        name = str(exercise.get("name") or f"Exercise {exercise_id or index}").strip()
        goals = [
            goal
            for goal in (
                _format_goal(item.get("iterations"), "repetitions"),
                _format_goal(item.get("runtime"), "seconds max"),
            )
            if goal
        ]
        description = _brief_description(exercise.get("description"))
        details = ", ".join(goals)
        line = f"{index}. {name}"
        if details:
            line += f": {details}"
        if description:
            line += f". {description}"
        lines.append(line)
    return "\n".join(lines)


def _format_duration(seconds: Any) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return ""
    if total <= 0:
        return ""
    minutes, remainder = divmod(total, 60)
    if minutes and remainder:
        return f"{minutes} min {remainder} sec"
    if minutes:
        return f"{minutes} min"
    return f"{remainder} sec"


def _format_session_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unknown date"
    return text.replace("T", " ").replace("Z", "").split("+", 1)[0]


def _exercise_lookup(data: dict[str, Any]) -> dict[str, str]:
    exercises = data.get("exercises") or {}
    if not isinstance(exercises, dict):
        return {}
    lookup: dict[str, str] = {}
    for exercise_id, exercise in exercises.items():
        if isinstance(exercise, dict):
            name = str(exercise.get("name") or "").strip()
            if name:
                lookup[str(exercise_id)] = name
    return lookup


def _format_exercise_history(
    session_data: dict[str, Any],
    routine_data: dict[str, Any] | None = None,
    max_sessions: int = 3,
) -> str:
    error = session_data.get("error")
    if error:
        return f"I couldn't fetch your exercise history: {error}"

    sessions = session_data.get("sessions") or []
    if not isinstance(sessions, list) or not sessions:
        return "I couldn't find any previous Phyxio exercise sessions yet."

    exercise_names = _exercise_lookup(routine_data or {})
    page = session_data.get("page")
    prefix = "Recent exercise sessions"
    if page is not None:
        prefix += f" (page {page})"

    lines = [f"{prefix}:"]
    for session in sessions[:max_sessions]:
        if not isinstance(session, dict):
            continue
        date = _format_session_date(session.get("date"))
        routine_id = str(session.get("routine_id") or "").strip()
        results = session.get("results") or []
        if not isinstance(results, list):
            results = []

        statuses = Counter(str(result.get("status") or "unknown")
                           for result in results if isinstance(result, dict))
        header = f"- {date}"
        if routine_id:
            header += f", routine {routine_id}"
        if statuses:
            header += ": " + ", ".join(f"{count} {status}" for status, count in statuses.items())
        lines.append(header)

        for result in results[:4]:
            if not isinstance(result, dict):
                continue
            exercise_id = str(result.get("exercise_id") or "").strip()
            name = exercise_names.get(exercise_id) or f"Exercise {exercise_id or '?'}"
            status = str(result.get("status") or "unknown").replace("_", " ")
            goal_iter = result.get("goal_iter")
            done_iter = result.get("done_iter")
            elapsed = _format_duration(result.get("elapsed_time"))
            parts = [status]
            if goal_iter is not None or done_iter is not None:
                parts.append(f"{done_iter or 0}/{goal_iter or 0} reps")
            if elapsed:
                parts.append(elapsed)
            lines.append(f"  {name}: {', '.join(parts)}")

        if len(results) > 4:
            lines.append(f"  Plus {len(results) - 4} more result(s).")

    if len(sessions) > max_sessions:
        lines.append(f"I found {len(sessions)} sessions on this page; showing the latest {max_sessions}.")

    return "\n".join(lines)


@tool("show_exercises")
def show_exercises(routine_id: str = "") -> str:
    """Show the configured exercise routine on the mirror."""

    return _call_service("show", routine_id)


@tool("calibrate_exercises")
def calibrate_exercises(routine_id: str = "") -> str:
    """Calibrate the configured exercise routine on the mirror."""

    return _call_service("calibrate", routine_id)


@tool("start_exercises")
def start_exercises(routine_id: str = "") -> str:
    """Start the configured exercise routine on the mirror."""

    return _call_service("start", routine_id)


@tool("list_exercises")
def list_exercises(routine_id: str = "") -> str:
    """List the exercises in the user's configured Phyxio routine."""

    try:
        return _format_spoken_routine_summary(_get_routine_list(), routine_id)

    except Exception as exc:
        return f"I couldn't fetch your exercise list: {exc}"


@tool("get_exercise_history")
def get_exercise_history(config: RunnableConfig, page: str = "") -> str:
    """Summarize recent Phyxio exercise session history."""

    try:
        service = load_session_service()
        if not isinstance(service.provider, PhyxioSessionProvider):
            return service.config.fallback_text
        user_id = str(config.get("configurable", {}).get("user_id") or "")
        data, catalog = service.provider.fetch_page(user_id, _parse_page(page, service.config.first_page))
        return _format_exercise_history(data, catalog)
    except Exception as exc:
        return f"I couldn't fetch your exercise history: {exc}"

_TOOL_REGISTRY: Mapping[str, BaseTool] = {
    "show_exercises": show_exercises,
    "calibrate_exercises": calibrate_exercises,
    "start_exercises": start_exercises,
    "list_exercises": list_exercises,
    "get_exercise_history": get_exercise_history,
    "get_latest_exercise_session": get_latest_exercise_session,
}


def _require_prompt(spec: Dict[str, Any], field_name: str) -> str:
    prompt = spec.get(field_name)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"`{field_name}` missing or empty for agent '{AGENT_ID}'.")
    return prompt.strip()


def _normalize_tools(raw: Sequence[str] | str) -> Sequence[str]:
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(tool_id) for tool_id in raw)


def _include_default_tools(tool_ids: Sequence[str]) -> Sequence[str]:
    ordered = list(tool_ids)
    for tool_id in DEFAULT_TOOLS:
        if tool_id not in ordered:
            ordered.append(tool_id)
    return tuple(ordered)


def _resolve_tools(tool_ids: Sequence[str]) -> Sequence[BaseTool]:
    resolved: list[BaseTool] = []
    missing: list[str] = []
    for ident in tool_ids:
        tool_obj = _TOOL_REGISTRY.get(ident)
        if tool_obj is None:
            missing.append(ident)
            continue
        resolved.append(tool_obj)
    if missing:
        raise RuntimeError(f"Unknown tools for '{AGENT_ID}': {', '.join(missing)}")
    return resolved


def _invoke_registered_tool(tool_name: str, config=None) -> Any:
    return _TOOL_REGISTRY[tool_name].invoke({}, config=config)


def _extract_latest_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type")
            if role in {"human", "user"}:
                text = msg.get("content")
                return _to_text(text).strip() if text is not None else ""
        else:
            role = getattr(msg, "type", None)
            if role in {"human", "user"}:
                text = getattr(msg, "content", "")
                return _to_text(text).strip()
    return ""


class _PhyxioExerciseNode:
    def __init__(
        self,
        *,
        decision_model: Any,
        response_model: Any,
        decision_prompt: str,
        response_prompt: str,
        tools: Sequence[BaseTool],
    ):
        self.name = AGENT_ID
        self._decision_model = decision_model
        self._response_model = response_model
        self._decision_prompt = decision_prompt
        self._response_prompt = response_prompt
        self._tool_catalog = "\n".join(
            f"- {tool_obj.name}: {tool_obj.description or 'No description supplied.'}"
            for tool_obj in tools
        )

    @staticmethod
    def _normalise_decision(decision: ExerciseActionDecision) -> ExerciseActionDecision:
        """Enforce schema consistency without interpreting the user's language."""

        if decision.safety_hold:
            return decision.model_copy(
                update={"action": "none", "needs_clarification": False}
            )
        if decision.needs_clarification:
            return decision.model_copy(update={"action": "none"})
        return decision

    @staticmethod
    def _forward_config(config: Any) -> dict[str, Any] | None:
        if not isinstance(config, dict):
            return None

        invoke_config: dict[str, Any] = {}
        configurable = config.get("configurable")
        if isinstance(configurable, dict):
            forwarded = {
                key: configurable[key]
                for key in ("thread_id", "user_id", "preferences")
                if configurable.get(key) is not None
            }
            if forwarded:
                invoke_config["configurable"] = forwarded
        if "recursion_limit" in config:
            invoke_config["recursion_limit"] = config["recursion_limit"]
        return invoke_config or None

    @staticmethod
    def _result_message(text: str, tool_call_id: str | None, extra=None) -> dict[str, list[Any]]:
        if tool_call_id:
            return {
                "messages": [
                    ToolMessage(
                        content=text,
                        tool_call_id=tool_call_id,
                        name="RouteState",
                        additional_kwargs={"source": AGENT_ID, **(extra or {})},
                    )
                ]
            }
        return {
            "messages": [
                AIMessage(content=text, additional_kwargs={"source": AGENT_ID, **(extra or {})})
            ]
        }

    @staticmethod
    def _extract_tool_call_id(messages: Any) -> str | None:
        """Match only a pending call in this turn, never historical routing calls."""
        if not isinstance(messages, list) or not messages:
            return None
        answered: set[str] = set()
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get("role") or msg.get("type")
                response_id = msg.get("tool_call_id")
                tool_calls = msg.get("tool_calls") or []
            else:
                role = getattr(msg, "type", None)
                response_id = getattr(msg, "tool_call_id", None)
                tool_calls = getattr(msg, "tool_calls", None) or []
            if role in {"human", "user"}:
                return None
            if role == "tool":
                if response_id:
                    answered.add(str(response_id))
                continue
            if role in {"ai", "assistant"}:
                for tool_call in tool_calls:
                    call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
                    if call_id and str(call_id) not in answered:
                        return str(call_id)
                return None
        return None

    def __call__(self, state: Dict[str, Any], config=None, store: Any = None):
        del store

        messages = state.get("messages", [])
        user_text = _extract_latest_user_text(messages)
        ontology_context = str(state.get("ontology_context") or "").strip()
        tool_call_id = self._extract_tool_call_id(messages)
        if not user_text:
            return self._result_message("Exercise request is empty.", tool_call_id)

        invoke_config = self._forward_config(config)
        response_style = preferences_from_config(config).response_style()

        try:
            decision = self._decision_model.invoke(
                [
                    SystemMessage(
                        content=(
                            f"{self._decision_prompt}\n\n"
                            f"{ontology_context}\n\n"
                            f"AVAILABLE PHYXIO TOOLS\n{self._tool_catalog}"
                            f"{response_style}"
                        ).strip()
                    ),
                    *(recent_dialogue(messages) or [HumanMessage(content=user_text)]),
                ],
                config=invoke_config,
            )
        except Exception as exc:
            return self._result_message(f"Exercise decision failed: {exc}", tool_call_id)

        if not isinstance(decision, ExerciseActionDecision):
            return self._result_message(
                "Exercise decision failed validation.",
                tool_call_id,
            )

        decision = self._normalise_decision(decision)
        if decision.action == "none":
            answer = decision.user_response.strip() or decision.reason.strip()
            return self._result_message(
                answer or "The exercise request needs more information.",
                tool_call_id,
            )

        if decision.action in {"get_latest_exercise_session", "get_exercise_history"} and decision.session_subject != "current_user":
            return self._result_message(load_session_service().config.subject_clarification, tool_call_id)

        service_result = _invoke_registered_tool(decision.action, config=invoke_config)
        extra = {"executed_tool": decision.action}
        fallback = _to_text(service_result)
        if isinstance(service_result, dict):
            extra["exercise_session"] = service_result
            fallback = service_result["fallback_text"]
        try:
            response = self._response_model.invoke(
                [
                    SystemMessage(content=self._response_prompt + response_style),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "user_request": user_text,
                                "decision": decision.model_dump(),
                                "executed_tool": decision.action,
                                "tool_result": service_result,
                            },
                            ensure_ascii=False,
                        )
                    ),
                ],
                config=invoke_config,
            )
        except Exception:
            return self._result_message(fallback, tool_call_id, extra)

        if not isinstance(response, ExerciseAgentResponse):
            return self._result_message(fallback, tool_call_id, extra)
        answer = response.response.strip() or fallback
        return self._result_message(answer, tool_call_id, extra)


def build(spec: Dict[str, Any] | None = None):
    spec = spec or {}

    decision_prompt = _require_prompt(spec, "prompt")
    response_prompt = _require_prompt(spec, "response_prompt")
    tool_ids = _include_default_tools(_normalize_tools(spec.get("tools") or DEFAULT_TOOLS))
    tools = _resolve_tools(tool_ids)

    llm = get_chat_model()
    return _PhyxioExerciseNode(
        decision_model=llm.with_structured_output(ExerciseActionDecision, strict=True),
        response_model=llm.with_structured_output(ExerciseAgentResponse, strict=True),
        decision_prompt=decision_prompt,
        response_prompt=response_prompt,
        tools=tools,
    )
