from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import patch

from maya.tests.model_stubs import model_call_stub

from langchain_core.messages import HumanMessage

from maya.agents.phyxio_exercise_agent.agent import (
    ExerciseActionDecision,
    ExerciseAgentResponse,
    _PhyxioExerciseNode,
    _routine_candidates,
    build,
    calibrate_exercises,
    get_exercise_history,
    list_exercises,
    show_exercises,
    start_exercises,
)


TOOLS = (
    show_exercises,
    calibrate_exercises,
    start_exercises,
    list_exercises,
    get_exercise_history,
)


class RecordingModel:
    def __init__(self, response: Any = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[tuple[Any, Any]] = []

    def invoke(self, value: Any, config: Any = None) -> Any:
        self.calls.append((value, config))
        if self.error is not None:
            raise self.error
        return self.response


def _decision(
    action: str,
    *,
    safety_hold: bool = False,
    needs_clarification: bool = False,
    user_response: str = "",
) -> ExerciseActionDecision:
    return ExerciseActionDecision(
        action=action,
        safety_hold=safety_hold,
        needs_clarification=needs_clarification,
        reason="Structured semantic assessment.",
        user_response=user_response,
    )


def _node(
    decision: ExerciseActionDecision | None = None,
    *,
    decision_error: Exception | None = None,
    response: ExerciseAgentResponse | None = None,
) -> tuple[_PhyxioExerciseNode, RecordingModel, RecordingModel]:
    decision_model = RecordingModel(decision, error=decision_error)
    response_model = RecordingModel(
        response or ExerciseAgentResponse(response="Completed from the real service result.")
    )
    return (
        _PhyxioExerciseNode(
            decision_model=decision_model,
            response_model=response_model,
            decision_prompt="Decide semantically.",
            response_prompt="Ground the response in the tool result.",
            tools=TOOLS,
        ),
        decision_model,
        response_model,
    )


class PhyxioStructuredDecisionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("maya.agents.phyxio_exercise_agent.agent.model_call", side_effect=model_call_stub))

    def test_validated_action_executes_exactly_one_selected_tool(self):
        node, decision_model, response_model = _node(_decision("show_exercises"))

        with patch(
            "maya.agents.phyxio_exercise_agent.agent._invoke_registered_tool",
            return_value='{"status":"shown"}',
        ) as invoke_tool:
            result = node(
                {
                    "messages": [HumanMessage(content="Could you put my programme on the mirror?")],
                    "ontology_context": "ExercisePlan prescribes PlannedExerciseSession.",
                },
                config={"configurable": {"thread_id": "thread-1", "user_id": "user-1"}},
            )

        invoke_tool.assert_called_once_with("show_exercises", config={"configurable": {"thread_id": "thread-1", "user_id": "user-1"}})
        self.assertEqual(result["messages"][0].content, "Completed from the real service result.")
        self.assertEqual(len(decision_model.calls), 1)
        self.assertEqual(len(response_model.calls), 1)
        self.assertIn(
            "ExercisePlan prescribes PlannedExerciseSession.",
            decision_model.calls[0][0][0].content,
        )
        response_payload = json.loads(response_model.calls[0][0][1].content)
        self.assertEqual(response_payload["executed_tool"], "show_exercises")
        self.assertEqual(response_payload["tool_result"], '{"status":"shown"}')

    def test_negated_request_is_decided_by_model_not_python_keywords(self):
        node, _, response_model = _node(
            _decision(
                "none",
                user_response="I will not start the routine.",
            )
        )

        with patch(
            "maya.agents.phyxio_exercise_agent.agent._invoke_registered_tool"
        ) as invoke_tool:
            result = node(
                {"messages": [HumanMessage(content="I do not want to start the routine.")]}
            )

        invoke_tool.assert_not_called()
        self.assertEqual(result["messages"][0].content, "I will not start the routine.")
        self.assertEqual(response_model.calls, [])

    def test_safety_hold_overrides_an_inconsistent_action(self):
        node, _, response_model = _node(
            _decision(
                "start_exercises",
                safety_hold=True,
                needs_clarification=True,
                user_response="Do not begin the session; seek appropriate urgent help.",
            )
        )

        with patch(
            "maya.agents.phyxio_exercise_agent.agent._invoke_registered_tool"
        ) as invoke_tool:
            result = node(
                {"messages": [HumanMessage(content="Start it, but I feel seriously unwell now.")]}
            )

        invoke_tool.assert_not_called()
        self.assertEqual(
            result["messages"][0].content,
            "Do not begin the session; seek appropriate urgent help.",
        )
        self.assertEqual(response_model.calls, [])

    def test_clarification_prevents_tool_execution(self):
        node, _, _ = _node(
            _decision(
                "calibrate_exercises",
                needs_clarification=True,
                user_response="Would you like to view, calibrate, or begin the routine?",
            )
        )

        with patch(
            "maya.agents.phyxio_exercise_agent.agent._invoke_registered_tool"
        ) as invoke_tool:
            result = node({"messages": [HumanMessage(content="Can you do that now?")]})

        invoke_tool.assert_not_called()
        self.assertIn("view, calibrate, or begin", result["messages"][0].content)

    def test_decision_failure_fails_closed_without_executing_a_tool(self):
        node, _, _ = _node(decision_error=RuntimeError("model unavailable"))

        with patch(
            "maya.agents.phyxio_exercise_agent.agent._invoke_registered_tool"
        ) as invoke_tool:
            result = node({"messages": [HumanMessage(content="Begin my session.")]})

        invoke_tool.assert_not_called()
        self.assertIn("decision failed", result["messages"][0].content.casefold())

    def test_response_failure_returns_actual_service_result(self):
        decision_model = RecordingModel(_decision("list_exercises"))
        response_model = RecordingModel(error=RuntimeError("formatter unavailable"))
        node = _PhyxioExerciseNode(
            decision_model=decision_model,
            response_model=response_model,
            decision_prompt="Decide semantically.",
            response_prompt="Use the result.",
            tools=TOOLS,
        )

        with patch(
            "maya.agents.phyxio_exercise_agent.agent._invoke_registered_tool",
            return_value="Actual assigned routine details.",
        ):
            result = node({"messages": [HumanMessage(content="What has been assigned to me?")]})

        self.assertEqual(result["messages"][0].content, "Actual assigned routine details.")

    def test_routine_candidates_have_no_fixed_project_fallback(self):
        with patch("maya.agents.phyxio_exercise_agent.agent.ROUTINE_ID", ""):
            self.assertEqual(_routine_candidates(), [])
            self.assertEqual(_routine_candidates("42"), ["42"])

    def test_build_uses_strict_structured_output_for_both_stages(self):
        class StubChatModel:
            def __init__(self):
                self.calls: list[tuple[Any, bool]] = []

            def with_structured_output(self, schema: Any, *, strict: bool):
                self.calls.append((schema, strict))
                return RecordingModel()

        model = StubChatModel()
        spec = {
            "prompt": "Decision policy",
            "response_prompt": "Response policy",
            "tools": [tool_obj.name for tool_obj in TOOLS],
        }

        with patch(
            "maya.agents.phyxio_exercise_agent.agent.get_chat_model",
            return_value=model,
        ):
            built = build(spec)

        self.assertIsInstance(built, _PhyxioExerciseNode)
        self.assertEqual(
            model.calls,
            [
                (ExerciseActionDecision, True),
                (ExerciseAgentResponse, True),
            ],
        )


if __name__ == "__main__":
    unittest.main()
